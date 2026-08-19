# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import stat
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import TransferResult
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.tiering.fs.common import (
    FileSystemControlSpec,
    FileSystemLoadStoreSpec,
    FileSystemLookupSpec,
    ensure_config_file,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

logger = init_logger(__name__)


def _write_all_at(fd: int, view: memoryview, offset: int) -> None:
    written = 0
    while written < len(view):
        n = os.pwrite(fd, view[written:], offset + written)
        if n <= 0:
            raise OSError("pwrite made no progress")
        written += n


def _read_exact_at(fd: int, view: memoryview, offset: int) -> None:
    read = 0
    while read < len(view):
        chunk = os.pread(fd, len(view) - read, offset + read)
        if not chunk:
            raise OSError(
                f"Short read: expected {len(view)} bytes, read {read}"
            )
        end = read + len(chunk)
        view[read:end] = chunk
        read = end


class FileSystemWorkerTransferHandler:
    """Worker-side CPU shard <-> filesystem transfer executor."""

    def __init__(
        self,
        cpu_tensors: list[torch.Tensor],
        rank: int,
        n_read_threads: int = 4,
        n_write_threads: int = 4,
    ) -> None:
        self._cpu_tensors = cpu_tensors
        self._rank = rank
        self._rank_size = sum(int(t.shape[1]) for t in cpu_tensors)
        self._rank_offset = rank * self._rank_size
        self._store_temp_paths: dict[int, list[str]] = {}
        self._control_jobs: dict[int, str] = {}
        self._validated_configs: dict[str, dict[str, Any]] = {}
        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_worker_fs",
        )

    def _select_rank_paths(self, paths: list[str], num_ranks: int) -> list[str]:
        if num_ranks <= 0:
            raise ValueError(f"Invalid num_ranks={num_ranks}")
        if self._rank < 0 or self._rank >= num_ranks:
            raise ValueError(
                f"Worker rank {self._rank} is outside num_ranks={num_ranks}"
            )
        if len(paths) % num_ranks != 0:
            raise ValueError(
                f"Cannot split {len(paths)} paths across {num_ranks} ranks"
            )
        per_rank = len(paths) // num_ranks
        start = self._rank * per_rank
        return paths[start : start + per_rank]

    def _ensure_config(
        self,
        spec: FileSystemLoadStoreSpec | FileSystemLookupSpec | FileSystemControlSpec,
    ) -> None:
        if spec.config_path is None:
            return
        if spec.run_config is None:
            raise ValueError("Filesystem worker config is missing run_config")
        cached = self._validated_configs.get(spec.config_path)
        if cached == spec.run_config:
            return
        ensure_config_file(spec.config_path, spec.run_config)
        self._validated_configs[spec.config_path] = dict(spec.run_config)

    def submit_store(
        self,
        job_id: int,
        src_spec: CPULoadStoreSpec,
        dst_spec: FileSystemLoadStoreSpec,
    ) -> bool:
        self._ensure_config(dst_spec)
        if dst_spec.temp_file_paths is None:
            raise ValueError("Filesystem worker stores require temp file paths")
        # Select this rank's slice of the concatenated per-rank paths.
        num_ranks: int = getattr(dst_spec, "num_ranks", 1)
        my_final = self._select_rank_paths(dst_spec.file_paths, num_ranks)
        my_temp = self._select_rank_paths(dst_spec.temp_file_paths, num_ranks)

        if len(src_spec.block_ids) != len(my_final):
            raise ValueError(
                "CPU block count does not match filesystem path count: "
                f"{len(src_spec.block_ids)} != {len(my_final)}"
            )
        if len(src_spec.block_ids) != len(my_temp):
            raise ValueError(
                "CPU block count does not match filesystem temp path count: "
                f"{len(src_spec.block_ids)} != {len(my_temp)}"
            )
        self._store_temp_paths[job_id] = list(my_temp)

        tasks = (
            lambda bid=int(block_id), final_path=final_path, temp_path=temp_path: (
                self._store_one(
                    block_id=bid,
                    final_path=final_path,
                    temp_path=temp_path,
                    block_size=dst_spec.block_size,
                )
            )
            for block_id, final_path, temp_path in zip(
                src_spec.block_ids, my_final, my_temp
            )
        )
        self._pool.enqueue_store(job_id, len(src_spec.block_ids), tasks)
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: FileSystemLoadStoreSpec,
        dst_spec: CPULoadStoreSpec,
    ) -> bool:
        self._ensure_config(src_spec)
        # Select this rank's slice of the concatenated per-rank paths.
        num_ranks: int = getattr(src_spec, "num_ranks", 1)
        my_files = self._select_rank_paths(src_spec.file_paths, num_ranks)

        if len(my_files) != len(dst_spec.block_ids):
            raise ValueError(
                "Filesystem path count does not match CPU block count: "
                f"{len(my_files)} != {len(dst_spec.block_ids)}"
            )
        tasks = (
            lambda source_path=source_path, bid=int(block_id): self._load_one(
                source_path=source_path,
                block_id=bid,
            )
            for source_path, block_id in zip(my_files, dst_spec.block_ids)
        )
        self._pool.enqueue_load(job_id, len(dst_spec.block_ids), tasks)
        return True

    def submit_lookup(
        self, job_id: int, spec: FileSystemLookupSpec
    ) -> bool:
        self._ensure_config(spec)
        num_ranks = spec.num_ranks
        my_files = self._select_rank_paths(spec.file_paths, num_ranks)
        self._pool.enqueue_load(
            job_id,
            1,
            [lambda: self._lookup_files(my_files, spec.block_size)],
        )
        return True

    def submit_control(
        self, job_id: int, spec: FileSystemControlSpec
    ) -> bool:
        self._ensure_config(spec)
        my_final = self._select_rank_paths(spec.file_paths, spec.num_ranks)
        my_temp = self._select_rank_paths(spec.temp_file_paths, spec.num_ranks)
        if len(my_final) != len(my_temp):
            raise ValueError("Filesystem control paths must have equal lengths")
        self._control_jobs[job_id] = spec.action
        task = self._commit_one if spec.action == "commit" else self._abort_one
        if spec.action not in {"commit", "abort"}:
            raise ValueError(f"Unknown filesystem control action: {spec.action}")
        tasks = (
            lambda final_path=final_path, temp_path=temp_path: task(
                final_path=final_path,
                temp_path=temp_path,
                block_size=spec.block_size,
            )
            for final_path, temp_path in zip(my_final, my_temp)
        )
        self._pool.enqueue_store(job_id, len(my_final), tasks)
        return True

    @staticmethod
    def _lookup_files(paths: list[str], block_size: int) -> None:
        for path in paths:
            info = os.stat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_size != block_size:
                raise OSError(f"Invalid filesystem block file: {path}")

    def _store_one(
        self,
        *,
        block_id: int,
        final_path: str,
        temp_path: str,
        block_size: int,
    ) -> None:
        try:
            info = os.stat(final_path)
        except FileNotFoundError:
            info = None
        if info is not None:
            if stat.S_ISREG(info.st_mode) and info.st_size == block_size:
                return
            raise OSError(f"Pre-existing final is not a valid block: {final_path}")

        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_TRUNC
        fd = os.open(temp_path, flags, 0o644)
        try:
            os.ftruncate(fd, block_size)
            offset = self._rank_offset
            for tensor in self._cpu_tensors:
                row = tensor[block_id].numpy()
                view = memoryview(row).cast("B")
                _write_all_at(fd, view, offset)
                offset += len(view)
            os.fsync(fd)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_path)
            raise
        finally:
            os.close(fd)

    def _commit_one(
        self, *, final_path: str, temp_path: str, block_size: int
    ) -> None:
        try:
            info = os.stat(final_path)
        except FileNotFoundError:
            info = None
        if info is not None:
            if not stat.S_ISREG(info.st_mode) or info.st_size != block_size:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temp_path)
                raise OSError(f"Pre-existing final is not a valid block: {final_path}")
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_path)
        else:
            try:
                os.replace(temp_path, final_path)
            except Exception:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temp_path)
                raise

        dir_fd = os.open(os.path.dirname(final_path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    @staticmethod
    def _abort_one(*, final_path: str, temp_path: str, block_size: int) -> None:
        del final_path, block_size
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_path)

    def _load_one(self, *, source_path: str, block_id: int) -> None:
        fd = os.open(source_path, os.O_RDONLY)
        try:
            offset = self._rank_offset
            for tensor in self._cpu_tensors:
                row = tensor[block_id].numpy()
                view = memoryview(row).cast("B")
                _read_exact_at(fd, view, offset)
                offset += len(view)
        finally:
            os.close(fd)

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id, success in self._pool.get_finished():
            if job_id in self._control_jobs:
                self._control_jobs.pop(job_id, None)
                self._cleanup_store_temps({job_id})
            elif not success:
                self._cleanup_store_temps({job_id})
            results.append(
                TransferResult(
                    job_id=job_id,
                    success=success,
                    transfer_size=None,
                    transfer_time=None,
                )
            )
        return results

    def _cleanup_store_temps(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            for temp_path in self._store_temp_paths.pop(job_id, []):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(temp_path)

    def wait(self, job_ids: set[int] | None = None) -> None:
        del job_ids
        self._pool.wait_idle()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        self._cleanup_store_temps(set(self._store_temp_paths))

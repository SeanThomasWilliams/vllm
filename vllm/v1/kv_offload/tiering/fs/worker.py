# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import hashlib
import hmac
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

_DIGEST_SIZE = hashlib.sha256().digest_size
_DIGEST_CHUNK_SIZE = 1024 * 1024


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
            raise OSError(f"Short read: expected {len(view)} bytes, read {read}")
        end = read + len(chunk)
        view[read:end] = chunk
        read = end


def _digest_at(fd: int, length: int) -> bytes:
    digest = hashlib.sha256()
    offset = 0
    while offset < length:
        chunk = os.pread(fd, min(_DIGEST_CHUNK_SIZE, length - offset), offset)
        if not chunk:
            raise OSError(f"Short read while hashing filesystem block at {offset}")
        digest.update(chunk)
        offset += len(chunk)
    return digest.digest()


def _validate_block_fd(fd: int, block_size: int) -> None:
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size != block_size + _DIGEST_SIZE
    ):
        raise OSError("Filesystem block type, link count, or size is invalid")
    expected = os.pread(fd, _DIGEST_SIZE, block_size)
    if len(expected) != _DIGEST_SIZE or not hmac.compare_digest(
        expected, _digest_at(fd, block_size)
    ):
        raise OSError("Filesystem block integrity digest does not match")


def _open_validated_block(path: str, block_size: int) -> int:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        _validate_block_fd(fd, block_size)
    except Exception:
        os.close(fd)
        raise
    return fd


def _lstat_safe_final(path: str):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise OSError(f"Filesystem block final is not a safe file: {path}")
    return info


def _is_valid_block(path: str, block_size: int) -> bool:
    try:
        fd = _open_validated_block(path, block_size)
    except (FileNotFoundError, OSError):
        return False
    os.close(fd)
    return True


def _fsync_parent(path: str) -> None:
    dir_fd = os.open(
        os.path.dirname(path) or ".",
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY,
    )
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


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
        # Final files replaced by a commit are retained until finalize. The
        # release phase is only a barrier: a failure there must still leave
        # every rank's matching-inode rollback record available to abort.
        self._committed_final_paths: dict[int, dict[str, tuple[int, int]]] = {}
        self._validated_configs: dict[str, dict[str, Any]] = {}
        self._lookup_results: list[TransferResult] = []
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
                block_size=src_spec.block_size,
            )
            for source_path, block_id in zip(my_files, dst_spec.block_ids)
        )
        self._pool.enqueue_load(job_id, len(dst_spec.block_ids), tasks)
        return True

    def submit_lookup(self, job_id: int, spec: FileSystemLookupSpec) -> bool:
        self._ensure_config(spec)
        num_ranks = spec.num_ranks
        my_files = self._select_rank_paths(spec.file_paths, num_ranks)
        # Worker lookups are already dispatched asynchronously from the
        # scheduler in one batched collective. Resolve each local rank's paths
        # inline so expected cold-cache misses do not create and error-log one
        # thread-pool task per four-token DeepSeek block.
        success = True
        try:
            self._lookup_files(my_files, spec.block_size)
        except FileNotFoundError:
            success = False
        except OSError as exc:
            logger.error("Filesystem lookup failed: %s", exc)
            success = False
        self._lookup_results.append(
            TransferResult(
                job_id=job_id,
                success=success,
                transfer_size=None,
                transfer_time=None,
            )
        )
        return True

    def submit_control(self, job_id: int, spec: FileSystemControlSpec) -> bool:
        self._ensure_config(spec)
        my_final = self._select_rank_paths(spec.file_paths, spec.num_ranks)
        my_temp = self._select_rank_paths(spec.temp_file_paths, spec.num_ranks)
        if len(my_final) != len(my_temp):
            raise ValueError("Filesystem control paths must have equal lengths")
        if spec.action not in {"commit", "abort", "release", "finalize"}:
            raise ValueError(f"Unknown filesystem control action: {spec.action}")
        self._control_jobs[job_id] = spec.action
        if spec.action == "commit":
            tasks = (
                lambda final_path=final_path, temp_path=temp_path: self._commit_one(
                    job_id=job_id,
                    final_path=final_path,
                    temp_path=temp_path,
                    block_size=spec.block_size,
                )
                for final_path, temp_path in zip(my_final, my_temp)
            )
        elif spec.action == "abort":
            committed_final_paths = self._committed_final_paths.get(job_id, {})
            tasks = (
                lambda final_path=final_path, temp_path=temp_path: self._abort_one(
                    final_path=final_path,
                    temp_path=temp_path,
                    block_size=spec.block_size,
                    committed_final=committed_final_paths.get(final_path),
                )
                for final_path, temp_path in zip(my_final, my_temp)
            )
        elif spec.action == "release":
            tasks = (
                lambda final_path=final_path, temp_path=temp_path: self._release_one(
                    final_path=final_path,
                    temp_path=temp_path,
                    block_size=spec.block_size,
                )
                for final_path, temp_path in zip(my_final, my_temp)
            )
        else:
            tasks = (
                lambda final_path=final_path, temp_path=temp_path: self._finalize_one(
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
            fd = _open_validated_block(path, block_size)
            os.close(fd)

    def _store_one(
        self,
        *,
        block_id: int,
        final_path: str,
        temp_path: str,
        block_size: int,
    ) -> None:
        _lstat_safe_final(final_path)
        if _is_valid_block(final_path, block_size):
            return

        os.makedirs(os.path.dirname(temp_path) or ".", exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        fd = os.open(temp_path, flags, 0o600)
        try:
            os.ftruncate(fd, block_size + _DIGEST_SIZE)
            offset = self._rank_offset
            for tensor in self._cpu_tensors:
                row = tensor[block_id].numpy()
                view = memoryview(row).cast("B")
                _write_all_at(fd, view, offset)
                offset += len(view)
            _write_all_at(fd, memoryview(_digest_at(fd, block_size)), block_size)
            os.fsync(fd)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_path)
            raise
        finally:
            os.close(fd)

    def _commit_one(
        self,
        *,
        job_id: int,
        final_path: str,
        temp_path: str,
        block_size: int,
    ) -> None:
        try:
            _lstat_safe_final(final_path)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_path)
            raise
        if _is_valid_block(final_path, block_size):
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                return
            _fsync_parent(final_path)
            return

        try:
            temp_fd = _open_validated_block(temp_path, block_size)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_path)
            raise
        else:
            os.close(temp_fd)

        try:
            os.replace(temp_path, final_path)
            info = os.stat(final_path)
            self._committed_final_paths.setdefault(job_id, {})[final_path] = (
                info.st_dev,
                info.st_ino,
            )
            _fsync_parent(final_path)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_path)
            raise

    @staticmethod
    def _abort_one(
        *,
        final_path: str,
        temp_path: str,
        block_size: int,
        committed_final: tuple[int, int] | None,
    ) -> None:
        del block_size
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_path)
        if committed_final is None:
            return
        try:
            info = os.lstat(final_path)
        except FileNotFoundError:
            return
        if (
            stat.S_ISREG(info.st_mode)
            and info.st_nlink == 1
            and (info.st_dev, info.st_ino) == committed_final
        ):
            os.unlink(final_path)
            _fsync_parent(final_path)

    @staticmethod
    def _release_one(*, final_path: str, temp_path: str, block_size: int) -> None:
        """Acknowledge a successful commit without touching its final file."""
        del final_path, temp_path, block_size

    @staticmethod
    def _finalize_one(*, final_path: str, temp_path: str, block_size: int) -> None:
        """Acknowledge cleanup without touching a published final file."""
        del final_path, temp_path, block_size

    def _load_one(self, *, source_path: str, block_id: int, block_size: int) -> None:
        fd = _open_validated_block(source_path, block_size)
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
        results = self._lookup_results
        self._lookup_results = []
        for job_id, success, transfer_time in self._pool.get_finished():
            action = self._control_jobs.pop(job_id, None)
            if action is not None:
                self._cleanup_store_temps({job_id})
                if success and action in {"abort", "finalize"}:
                    self._committed_final_paths.pop(job_id, None)
            elif not success:
                self._cleanup_store_temps({job_id})
            results.append(
                TransferResult(
                    job_id=job_id,
                    success=success,
                    transfer_size=None,
                    transfer_time=transfer_time,
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
        self._committed_final_paths.clear()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

from typing_extensions import override

from vllm.v1.kv_offload.base import LoadStoreSpec


def ensure_config_file(config_path: str, run_config: dict[str, Any]) -> None:
    """Create and validate a filesystem-tier config from a worker."""
    parent = os.path.dirname(config_path)
    os.makedirs(parent, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix=".config.", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as config_file:
            json.dump(run_config, config_file, indent=2, sort_keys=True)
            config_file.flush()
            os.fsync(config_file.fileno())
        with contextlib.suppress(FileExistsError):
            os.link(temp_path, config_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_path)

    try:
        with open(config_path, encoding="utf-8") as config_file:
            existing_config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid filesystem-tier config: {config_path}") from exc
    if existing_config != run_config:
        raise ValueError(f"Filesystem-tier config mismatch: {config_path}")


@dataclass
class FileSystemLoadStoreSpec(LoadStoreSpec):
    """Worker-visible filesystem load/store spec."""

    file_paths: list[str]
    block_size: int
    temp_file_paths: list[str] | None = None
    num_ranks: int = 1
    config_path: str | None = None
    run_config: dict[str, Any] | None = None

    @staticmethod
    @override
    def medium() -> str:
        return "file_system"


@dataclass
class FileSystemLookupSpec(LoadStoreSpec):
    """Worker-visible all-rank regular-file lookup spec."""

    file_paths: list[str]
    block_size: int
    num_ranks: int = 1
    config_path: str | None = None
    run_config: dict[str, Any] | None = None

    @staticmethod
    @override
    def medium() -> str:
        return "file_system_lookup"


@dataclass
class FileSystemControlSpec(LoadStoreSpec):
    """Worker-visible store commit, abort, or release control spec."""

    action: str
    file_paths: list[str]
    temp_file_paths: list[str]
    block_size: int
    num_ranks: int = 1
    config_path: str | None = None
    run_config: dict[str, Any] | None = None

    @staticmethod
    @override
    def medium() -> str:
        return "file_system_control"

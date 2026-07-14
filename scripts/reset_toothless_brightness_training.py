#!/usr/bin/env python3
# ruff: noqa: EM101, EM102, TRY003, TRY004
"""One-time Toothless recovery for startup-contaminated brightness learning.

The CLI refuses to run unless Docker confirms that Home Assistant is stopped.
It snapshots the installed component and both private learning Stores before it
atomically resets only brightness learner state in the training Store.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4


def assert_container_stopped(container_name: str) -> None:
    """Fail closed unless Docker confirms the named container is stopped."""
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["/usr/bin/docker", "inspect", "--format", "{{.State.Running}}", container_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect Home Assistant: {result.stderr.strip()}")
    if result.stdout.strip().lower() != "false":
        raise RuntimeError("Home Assistant must be stopped before Store recovery")


def _fsync_directory(path: Path) -> None:
    """Make directory-entry changes durable across a sudden power loss."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    """Flush a copied rollback file before its backup is reported usable."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _copy_file_durable(source: Path | str, destination: Path | str) -> str:
    """Copy bytes/metadata/ownership and fsync the rollback copy."""
    source_path = Path(source)
    destination_path = Path(destination)
    shutil.copy2(source_path, destination_path)
    source_stat = source_path.stat()
    os.chown(destination_path, source_stat.st_uid, source_stat.st_gid)
    _fsync_file(destination_path)
    return str(destination_path)


def _copy_tree_durable(source: Path, destination: Path) -> None:
    """Recursively copy a component with durable files and directory metadata."""
    shutil.copytree(source, destination, copy_function=_copy_file_durable)
    source_directories = [source, *(item for item in source.rglob("*") if item.is_dir())]
    for source_directory in source_directories:
        destination_directory = destination / source_directory.relative_to(source)
        source_stat = source_directory.stat()
        os.chown(destination_directory, source_stat.st_uid, source_stat.st_gid)
        _fsync_directory(destination_directory)


def _backup_inputs(
    *,
    training_store: Path,
    behavior_store: Path,
    component_directory: Path,
    backup_root: Path,
    now: datetime,
) -> Path:
    """Snapshot every rollback input into one collision-resistant directory."""
    suffix = f"{now:%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:8]}"
    backup = backup_root / f"adaptive-lighting-pre-hardening-{suffix}"
    backup.mkdir(parents=True, exist_ok=False)
    _copy_file_durable(training_store, backup / training_store.name)
    _copy_file_durable(behavior_store, backup / behavior_store.name)
    _copy_tree_durable(component_directory, backup / component_directory.name)
    _fsync_directory(backup)
    _fsync_directory(backup_root)
    return backup


def reset_store(
    training_store: Path,
    *,
    behavior_store: Path,
    component_directory: Path,
    backup_root: Path,
    training_days: int = 7,
    home_assistant_stopped: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Back up rollback inputs and reset only contaminated brightness state."""
    if not home_assistant_stopped:
        raise RuntimeError("refusing recovery without stopped-HA confirmation")
    if training_days < 1:
        raise ValueError("training days must be positive")

    current = (now or datetime.now(UTC)).astimezone(UTC)
    file_stat = training_store.stat()
    envelope = json.loads(training_store.read_text(encoding="utf-8"))
    data = envelope.get("data")
    if not isinstance(data, dict):
        raise ValueError("expected a Home Assistant Store envelope")
    learner = data.get("learner")
    counts = data.get("sample_counts")
    behavior_ids = data.get("behavior_sample_ids")
    day_counts = data.get("day_type_counts")
    if not isinstance(learner, dict) or not isinstance(learner.get("entries"), list):
        raise ValueError("expected a versioned preference learner")
    if not isinstance(counts, dict) or not isinstance(behavior_ids, list):
        raise ValueError("expected commissioning counters and behavior ledger")
    if not isinstance(day_counts, dict):
        raise ValueError("expected day-type counters")

    behavior_count = int(counts.get("behavior_accepted", 0))
    if behavior_count != len(behavior_ids):
        raise ValueError("behavior count does not match the persisted ledger")
    # Verified against the live first-day Store before this one-time recovery.
    # Refuse to invent a distribution if later data invalidates that fact.
    if set(day_counts) - {"weekday"} or int(
        day_counts.get("weekday", 0),
    ) < behavior_count:
        raise ValueError("retained behavior day types are no longer known to be weekdays")

    backup = _backup_inputs(
        training_store=training_store,
        behavior_store=behavior_store,
        component_directory=component_directory,
        backup_root=backup_root,
        now=current,
    )

    deadline = current + timedelta(days=training_days)
    rejected = max(0, int(counts.get("rejected", 0)))
    superseded = max(0, int(counts.get("superseded", 0)))
    learner["entries"] = []
    data.update(
        {
            "training_started_at": current.isoformat(),
            "start": current.isoformat(),
            "training_deadline": deadline.isoformat(),
            "deadline": deadline.isoformat(),
            "phase": "shadow_learning",
            "promotion_reason": "training_in_progress",
            "sample_counts": {
                "accepted": behavior_count,
                "behavior_accepted": behavior_count,
                "rejected": rejected,
                "superseded": superseded,
                "pending": 0,
                "total": behavior_count + rejected,
            },
            "sample_count": behavior_count,
            "day_type_counts": ({"weekday": behavior_count} if behavior_count else {}),
            # Keep last_sample and last_rejection_reason as incident diagnostics.
            "pending": [],
        },
    )

    payload = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{training_store.name}.",
        dir=training_store.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(stat.S_IMODE(file_stat.st_mode))
        os.chown(temporary_name, file_stat.st_uid, file_stat.st_gid)
        temporary_path.replace(training_store)
        _fsync_directory(training_store.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "backup": str(backup),
        "phase": data["phase"],
        "training_started_at": data["training_started_at"],
        "training_deadline": data["training_deadline"],
        "accepted": behavior_count,
        "behavior_accepted": behavior_count,
        "brightness_entries": 0,
        "pending": 0,
        "rejected_preserved": rejected,
        "superseded_preserved": superseded,
    }


def main() -> None:
    """Verify stopped state, perform recovery, and print safe diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("training_store", type=Path)
    parser.add_argument("behavior_store", type=Path)
    parser.add_argument("component_directory", type=Path)
    parser.add_argument("backup_root", type=Path)
    parser.add_argument("--container-name", default="homeassistant")
    parser.add_argument("--training-days", type=int, default=7)
    args = parser.parse_args()
    assert_container_stopped(args.container_name)
    result = reset_store(
        args.training_store,
        behavior_store=args.behavior_store,
        component_directory=args.component_directory,
        backup_root=args.backup_root,
        training_days=args.training_days,
        home_assistant_stopped=True,
    )
    print(json.dumps(result, indent=2))  # noqa: T201 - CLI verification output


if __name__ == "__main__":
    main()

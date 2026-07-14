"""Tests for the one-time Toothless Store recovery tool."""

import json
import stat
from datetime import UTC, datetime

import pytest

from scripts import reset_toothless_brightness_training as recovery

reset_store = recovery.reset_store


def _write_inputs(tmp_path):
    training = tmp_path / "adaptive_lighting_training"
    behavior = tmp_path / "adaptive_lighting_behavior"
    component = tmp_path / "adaptive_lighting"
    backup_root = tmp_path / "backups"
    component.mkdir()
    (component / "manifest.json").write_text("{}", encoding="utf-8")
    behavior.write_text('{"version":1,"data":{}}', encoding="utf-8")
    training.write_text(
        json.dumps(
            {
                "version": 1,
                "data": {
                    "learner": {"version": 1, "config": {}, "entries": [{"x": 1}]},
                    "sample_counts": {
                        "accepted": 29,
                        "behavior_accepted": 2,
                        "rejected": 7,
                        "superseded": 5,
                        "pending": 1,
                        "total": 36,
                    },
                    "behavior_sample_ids": ["a", "b"],
                    "day_type_counts": {"weekday": 29},
                    "last_sample": {"diagnostic": True},
                    "last_rejection_reason": "superseded",
                    "pending": [{"id": "noise"}],
                },
            },
        ),
        encoding="utf-8",
    )
    training.chmod(0o640)
    return training, behavior, component, backup_root


def test_recovery_is_atomic_scoped_and_fully_backed_up(tmp_path) -> None:
    """Recovery keeps behavior/diagnostics and snapshots every rollback input."""
    training, behavior, component, backup_root = _write_inputs(tmp_path)
    result = reset_store(
        training,
        behavior_store=behavior,
        component_directory=component,
        backup_root=backup_root,
        home_assistant_stopped=True,
        now=datetime(2026, 7, 14, 8, tzinfo=UTC),
    )
    data = json.loads(training.read_text(encoding="utf-8"))["data"]
    backup = backup_root / result["backup"].split("/")[-1]

    assert data["learner"]["entries"] == []
    assert data["pending"] == []
    assert data["sample_counts"] == {
        "accepted": 2,
        "behavior_accepted": 2,
        "rejected": 7,
        "superseded": 5,
        "pending": 0,
        "total": 9,
    }
    assert data["last_sample"] == {"diagnostic": True}
    assert data["last_rejection_reason"] == "superseded"
    assert stat.S_IMODE(training.stat().st_mode) == 0o640
    assert (backup / training.name).is_file()
    assert (backup / behavior.name).is_file()
    assert (backup / behavior.name).read_bytes() == behavior.read_bytes()
    assert (backup / component.name / "manifest.json").is_file()
    assert (backup / component.name / "manifest.json").read_bytes() == (
        component / "manifest.json"
    ).read_bytes()
    assert (backup / behavior.name).stat().st_uid == behavior.stat().st_uid
    assert (backup / component.name).stat().st_uid == component.stat().st_uid


def test_repeated_recovery_uses_distinct_backups(tmp_path) -> None:
    """Two invocations at the same instant cannot overwrite rollback data."""
    training, behavior, component, backup_root = _write_inputs(tmp_path)
    moment = datetime(2026, 7, 14, 8, tzinfo=UTC)
    first = reset_store(
        training,
        behavior_store=behavior,
        component_directory=component,
        backup_root=backup_root,
        home_assistant_stopped=True,
        now=moment,
    )
    second = reset_store(
        training,
        behavior_store=behavior,
        component_directory=component,
        backup_root=backup_root,
        home_assistant_stopped=True,
        now=moment,
    )
    assert first["backup"] != second["backup"]


def test_recovery_fails_closed_for_running_or_unknown_day_types(tmp_path) -> None:
    """The tool requires stopped HA and refuses an invented day distribution."""
    training, behavior, component, backup_root = _write_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="stopped-HA"):
        reset_store(
            training,
            behavior_store=behavior,
            component_directory=component,
            backup_root=backup_root,
        )

    envelope = json.loads(training.read_text(encoding="utf-8"))
    envelope["data"]["day_type_counts"] = {"weekend": 29}
    training.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="day types"):
        reset_store(
            training,
            behavior_store=behavior,
            component_directory=component,
            backup_root=backup_root,
            home_assistant_stopped=True,
        )


def test_container_guard_accepts_only_confirmed_stopped(monkeypatch) -> None:
    """The CLI guard fails for running or uninspectable Home Assistant."""
    completed = recovery.subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="false\n",
        stderr="",
    )
    monkeypatch.setattr(
        recovery.subprocess,
        "run",
        lambda *_args, **_kwargs: completed,
    )
    recovery.assert_container_stopped("homeassistant")

    completed.stdout = "true\n"
    with pytest.raises(RuntimeError, match="must be stopped"):
        recovery.assert_container_stopped("homeassistant")
    completed.returncode = 1
    completed.stderr = "not found"
    with pytest.raises(RuntimeError, match="cannot inspect"):
        recovery.assert_container_stopped("homeassistant")


def test_backup_fsyncs_every_copied_file(tmp_path, monkeypatch) -> None:
    """Training, behavior, and recursive component files are flushed."""
    training, behavior, component, backup_root = _write_inputs(tmp_path)
    nested = component / "nested"
    nested.mkdir()
    (nested / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    synced: list[str] = []
    original = recovery._fsync_file

    def capture(path):
        synced.append(path.name)
        original(path)

    monkeypatch.setattr(recovery, "_fsync_file", capture)
    reset_store(
        training,
        behavior_store=behavior,
        component_directory=component,
        backup_root=backup_root,
        home_assistant_stopped=True,
    )

    assert {training.name, behavior.name, "manifest.json", "module.py"} <= set(synced)

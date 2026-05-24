"""Artifact paths and guards."""

from pathlib import Path

import pytest

from retainiq import artifacts, config


def test_config_import_does_not_create_processed_dir(tmp_path, monkeypatch):
    processed = tmp_path / "processed_new"
    monkeypatch.setattr(config, "DATA_PROCESSED", processed)
    assert not processed.exists()
    import importlib
    import retainiq.config as cfg

    importlib.reload(cfg)
    assert not processed.exists()


def test_ensure_dirs_creates_outputs(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    submission = tmp_path / "submission"
    monkeypatch.setattr(config, "DATA_PROCESSED", processed)
    monkeypatch.setattr(config, "SUBMISSION_DIR", submission)
    artifacts.ensure_dirs()
    assert processed.is_dir()
    assert submission.is_dir()
    assert artifacts.chart_dir().is_dir()


def test_require_trained_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing trained artifacts"):
        artifacts.require_trained()


def test_trained_bundle_exists_false_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    assert artifacts.trained_bundle_exists() is False


def test_write_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    out = artifacts.write_manifest({"optimal_threshold": 0.01})
    assert out.is_file()
    assert config.TEAM_NAME in out.read_text()

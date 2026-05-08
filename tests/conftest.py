"""pytest fixtures."""

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch):
    """Use a per-test temporary SQLite DB so tests don't share state."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "polyclaude_test.sqlite")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("PRIVATE_KEY", "")
    monkeypatch.setenv("FUNDER", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    # Reset cached settings + engine
    from polyclaude import config as cfg
    from polyclaude.ledger import db as ldb

    cfg.get_settings.cache_clear()
    ldb.get_engine.cache_clear()
    ldb._session_factory.cache_clear()
    yield
    cfg.get_settings.cache_clear()
    ldb.get_engine.cache_clear()
    ldb._session_factory.cache_clear()

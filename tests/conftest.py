import pytest


@pytest.fixture(autouse=True)
def isolated_run_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PURSUIT_RUNS_DIR", str(tmp_path / "runs"))

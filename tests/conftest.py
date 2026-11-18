"""Every test runs against the sibling IDE (``THETA_IDE_ROOT``) in its own empty workspace."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IDE = Path(os.environ.get('THETA_IDE_ROOT', ROOT.parent / 'Theta-IDE')).resolve()
for path in (str(IDE), str(Path(__file__).parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Selected at import, so module-level catalog lookups during collection never see a real workspace.
os.environ['THETA_WORKSPACE'] = tempfile.mkdtemp(prefix='theta-plugins-tests-')
os.environ.pop('THETA_DATA_DIR', None)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@pytest.fixture(autouse=True)
def empty_workspace(tmp_path_factory, monkeypatch):
    """An empty process workspace; tests that need packages build their own."""
    root = tmp_path_factory.mktemp('process-workspace')
    monkeypatch.setenv('THETA_WORKSPACE', str(root))
    monkeypatch.delenv('THETA_DATA_DIR', raising=False)
    return root


@pytest.fixture
def workspace_results(tmp_path, monkeypatch):
    """Select ``tmp_path`` as the process workspace and return its results folder."""
    from src.app.workspace import WorkspacePaths
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path))
    return WorkspacePaths.for_process().results

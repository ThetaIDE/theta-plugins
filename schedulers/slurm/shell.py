"""Shell environment for Slurm job scripts.

Job scripts resolve the project from the submission directory, initialize
environment modules in non-interactive shells, and export the workspace the
job runs against, so a script stays portable across users and clusters.
"""
import os
import shlex
from pathlib import Path

from src.app.api.workspace import WorkspacePaths


def env_block(site_cfg=None) -> str:
    """Shell export lines: PROJECT_ROOT, module loads, workspace, PYTHONPATH and extra variables."""
    explicit_root = getattr(site_cfg, "project_root", None) if site_cfg else None
    if explicit_root:
        lines = [f'export PROJECT_ROOT="${{PROJECT_ROOT:-{explicit_root}}}"']
    else:
        lines = ['export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"']

    modules = getattr(site_cfg, "module_loads", []) or []
    if modules:
        lines.append("if ! command -v module &> /dev/null; then")
        lines.append("    [ -f /usr/share/modules/init/bash ] && source /usr/share/modules/init/bash")
        lines.append("    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh")
        lines.append("fi")
        for module in modules:
            lines.append(f"command -v module &> /dev/null && module load {module} || true")

    # Remote workspaces resolve installed/user providers from the explicitly selected tree.
    lines.extend(['export THETA_WORKSPACE="${THETA_WORKSPACE:-$PROJECT_ROOT}"',
                  'export THETA_DATA_DIR="${THETA_DATA_DIR:-$THETA_WORKSPACE/.thetaide}"'])
    parts = ["$PROJECT_ROOT", "$PROJECT_ROOT/src"]
    parts.extend(f"$PROJECT_ROOT/{extra}" for extra in getattr(site_cfg, "extra_pythonpath", []) or [])
    parts.append("$PYTHONPATH")
    lines.append(f"export PYTHONPATH={':'.join(parts)}")

    snapshot = os.environ.get("THETA_PROVIDER_SNAPSHOT")
    if snapshot:
        try:
            relative = Path(snapshot).relative_to(WorkspacePaths.for_process().root)
            lines.append(f'export THETA_PROVIDER_SNAPSHOT="$THETA_WORKSPACE/{relative.as_posix()}"')
        except ValueError:
            lines.append("export THETA_PROVIDER_SNAPSHOT=" + shlex.quote(snapshot))
    for key, value in (getattr(site_cfg, "extra_env_vars", {}) or {}).items():
        lines.append(f"export {key}={value}")
    return "\n".join(lines) + "\n"


def python_cmd(site_cfg=None) -> str:
    """The job script's interpreter inside the site's virtual environment."""
    venv_dir = str(getattr(site_cfg, "venv_dir", "venv")) if site_cfg is not None else "venv"
    return f"$PROJECT_ROOT/{venv_dir}/bin/python3"

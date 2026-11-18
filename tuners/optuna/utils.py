"""Optuna-related utilities for the pipeline.

Provides functions for querying best trials, managing studies,
and launching the Optuna dashboard.
"""

import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)

MIN_IMPORTANCE_TRIALS = 5
_importance_cache = WeakKeyDictionary()


from src.app.api.execution import get_python_executable


def results_root() -> Path:
    """The results directory of the workspace this process runs in."""
    from src.app.api.workspace import WorkspacePaths
    return WorkspacePaths.for_process().results


def default_storage_url() -> str:
    """The study database used when the tuning settings name no storage."""
    return "sqlite:///" + str(results_root() / "optuna" / "optuna.db")


def is_valid_storage_url(storage_url) -> bool:
    """Checks whether the given storage URL is non-empty and not in-memory (None/null)."""
    if not storage_url:
        return False
    s = str(storage_url).strip().lower()
    return s not in ("none", "null", "", "false")


def get_best_trial_id(storage_url, study_name):
    """Queries the Optuna database to find the best trial ID for a given study.

    Uses the Optuna Python API instead of raw SQL for robustness across
    schema versions.
    """
    if not is_valid_storage_url(storage_url):
        return "0"

    try:
        import optuna

        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            return str(study.best_trial.number)
        except KeyError as e:
            logger.debug("Study '%s' not found directly, checking versioned studies: %s", study_name, e)
            all_studies = optuna.get_all_study_summaries(storage=storage_url)
            matches = [
                s for s in all_studies if s.study_name == study_name or s.study_name.startswith(f"{study_name}_v")
            ]
            if matches:
                matches.sort(key=lambda s: s.study_name, reverse=True)
                target_study = optuna.load_study(study_name=matches[0].study_name, storage=storage_url)
                return str(target_study.best_trial.number)
            raise
        except Exception as e:
            logger.debug("Could not load study '%s' directly: %s", study_name, e)
            all_studies = optuna.get_all_study_summaries(storage=storage_url)
            matches = [
                s for s in all_studies if s.study_name == study_name or s.study_name.startswith(f"{study_name}_v")
            ]
            if matches:
                matches.sort(key=lambda s: s.study_name, reverse=True)
                target_study = optuna.load_study(study_name=matches[0].study_name, storage=storage_url)
                return str(target_study.best_trial.number)
            raise
    except Exception as e:
        print(f"Warning: Could not query best trial from Optuna: {e}")
        return "0"


def launch_optuna_dashboard(storage_url):
    """Launches the optuna-dashboard in the background and opens the browser."""
    if not storage_url:
        return

    # Ensure directory exists if using sqlite
    if storage_url.startswith("sqlite:///"):
        db_path = storage_url.replace("sqlite:///", "")
        if "/" in db_path:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Check if a dashboard is already running for this storage_url
    try:
        ps_output = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
        if storage_url in ps_output and "optuna_dashboard" in ps_output:
            print(f"Optuna Dashboard already running for: {storage_url}. Skipping launch.")
            return
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("Could not check running processes via ps aux: %s", e)
    except Exception as e:
        logger.debug("Unexpected error checking running processes: %s", e)

    print(f"Launching Optuna Dashboard for: {storage_url}")

    # Use managed python to ensure dashboard is available
    venv_python = get_python_executable()
    venv_bin_dir = os.path.dirname(venv_python)
    optuna_dashboard_bin = os.path.join(venv_bin_dir, "optuna-dashboard")

    # Try to launch optuna-dashboard
    try:
        # Launch in background
        port = 8080
        # Try different ports if 8080 is taken
        for p in range(8080, 8090):
            try:
                # Use a dummy check to see if port is in use
                subprocess.run(["nc", "-z", "localhost", str(p)], capture_output=True, check=True)
                continue  # Port is in use
            except (subprocess.CalledProcessError, FileNotFoundError):
                # CalledProcessError means port is not in use; FileNotFoundError means nc is missing
                port = p
                break
            except subprocess.SubprocessError as e:
                logger.debug("Subprocess error checking port %s: %s; selecting port", p, e)
                port = p
                break

        # Ensure logging directory exists
        log_file_path = str(results_root() / "optuna" / "dashboard.log")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        log_file = open(log_file_path, "w")

        cmd = [optuna_dashboard_bin, storage_url, "--port", str(port)]
        subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

        print(f"Optuna Dashboard started at http://127.0.0.1:{port}")

        # Wait a moment for server to start then open browser in a separate thread
        def open_browser():
            time.sleep(2)
            url = f"http://127.0.0.1:{port}"
            if sys.platform == "darwin":
                # On macOS, -g opens in background without focusing
                subprocess.run(["open", "-g", url])
            else:
                webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: optuna-dashboard could not be started: {e}")
        print("Please ensure 'optuna-dashboard' is installed in your environment.")


def get_optuna_storage(storage_url: str):
    """Safely return an Optuna RDBStorage object with timeout and clean schema handling."""
    if not storage_url:
        return None
    import optuna

    if storage_url.startswith("sqlite:///"):
        import sqlite3

        db_raw = storage_url.replace("sqlite:///", "").split("?")[0]
        if db_raw and os.path.dirname(db_raw):
            os.makedirs(os.path.dirname(os.path.abspath(db_raw)), exist_ok=True)
        if os.environ.get('THETA_SQLITE_ROOT'):
            from pathlib import Path
            allowed = Path(os.environ['THETA_SQLITE_ROOT']).resolve()
            if Path(db_raw).resolve().parent != allowed:
                raise ValueError('Allocation SQLite must use its exclusively owned scratch database')
        # If DB file exists but is 0 bytes (empty file created by touch/connect),
        # remove it so Optuna creates the full table schema cleanly without AssertionError
        if os.path.exists(db_raw) and os.path.getsize(db_raw) == 0:
            try:
                os.remove(db_raw)
            except OSError as e:
                logger.debug("Could not remove 0-byte SQLite DB %s: %s", db_raw, e)
        # Pre-configure SQLite database with WAL journal mode and busy timeout via Python
        try:
            conn = sqlite3.connect(db_raw, timeout=120.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=120000;")
            conn.close()
        except sqlite3.Error as e:
            logger.debug("Could not pre-configure SQLite WAL/busy timeout on %s: %s", db_raw, e)
        except Exception as e:
            logger.debug("Unexpected error configuring SQLite DB %s: %s", db_raw, e)
        return optuna.storages.RDBStorage(url=f"sqlite:///{db_raw}", engine_kwargs={"connect_args": {"timeout": 120}})
    return storage_url


def get_next_study_name(group: str, experiment_id: str, agent_name: str, storage_url: str | None = None) -> str:
    """Choose a fresh version across sweep artifacts and storage.

    A cancelled study can exist without a completed training output folder.
    Storage is authoritative when provided; archived artifacts also reserve
    their version so reusing a database cannot overwrite a previous sweep.
    """
    import json
    import re
    from pathlib import Path
    from urllib.parse import unquote

    base_prefix = f"{experiment_id}_{agent_name}"
    study_pattern = re.compile(re.escape(base_prefix) + r"_v(\d+)$")
    version = 0

    def include_study(name):
        nonlocal version
        match = study_pattern.fullmatch(str(name))
        if match:
            version = max(version, int(match[1]) + 1)

    for directory in (results_root() / "logs" / group / experiment_id / agent_name,
                      results_root() / "checkpoints" / group / experiment_id / agent_name):
        for path in (directory / "sweeps").glob("*"):
            if path.is_dir():
                include_study(unquote(path.name))
        try:
            include_study(json.loads((directory / "tuning_summary.json").read_text())["study"])
        except (OSError, ValueError, KeyError, TypeError):
            pass

    if storage_url:
        import optuna
        for name in optuna.study.get_all_study_names(storage=get_optuna_storage(storage_url)):
            include_study(name)
    return f"{base_prefix}_v{version}"


def delete_optuna_study(storage_url, study_name):
    """Deletes an existing study from the Optuna database to start fresh."""
    if not is_valid_storage_url(storage_url):
        return
    try:
        import optuna

        storage = get_optuna_storage(storage_url)
        optuna.delete_study(study_name=study_name, storage=storage)
        print(f"Reset existing Optuna study: {study_name}")
    except KeyError:
        logger.debug("Optuna study '%s' does not exist to delete.", study_name)
    except Exception as e:
        logger.warning("Could not delete Optuna study '%s': %s", study_name, e)


def create_optuna_study(storage_url, study_name, direction="minimize"):
    """Pre-creates/initializes an Optuna study to avoid schema initialization race conditions on cluster nodes."""
    if not is_valid_storage_url(storage_url):
        return
    try:
        import optuna

        storage = get_optuna_storage(storage_url)
        optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction=direction)
        print(f"Pre-initialized Optuna study: {study_name}")
    except Exception as e:
        print(f"Warning: Failed to pre-initialize Optuna study '{study_name}': {e}")


def find_best_trial_from_logs(group: str, experiment_id: str, agent_name: str, direction: str = "minimize"):
    """Scan CSV logs to find the winning trial number and score for in-memory sweeps."""
    from pathlib import Path

    import pandas as pd

    log_dir = results_root() / "logs" / group / experiment_id / agent_name
    best_id = "0"
    best_val = float("inf") if direction == "minimize" else float("-inf")

    if not log_dir.exists():
        return best_id, None

    for v_dir in sorted(
        log_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0
    ):
        metrics_file = v_dir / "metrics.csv"
        if not metrics_file.exists():
            continue
        try:
            df = pd.read_csv(metrics_file)
            trial_num = v_dir.name.split("_")[-1]
            val = None
            for col in ["val/loss", "val/robust_loss", "losses/bellman_loss", "losses/total_loss", "eval/reward"]:
                if col in df.columns:
                    valid_series = df[col].dropna()
                    if len(valid_series) > 0:
                        val = float(valid_series.iloc[-1])
                        break
            if val is not None:
                if (direction == "minimize" and val < best_val) or (direction == "maximize" and val > best_val):
                    best_val = val
                    best_id = trial_num
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.debug("Skipping unparseable metrics file %s: %s", metrics_file, e)
        except (KeyError, ValueError, OSError) as e:
            logger.warning("Error reading metrics from %s: %s", metrics_file, e)
        except Exception as e:
            logger.warning("Unexpected error reading metrics from %s: %s", metrics_file, e)

    return best_id, (best_val if best_val != float("inf") and best_val != float("-inf") else None)


def _study_importance(study, trials):
    """Estimate once per completed trial set, outside the desktop's UI thread."""
    import json

    completed = [trial for trial in trials if trial["state"] == "complete" and trial["value"] is not None]
    info = {"completed_trials": len(completed), "minimum_trials": MIN_IMPORTANCE_TRIALS, "evaluator": "fANOVA"}
    if len(completed) < MIN_IMPORTANCE_TRIALS:
        return {}, dict(info, status="waiting", message=f"Importance becomes available after {MIN_IMPORTANCE_TRIALS} completed trials.")
    signature = tuple((trial["id"], trial["value"], json.dumps(trial["parameters"], sort_keys=True)) for trial in completed)
    cached = _importance_cache.get(study)
    if cached is not None and cached[0] == signature:
        return cached[1]
    if len({trial["value"] for trial in completed}) < 2:
        result = {}, dict(info, status="unavailable", message="Importance needs variation in the completed trials' objective values.")
    else:
        try:
            from optuna.importance import FanovaImportanceEvaluator, get_param_importances

            scores = get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=0))
            result = scores, dict(info, status="ready" if scores else "unavailable",
                                  message="" if scores else "No shared varying parameters are available to compare.")
        except Exception as exc:
            logger.debug("Parameter importance unavailable for study %s: %s", study.study_name, exc)
            result = {}, dict(info, status="unavailable", message="Importance could not be estimated from these trials. It will retry when another trial completes.")
    _importance_cache[study] = signature, result
    return result


def write_study_summary(study, ckpt_root):
    """Export a backend-neutral snapshot, including incomplete and failed trials."""
    import json
    import math
    import tempfile
    from pathlib import Path

    from src.app.api.tuning import trial_checkpoint_dir

    ckpt_root = Path(ckpt_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    try:
        best_trial_id = study.best_trial.number
    except ValueError:
        best_trial_id = None
    trials = []
    for item in study.trials:
        directory = trial_checkpoint_dir(ckpt_root, study.study_name, item.number)
        available = [str(path) for path in (directory / "best_model.ckpt", directory / "best_model.zip") if path.is_file()]
        trials.append({
            "id": item.number, "state": (item.user_attrs.get('theta_execution_state', 'running')
                                           if item.state.name == 'RUNNING' else item.state.name.lower()),
            "value": item.value if item.value is not None and math.isfinite(item.value) else None,
            "parameters": item.params,
            "log_version": item.user_attrs.get("theta_log_version"),
            "started": item.datetime_start.isoformat() if item.datetime_start else None,
            "execution_started": item.user_attrs.get('theta_worker_started_at'),
            "finished": item.datetime_complete.isoformat() if item.datetime_complete else None,
            "duration_seconds": item.duration.total_seconds() if item.duration else None,
            "checkpoint": available[0] if item.state.name == "COMPLETE" and len(available) == 1 else None,
            "objective_artifact": str(directory / "objective.json") if (directory / "objective.json").is_file() else None,
            "intermediate_values": {str(step): value for step, value in item.intermediate_values.items()
                                    if value is not None and math.isfinite(value)},
        })
    summary = {
        "schema_version": 1, "component": "optuna", "study": study.study_name,
        "direction": study.direction.name.lower(), "objective_metric": study.user_attrs.get("theta_objective_metric"),
        "stop_reason": study.user_attrs.get("theta_last_stop_reason"), "best_trial_id": best_trial_id, "trials": trials,
        "requested_trials": study.user_attrs.get("theta_requested_trials"),
        "initial_trials": study.user_attrs.get("theta_initial_trials"),
        "log_method": study.user_attrs.get("theta_log_method"),
    }
    current = trials[summary['initial_trials'] or 0:]
    requested = summary['requested_trials']
    future = max(0, requested - len(current)) if isinstance(requested, int) else 0
    summary['future_trials'] = future if summary['stop_reason'] == 'running' else 0
    summary['pending_trials'] = summary['future_trials'] + sum(t['state'] == 'pending' for t in current)
    summary["parameter_importance"], summary["parameter_importance_info"] = _study_importance(study, trials)
    study_root = trial_checkpoint_dir(ckpt_root, study.study_name, 0).parent
    study_root.mkdir(parents=True, exist_ok=True)
    for directory in (study_root, ckpt_root):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
                temporary = stream.name
                json.dump(summary, stream, indent=2, allow_nan=False)
            os.replace(temporary, directory / "tuning_summary.json")
            from src.app.api.tuning import commit_artifact as committed
            committed(directory / "tuning_summary.json")
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


def promote_best_trial_checkpoint(
    group: str, experiment_id: str, agent_name: str, storage_url: str | None = None, study_name: str | None = None
):
    """After an Optuna sweep, queries the winning trial, copies its checkpoint
    and hyperparameters to the main agent checkpoint root, and writes a summary."""
    import json
    import math
    import shutil
    from pathlib import Path

    import yaml

    from src.app.api.tuning import trial_checkpoint_dir
    import optuna

    if not is_valid_storage_url(storage_url) or not study_name:
        raise ValueError("Winner promotion requires persistent storage and an exact study name")
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    try:
        trial = study.best_trial
    except ValueError as exc:
        raise ValueError(f"Study {study_name!r} has no completed trial to promote") from exc
    if trial.value is None or not math.isfinite(trial.value):
        raise ValueError(f"Best trial in study {study_name!r} does not have a finite objective")
    ckpt_root = results_root() / "checkpoints" / group / experiment_id / agent_name
    trial_dir = trial_checkpoint_dir(ckpt_root, study_name, trial.number)
    objective_path = trial_dir / "objective.json"
    if objective_path.is_file():
        # A seed panel selects a configuration, not one privileged trained model.
        objective = json.loads(objective_path.read_text(encoding="utf-8"))
        best_info = {
            "study_name": study_name, "best_trial_id": trial.number, "best_value": trial.value,
            "best_params": trial.params, "direction": study.direction.name,
            "checkpoint": None, "objective_artifact": str(objective_path),
            "samples": [run["artifacts"] for run in objective["runs"]],
        }
        ckpt_root.mkdir(parents=True, exist_ok=True)
        (ckpt_root / "best_params.yaml").write_text(yaml.safe_dump(trial.params), encoding="utf-8")
        (ckpt_root / "best_trial_summary.json").write_text(json.dumps(best_info, indent=2), encoding="utf-8")
        write_study_summary(study, ckpt_root)
        print(f"\n[Optuna Winner] Selected configuration from trial #{trial.number}; {len(objective['runs'])} sample artifacts retained")
        return str(trial.number)
    candidates = [trial_dir / "best_model.ckpt", trial_dir / "best_model.zip"]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one checkpoint for study {study_name!r}, trial {trial.number} in {trial_dir}."
        )
    source = matches[0]
    suffix = source.suffix
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for destination in (ckpt_root / f"best_model{suffix}", ckpt_root / f"{agent_name}{suffix}",
                        ckpt_root.parent / f"{agent_name}{suffix}"):
        shutil.copy2(source, destination)
    best_info = {
        "study_name": study_name, "best_trial_id": trial.number,
        "best_value": trial.value, "best_params": trial.params,
        "direction": study.direction.name, "checkpoint": str(source),
    }
    with (ckpt_root / "best_params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(trial.params, f)
    with (ckpt_root / "best_trial_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)
    write_study_summary(study, ckpt_root)
    print(f"\n[Optuna Winner] Promoted trial #{trial.number} from {source}")
    return str(trial.number)


from functools import lru_cache


@lru_cache(maxsize=16)
def read_study_snapshot(database, mtime_ns, size):
    """Adapt a downloaded, closed SQLite export without changing the source database."""
    import json
    import shutil
    import tempfile
    from pathlib import Path
    import optuna
    with tempfile.TemporaryDirectory(prefix='theta-study-reader-') as temporary:
        root = Path(temporary)
        local = root / 'study.sqlite'
        shutil.copyfile(database, local)
        storage = 'sqlite:///' + str(local)
        summaries = []
        for item in optuna.get_all_study_summaries(storage=storage):
            study = optuna.load_study(study_name=item.study_name, storage=storage)
            method = study.user_attrs.get('theta_log_method')
            if not method:
                continue
            destination = root / 'summary'
            destination.mkdir(exist_ok=True)
            write_study_summary(study, destination)
            summaries.append((method, json.loads((destination / 'tuning_summary.json').read_text())))
        return summaries

import os
import sys
from pathlib import Path


import argparse
import json
from typing import Optional

import pandas as pd
import yaml

from src.app.core.plotting.base import BasePlotter, clean_label, get_method_aliases


def format_duration(seconds: float) -> str:
    """Formats seconds into human readable time (e.g., 2m 30.5s or 1h 15m 10s)."""
    if seconds is None or seconds <= 0:
        return "N/A"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.2f}s"


class ReportsPlotter(BasePlotter):
    def __init__(self):
        super().__init__("reports")

    def _find_hydra_config(self, group: str, exp_id: str, method: str) -> dict:
        """Attempt to load the resolved config for a specific method run from checkpoints, logs, or Hydra outputs."""
        clean_exp = Path(exp_id).stem
        aliases = get_method_aliases(method)
        norm_method = method.replace("/", "_")

        for alias in aliases:
            # 1. Check method-specific checkpoint directory
            ckpt_method_dir = Path("results/checkpoints") / group / clean_exp / alias
            if ckpt_method_dir.exists():
                for cfg_path in sorted(ckpt_method_dir.rglob("config.yaml")):
                    try:
                        with open(cfg_path) as f:
                            data = yaml.safe_load(f)
                        if data and "agent" in data:
                            return data
                    except Exception:
                        pass

            # 2. Check method-specific logs directory
            log_method_dir = Path("results/logs") / group / clean_exp / alias
            if log_method_dir.exists():
                for cfg_path in sorted(log_method_dir.rglob("config.yaml")):
                    try:
                        with open(cfg_path) as f:
                            data = yaml.safe_load(f)
                        if data and "agent" in data:
                            return data
                    except Exception:
                        pass

            for hp_path in sorted(log_method_dir.rglob("hparams.yaml")):
                try:
                    with open(hp_path) as f:
                        hp_data = yaml.safe_load(f)
                    if hp_data:
                        if "cfg" in hp_data and isinstance(hp_data["cfg"], dict) and "agent" in hp_data["cfg"]:
                            return hp_data["cfg"]
                        if "agent" in hp_data:
                            return hp_data
                except Exception:
                    pass

        # 3. Check experiment root configs
        for root_cand in [
            Path("results/checkpoints") / group / clean_exp / "config.yaml",
            Path("results/logs") / group / clean_exp / "config.yaml",
        ]:
            if root_cand.exists():
                try:
                    with open(root_cand) as f:
                        data = yaml.safe_load(f)
                    if data and "agent" in data:
                        return data
                except Exception:
                    pass

        # 4. Check in/config/agent/ directly
        for agent_file in Path("in/config/agent").rglob("*.yaml"):
            try:
                with open(agent_file) as f:
                    data = yaml.safe_load(f)
                if data and (data.get("name") == norm_method or agent_file.stem == norm_method):
                    return {"agent": data}
            except Exception:
                pass

        # 5. Search recent Hydra outputs
        hydra_base = Path("results/hydra/outputs")
        if hydra_base.exists():
            for date_dir in sorted(hydra_base.iterdir(), reverse=True):
                if not date_dir.is_dir():
                    continue
                for time_dir in sorted(date_dir.iterdir(), reverse=True):
                    cfg_path = time_dir / ".hydra" / "config.yaml"
                    if cfg_path.exists():
                        try:
                            with open(cfg_path) as f:
                                run_cfg = yaml.safe_load(f) or {}
                            exp_matches = (
                                run_cfg.get("experiment_id") in (exp_id, clean_exp) or run_cfg.get("group") == group
                            )
                            agent_name = run_cfg.get("agent", {}).get("name", "")
                            if exp_matches and (agent_name in (method, norm_method) or not agent_name):
                                return run_cfg
                        except Exception:
                            continue
        return {}

    def _extract_agent_params(self, hydra_cfg: dict) -> dict:
        """Extracts and flattens agent hyperparameters into a clean key-value dictionary."""
        if not hydra_cfg:
            return {}

        agent_cfg = hydra_cfg.get("agent", {})
        if not isinstance(agent_cfg, dict):
            return {}

        params = {}
        # Unpack nested agent if present (e.g. agent.agent.cql)
        if "agent" in agent_cfg and isinstance(agent_cfg["agent"], dict):
            sub = agent_cfg["agent"]
            for k, v in sub.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        params[sub_k] = sub_v
                else:
                    params[k] = v

        for k, v in agent_cfg.items():
            if k not in ("agent", "_target_"):
                params[k] = v

        return params

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        runs_data = self.load_metrics(group, exp_id)
        if not runs_data:
            return

        print(f"=== Generating Markdown Reports for '{exp_id}' ===")

        # Clean up legacy markdown comparison report if present
        old_report = output_dir / "methods_comparison_report.md"
        if old_report.exists():
            try:
                old_report.unlink()
            except Exception:
                pass

        if cfg.get("include_hyperparameters", True):
            hp_path = output_dir / "hyperparameters_report.md"
            with open(hp_path, "w") as f:
                f.write(f"# Hyperparameter Report: {clean_exp}\n\n")
                for method, versions in sorted(runs_data.items()):
                    f.write(f"## {clean_label(method)} (`{method}`)\n\n")

                    # Try to load actual hyperparameters from config
                    hydra_cfg = self._find_hydra_config(group, exp_id, method)
                    agent_params = self._extract_agent_params(hydra_cfg)

                    if agent_params:
                        # Extract key hyperparameters
                        hp_keys = [
                            "algorithm",
                            "lr",
                            "learning_rate",
                            "gamma",
                            "batch_size",
                            "cql_alpha",
                            "soft_target_tau",
                            "tau",
                            "beta",
                            "epochs",
                            "eval_every_n_epochs",
                            "actor_mode",
                            "blender_mode",
                            "blend_function",
                            "modules",
                            "hidden_sizes",
                            "ecm_dthr",
                            "fyd",
                            "fyd_top_k",
                        ]
                        f.write("| Parameter | Value |\n")
                        f.write("| --- | --- |\n")
                        written_keys = set()
                        for key in hp_keys:
                            if key in agent_params and agent_params[key] is not None:
                                val = agent_params[key]
                                if key == "modules" and isinstance(val, list):
                                    # Format module list nicely
                                    mod_strs = []
                                    for m in val:
                                        if isinstance(m, dict):
                                            m_type = m.get("type", "module")
                                            m_rules = m.get("rules", "")
                                            mod_strs.append(f"{m_type} (rules: {m_rules})" if m_rules else m_type)
                                        else:
                                            mod_strs.append(str(m))
                                    f.write(f"| `{key}` | `{', '.join(mod_strs)}` |\n")
                                else:
                                    f.write(f"| `{key}` | `{val}` |\n")
                                written_keys.add(key)

                        # Include any remaining parameters
                        extra_keys = set(agent_params.keys()) - set(hp_keys) - {"name", "_target_"}
                        for key in sorted(extra_keys):
                            val = agent_params[key]
                            if val is not None and not isinstance(val, (dict, list)):
                                f.write(f"| `{key}` | `{val}` |\n")
                        f.write("\n")
                    else:
                        f.write("_No configuration found for this run._\n\n")

            print(f"  Saved: {hp_path}")

        if cfg.get("include_time_report", True):
            time_csv_path = output_dir / "time_report.csv"
            # Remove legacy markdown time report if present
            old_time_md = output_dir / "time_report.md"
            if old_time_md.exists():
                try:
                    old_time_md.unlink()
                except Exception:
                    pass

            timing_rows = []

            for method, versions in sorted(runs_data.items()):
                method_label = clean_label(method)
                norm_method = method.replace("/", "_")
                times = []
                gpu_device = None
                peak_vram = None
                aliases = get_method_aliases(method)

                for v_name, df in sorted(versions.items()):
                    t_sec = None
                    v_num = v_name.replace("version_", "")

                    json_candidates = []
                    for alias in aliases:
                        json_candidates.extend(
                            [
                                Path("results/logs") / group / clean_exp / alias / v_name / "runtime.json",
                                Path("results/checkpoints") / group / clean_exp / alias / v_num / "runtime.json",
                                Path("results/logs") / group / clean_exp / alias / "runtime.json",
                                Path("results/checkpoints") / group / clean_exp / alias / "runtime.json",
                                Path("results/checkpoints") / group / clean_exp / alias / "0" / "runtime.json",
                            ]
                        )

                    for json_path in json_candidates:
                        if json_path.exists():
                            try:
                                with open(json_path) as jf:
                                    rdata = json.load(jf)
                                    t_sec = float(rdata.get("training_time_seconds", 0.0))
                                    if "gpu_device" in rdata and not gpu_device:
                                        gpu_device = rdata["gpu_device"]
                                    if "gpu_peak_alloc_gb" in rdata and (
                                        peak_vram is None or rdata["gpu_peak_alloc_gb"] > peak_vram
                                    ):
                                        peak_vram = float(rdata["gpu_peak_alloc_gb"])
                                    if t_sec > 0:
                                        break
                            except Exception:
                                pass

                    # 2. Fall back to metrics.csv training_time_seconds if present
                    if (t_sec is None or t_sec == 0) and "training_time_seconds" in df.columns:
                        vals = df["training_time_seconds"].dropna()
                        if not vals.empty:
                            t_sec = float(vals.iloc[-1])

                    if t_sec is not None and t_sec > 0:
                        times.append((v_name, t_sec))

                # If no times found per-version, check all runtime.json under checkpoint/log dirs across aliases
                if not times:
                    for alias in aliases:
                        for scan_dir in [
                            Path("results/checkpoints") / group / clean_exp / alias,
                            Path("results/logs") / group / clean_exp / alias,
                        ]:
                            if scan_dir.exists():
                                for r_json in sorted(scan_dir.rglob("runtime.json")):
                                    try:
                                        with open(r_json) as jf:
                                            rdata = json.load(jf)
                                            t_sec = float(rdata.get("training_time_seconds", 0.0))
                                            if "gpu_device" in rdata and not gpu_device:
                                                gpu_device = rdata["gpu_device"]
                                            if "gpu_peak_alloc_gb" in rdata and (
                                                peak_vram is None or rdata["gpu_peak_alloc_gb"] > peak_vram
                                            ):
                                                peak_vram = float(rdata["gpu_peak_alloc_gb"])
                                            if t_sec > 0:
                                                times.append((r_json.parent.name, t_sec))
                                    except Exception:
                                        pass

                # If still no times found, check Slurm output logs for execution time
                if not times:
                    for slurm_dir in [
                        Path("results/logs/slurm") / group / clean_exp,
                        Path("results/logs/slurm") / clean_exp,
                    ]:
                        if slurm_dir.exists():
                            for alias in aliases:
                                for out_file in sorted(slurm_dir.glob(f"*{alias}*.out")):
                                    try:
                                        with open(out_file) as sf:
                                            text = sf.read()
                                        import re

                                        time_match = re.search(r"Total execution time:\s*([0-9.]+)\s*seconds", text)
                                        if time_match:
                                            t_sec = float(time_match.group(1))
                                            if t_sec > 0:
                                                times.append((out_file.stem, t_sec))
                                        gpu_match = re.search(r"Device:\s*([^\n]+)", text)
                                        if gpu_match and not gpu_device:
                                            gpu_device = gpu_match.group(1).strip()
                                        vram_match = re.search(r"Peak Allocated:\s*([0-9.]+)\s*GB", text)
                                        if vram_match and peak_vram is None:
                                            peak_vram = float(vram_match.group(1))
                                    except Exception:
                                        pass

                if times:
                    avg_time = sum(t for _, t in times) / len(times)
                    formatted_avg = format_duration(avg_time)
                    timing_rows.append(
                        {
                            "method_raw": method,
                            "method": method_label,
                            "num_runs": len(times),
                            "avg_time_sec": avg_time,
                            "formatted_avg": formatted_avg,
                            "gpu_device": gpu_device or "CPU",
                            "peak_vram_gb": peak_vram,
                            "details": times,
                        }
                    )
                else:
                    timing_rows.append(
                        {
                            "method_raw": method,
                            "method": method_label,
                            "num_runs": 0,
                            "avg_time_sec": None,
                            "formatted_avg": "N/A",
                            "gpu_device": gpu_device or "CPU",
                            "peak_vram_gb": peak_vram,
                            "details": [],
                        }
                    )

            has_gpu_data = any(row["peak_vram_gb"] is not None for row in timing_rows)

            csv_records = []
            for row in timing_rows:
                csv_rec = {
                    "Method": row["method"],
                    "Raw_Method": row["method_raw"],
                    "Runs": row["num_runs"],
                    "Avg_Time_Seconds": row["avg_time_sec"] if row["avg_time_sec"] is not None else "",
                    "Formatted_Time": row["formatted_avg"],
                }
                if has_gpu_data:
                    csv_rec["Peak_VRAM_GB"] = row["peak_vram_gb"] if row["peak_vram_gb"] is not None else ""
                    csv_rec["GPU_Device"] = row["gpu_device"]
                csv_records.append(csv_rec)

            pd.DataFrame(csv_records).to_csv(time_csv_path, index=False)
            print(f"  Saved: {time_csv_path}")

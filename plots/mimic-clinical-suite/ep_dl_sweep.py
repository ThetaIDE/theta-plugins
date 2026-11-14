#!/usr/bin/env python3
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt

from src.app.api.plotting import BasePlotter, get_style

DISP_MAP = {
    "lstm_no_v": "LSTM (no V)",
    "lstm_with_v": "LSTM (with V)",
    "transformer_no_v": "Transformer (no V)",
    "transformer_with_v": "Transformer (with V)",
}


class EpDlSweepPlotter(BasePlotter):
    def __init__(self, *, paths):
        super().__init__("ep_dl_sweep", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        if not cfg.get("enabled", True):
            return

        json_files = list(output_dir.glob("metrics_*.json"))
        if not json_files:
            json_files = list(output_dir.rglob("metrics_*.json"))
        if not json_files:
            return

        all_results = {}
        for json_file in json_files:
            if json_file.stat().st_size == 0:
                continue
            try:
                m_key = json_file.stem.replace("metrics_", "")
                disp_name = DISP_MAP.get(m_key, m_key)
                with open(json_file) as f:
                    data = json.load(f)
                    if data.get("tau"):
                        all_results[disp_name] = data
            except Exception as e:
                print(f"Warning loading {json_file}: {e}")

        if not all_results:
            return

        print(f"\n--- Generating DL Sweep Consolidated Plots ({len(all_results)} model configs) ---")

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        markers = ["o", "s", "^", "D", "v", "P"]
        model_keys_sorted = sorted(all_results.keys())

        # 4-panel
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for idx, m_name in enumerate(model_keys_sorted):
            res = all_results[m_name]
            tau_arr = np.array(res["tau"])
            mean_arr = np.array(res["auc"])
            sem_arr = np.array(res["auc_sem"])
            style = get_style(m_name)
            c = style.get("color", colors[idx % len(colors)])
            m = style.get("marker", markers[idx % len(markers)])
            axes[0, 0].plot(tau_arr, mean_arr, marker=m, color=c, label=m_name, linewidth=2)
            axes[0, 0].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=c, alpha=0.15)
        axes[0, 0].set_title("AUC-ROC vs. Lead Time (τ)", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Lead Time (hours early - τ)", fontsize=11)
        axes[0, 0].set_ylabel("AUC-ROC", fontsize=11)
        axes[0, 0].grid(True, linestyle="--", alpha=0.6)
        axes[0, 0].legend(fontsize=10)

        for idx, m_name in enumerate(model_keys_sorted):
            res = all_results[m_name]
            tau_arr = np.array(res["tau"])
            mean_arr = np.array(res["auprc"])
            sem_arr = np.array(res["auprc_sem"])
            style = get_style(m_name)
            c = style.get("color", colors[idx % len(colors)])
            m = style.get("marker", markers[idx % len(markers)])
            axes[0, 1].plot(tau_arr, mean_arr, marker=m, color=c, label=m_name, linewidth=2)
            axes[0, 1].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=c, alpha=0.15)
        axes[0, 1].set_title("AUPRC (PR-AUC) vs. Lead Time (τ)", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Lead Time (hours early - τ)", fontsize=11)
        axes[0, 1].set_ylabel("AUPRC", fontsize=11)
        axes[0, 1].grid(True, linestyle="--", alpha=0.6)
        axes[0, 1].legend(fontsize=10)

        for idx, m_name in enumerate(model_keys_sorted):
            res = all_results[m_name]
            tau_arr = np.array(res["tau"])
            mean_arr = np.array(res["f1_opt"])
            sem_arr = np.array(res["f1_opt_sem"])
            style = get_style(m_name)
            c = style.get("color", colors[idx % len(colors)])
            m = style.get("marker", markers[idx % len(markers)])
            axes[1, 0].plot(tau_arr, mean_arr, marker=m, color=c, label=m_name, linewidth=2)
            axes[1, 0].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=c, alpha=0.15)
        axes[1, 0].set_title("Optimal F1-Score (θ*) vs. Lead Time (τ)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Lead Time (hours early - τ)", fontsize=11)
        axes[1, 0].set_ylabel("Optimal F1-Score", fontsize=11)
        axes[1, 0].grid(True, linestyle="--", alpha=0.6)
        axes[1, 0].legend(fontsize=10)

        for idx, m_name in enumerate(model_keys_sorted):
            res = all_results[m_name]
            tau_arr = np.array(res["tau"])
            mean_arr = np.array(res.get("f1_05", res.get("f1_opt")))
            sem_arr = np.array(res.get("f1_05_sem", res.get("f1_opt_sem")))
            style = get_style(m_name)
            c = style.get("color", colors[idx % len(colors)])
            m = style.get("marker", markers[idx % len(markers)])
            axes[1, 1].plot(tau_arr, mean_arr, marker=m, color=c, label=m_name, linewidth=2)
            axes[1, 1].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=c, alpha=0.15)
        axes[1, 1].set_title("Standard F1-Score (θ=0.5) vs. Lead Time (τ)", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Lead Time (hours early - τ)", fontsize=11)
        axes[1, 1].set_ylabel("F1-Score (θ=0.5)", fontsize=11)
        axes[1, 1].grid(True, linestyle="--", alpha=0.6)
        axes[1, 1].legend(fontsize=10)

        plt.tight_layout()
        plot_4panel_path = output_dir / "4panel.png"
        plt.savefig(plot_4panel_path, dpi=200)
        plt.close()

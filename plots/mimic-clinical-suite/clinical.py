#!/usr/bin/env python3
"""
plot/clinical.py — MIMIC Clinical Policy Validation Metrics Plotter.

Generates comparative trajectory plots across training epochs for clinical RL telemetry:
1. val_admin_rate.png: Antibiotic administration rate vs. clinician baseline (1.51%).
2. val_q_mean.png: Conservative Q-value push-down dynamics across validation checkpoints.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.app.api.plotting import BasePlotter, clean_label, get_style_info


class ClinicalPlotter(BasePlotter):
    def __init__(self, *, paths):
        super().__init__("clinical", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        runs_data = self.load_metrics(group, exp_id)
        if not runs_data:
            print(f"Notice [clinical]: No log data found for '{exp_id}' in group '{group}'.")
            return

        out_dir = output_dir / "clinical"
        out_dir.mkdir(parents=True, exist_ok=True)

        clean_exp = Path(exp_id).stem
        clean_title = clean_exp.replace("_", " ").title()

        dpi = cfg.get("dpi", 300)
        figsize = tuple(cfg.get("figsize", [9, 5]))
        clinician_baseline = cfg.get("clinician_baseline", 1.51)

        print("\n==================================================")
        print(f"=== Generating Clinical Telemetry Plots for '{exp_id}' ===")
        print("==================================================")

        # ----------------------------------------------------
        # 1. Validation Antibiotic Administration Rate
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        has_admin_data = False

        for method_name, versions in sorted(runs_data.items()):
            epoch_data = {}
            for v_name, df in versions.items():
                if "val/admin_rate" in df.columns and "epoch" in df.columns:
                    valid_df = df.dropna(subset=["val/admin_rate", "epoch"])
                    if not valid_df.empty:
                        for _, row in valid_df.iterrows():
                            ep = int(row["epoch"])
                            rate = float(row["val/admin_rate"]) * 100.0
                            epoch_data.setdefault(ep, []).append(rate)

            if epoch_data:
                has_admin_data = True
                display_name = clean_label(method_name)
                color, ls, _ = get_style_info(method_name)

                sorted_epochs = sorted(epoch_data.keys())
                means = [np.mean(epoch_data[ep]) for ep in sorted_epochs]

                if any(len(epoch_data[ep]) > 1 for ep in sorted_epochs):
                    sems = [np.std(epoch_data[ep]) / np.sqrt(len(epoch_data[ep])) for ep in sorted_epochs]
                    ax.plot(
                        sorted_epochs,
                        means,
                        label=display_name,
                        color=color,
                        linestyle=ls,
                        linewidth=2.0,
                        marker="o",
                        markersize=6,
                    )
                    ax.fill_between(
                        sorted_epochs,
                        np.array(means) - np.array(sems),
                        np.array(means) + np.array(sems),
                        color=color,
                        alpha=0.15,
                    )
                else:
                    ax.plot(
                        sorted_epochs,
                        means,
                        label=display_name,
                        color=color,
                        linestyle=ls,
                        linewidth=2.0,
                        marker="o",
                        markersize=6,
                    )

        if has_admin_data:
            ax.axhline(
                clinician_baseline,
                color="black",
                linestyle="--",
                linewidth=1.8,
                label=f"ICU Clinician Baseline ({clinician_baseline:.2f}%)",
            )
            ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
            ax.set_ylabel("Antibiotic Administration Rate (%)", fontsize=11, fontweight="bold")
            ax.set_title(f"{clean_title}: Validation Antibiotic Administration Rate", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="-", alpha=0.25)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.95)
            fig.tight_layout()
            admin_path = out_dir / "val_admin_rate.png"
            plt.savefig(admin_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {admin_path}")
        else:
            plt.close()

        # ----------------------------------------------------
        # 2. Conservative Q-Value Push-Down (val/q_mean)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        has_q_data = False

        for method_name, versions in sorted(runs_data.items()):
            epoch_data = {}
            for v_name, df in versions.items():
                if "val/q_mean" in df.columns and "epoch" in df.columns:
                    valid_df = df.dropna(subset=["val/q_mean", "epoch"])
                    if not valid_df.empty:
                        for _, row in valid_df.iterrows():
                            ep = int(row["epoch"])
                            q_val = float(row["val/q_mean"])
                            epoch_data.setdefault(ep, []).append(q_val)

            if epoch_data:
                has_q_data = True
                display_name = clean_label(method_name)
                color, ls, _ = get_style_info(method_name)

                sorted_epochs = sorted(epoch_data.keys())
                means = [np.mean(epoch_data[ep]) for ep in sorted_epochs]

                if any(len(epoch_data[ep]) > 1 for ep in sorted_epochs):
                    sems = [np.std(epoch_data[ep]) / np.sqrt(len(epoch_data[ep])) for ep in sorted_epochs]
                    ax.plot(
                        sorted_epochs,
                        means,
                        label=display_name,
                        color=color,
                        linestyle=ls,
                        linewidth=2.0,
                        marker="s",
                        markersize=6,
                    )
                    ax.fill_between(
                        sorted_epochs,
                        np.array(means) - np.array(sems),
                        np.array(means) + np.array(sems),
                        color=color,
                        alpha=0.15,
                    )
                else:
                    ax.plot(
                        sorted_epochs,
                        means,
                        label=display_name,
                        color=color,
                        linestyle=ls,
                        linewidth=2.0,
                        marker="s",
                        markersize=6,
                    )

        if has_q_data:
            ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
            ax.set_ylabel("Mean Q-Value", fontsize=11, fontweight="bold")
            ax.set_title(f"{clean_title}: Conservative Q-Value Push-Down (val/q_mean)", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="-", alpha=0.25)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.95)
            fig.tight_layout()
            q_path = out_dir / "val_q_mean.png"
            plt.savefig(q_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {q_path}")
        else:
            plt.close()

        print("==================================================\n")

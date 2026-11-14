#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.app.core.plotting.base import BasePlotter, clean_label, get_style_info, moving_average


class LossesPlotter(BasePlotter):
    def __init__(self):
        super().__init__("losses")

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        metrics = cfg.get(
            "metrics",
            [
                "losses/total_loss",
                "losses/bellman_loss",
                "losses/cql_loss",
                "losses/entropy",
                "losses/blend_entropy",
                "losses/actor_loss",
                "losses/q_loss",
                "losses/value_loss",
            ],
        )

        runs_data = self.load_metrics(group, exp_id)
        if not runs_data:
            print(f"No log data found for experiment '{exp_id}' in group '{group}'.")
            return

        window = cfg.get("smoothing_window", 10)
        dpi = cfg.get("dpi", 300)
        figsize = tuple(cfg.get("figsize", [8, 5]))
        x_axis_col = cfg.get("x_axis", "transitions")

        subdir = cfg.get("output_subdir", "losses")
        losses_base_dir = (output_dir / subdir) if subdir else output_dir
        losses_base_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Generating Separate Loss Plots for '{exp_id}' ===")

        for method_name, versions in sorted(runs_data.items()):
            agent_loss_dir = losses_base_dir / method_name
            color, ls, marker = get_style_info(method_name)
            display_name = clean_label(method_name)
            agent_saved = False

            for metric in metrics:
                all_x = []
                all_y = []
                used_xlabel = None
                for v_name, df in versions.items():
                    if metric in df.columns:
                        valid_df = df.dropna(subset=[metric])
                        if not valid_df.empty:
                            x_vals = None
                            if x_axis_col in df.columns and df[x_axis_col].notna().any():
                                full_x = df[x_axis_col].interpolate(method="linear").ffill().bfill()
                                s_x = full_x.loc[valid_df.index]
                                if not s_x.empty and s_x.nunique() > 1 and not s_x.isna().any():
                                    x_vals = s_x.values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", x_axis_col.replace("_", " ").title())
                            if x_vals is None:
                                if "step" in valid_df.columns and valid_df["step"].nunique() > 1:
                                    x_vals = valid_df["step"].values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", "Training Steps")
                                elif "epoch" in valid_df.columns and valid_df["epoch"].nunique() > 1:
                                    x_vals = valid_df["epoch"].values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", "Epoch")
                                else:
                                    x_vals = valid_df.index.values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", "Index")
                            y_vals = valid_df[metric].values
                            all_x.append(x_vals)
                            all_y.append(y_vals)

                if all_y:
                    agent_loss_dir.mkdir(parents=True, exist_ok=True)
                    plt.figure(figsize=figsize)
                    if len(all_y) > 1:
                        min_len = min(len(y) for y in all_y)
                        trimmed = np.array([y[:min_len] for y in all_y])
                        y_mean = np.mean(trimmed, axis=0)
                        y_sem = np.std(trimmed, axis=0) / np.sqrt(len(all_y))
                        y_smoothed = moving_average(y_mean, window)
                        sem_smoothed = moving_average(y_sem, window)
                        x_plot = all_x[0][: len(y_smoothed)]
                        plt.plot(x_plot, y_smoothed, label=display_name, color=color, linestyle=ls, linewidth=2.0)
                        plt.fill_between(
                            x_plot, y_smoothed - sem_smoothed, y_smoothed + sem_smoothed, color=color, alpha=0.15
                        )
                    else:
                        y_smoothed = moving_average(all_y[0], window)
                        x_plot = all_x[0][: len(y_smoothed)]
                        plt.plot(x_plot, y_smoothed, label=display_name, color=color, linestyle=ls, linewidth=2.0)

                    xlabel = cfg.get("xlabel") or used_xlabel or "Training Steps"
                    plt.xlabel(xlabel)

                    metric_clean_name = metric.split("/")[-1].replace("_", " ").title()
                    if isinstance(cfg.get("ylabel"), dict):
                        ylabel = cfg["ylabel"].get(metric, metric_clean_name)
                    elif cfg.get("ylabel"):
                        ylabel = str(cfg["ylabel"])
                    else:
                        ylabel = metric_clean_name
                    plt.ylabel(ylabel)

                    if isinstance(cfg.get("title"), dict):
                        title = cfg["title"].get(metric, f"{display_name}: {metric_clean_name}")
                    elif cfg.get("title"):
                        try:
                            title = str(cfg["title"]).format(
                                exp_id=exp_id.upper(),
                                display_name=display_name,
                                method=method_name,
                                metric=metric,
                                clean_metric=metric_clean_name,
                            )
                        except Exception:
                            title = str(cfg["title"])
                    else:
                        title = f"{display_name}: {metric_clean_name}"
                    plt.title(title)
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc="upper right")
                    plt.tight_layout()

                    safe_metric_name = metric.replace("/", "_")
                    out_path = agent_loss_dir / f"{safe_metric_name}.png"
                    plt.savefig(out_path, dpi=dpi)
                    plt.close()
                    print(f"  Saved ({method_name}): {out_path}")
                    agent_saved = True

            if agent_loss_dir.exists() and not agent_saved and not any(agent_loss_dir.iterdir()):
                try:
                    agent_loss_dir.rmdir()
                except Exception:
                    pass

        # Also generate comparative plots across all methods in losses/
        loss_cfg = dict(cfg)
        loss_cfg.setdefault("output_subdir", subdir)
        loss_cfg.setdefault("filename_prefix", "comparison_")
        self.plot_metric_series(exp_id, group, output_dir, metrics, loss_cfg)

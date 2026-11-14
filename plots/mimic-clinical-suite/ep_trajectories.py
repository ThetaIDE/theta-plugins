#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

from src.app.api.plotting import BasePlotter
from .ep_cache import get_ep_eval_data
from src.app.api.packages import provider_export
plot_ep_shock_over_tau = provider_export('mimic-early-prediction', 'plot_ep_shock_over_tau')


class EpTrajectoriesPlotter(BasePlotter):
    def __init__(self, *, paths):
        super().__init__("ep_trajectories", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        if not cfg.get("enabled", True):
            return

        data = get_ep_eval_data(exp_id, cfg, group, output_dir, paths=self.workspace)
        if not data:
            return

        ep_shock_results = data.get("ep_shock_results")

        if ep_shock_results:
            print(f"Generating EP Trajectories Plots in {output_dir}")
            plot_ep_shock_over_tau(ep_shock_results, output_dir)

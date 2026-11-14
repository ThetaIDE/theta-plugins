#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

from src.app.core.plotting.base import BasePlotter
from .ep_cache import get_ep_eval_data
from src.app.core.provider_loader import provider_object
plot_ep_shock_over_tau = provider_object('mimic-early-prediction', 'eval_logic.py:plot_ep_shock_over_tau')


class EpTrajectoriesPlotter(BasePlotter):
    def __init__(self):
        super().__init__("ep_trajectories")

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        if not cfg.get("enabled", True):
            return

        data = get_ep_eval_data(exp_id, cfg, group, output_dir)
        if not data:
            return

        ep_shock_results = data.get("ep_shock_results")

        if ep_shock_results:
            print(f"Generating EP Trajectories Plots in {output_dir}")
            plot_ep_shock_over_tau(ep_shock_results, output_dir)

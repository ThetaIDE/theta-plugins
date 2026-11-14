#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

from src.app.api.plotting import BasePlotter
from .ep_cache import get_ep_eval_data
from src.app.api.packages import provider_export
plot_agreement_vs_shock = provider_export('mimic-early-prediction', 'plot_agreement_vs_shock')


class EpAgreementPlotter(BasePlotter):
    def __init__(self, *, paths):
        super().__init__("ep_agreement", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        if not cfg.get("enabled", True):
            return

        data = get_ep_eval_data(exp_id, cfg, group, output_dir, paths=self.workspace)
        if not data:
            return

        rl_agreements = data.get("rl_agreements")
        y = data.get("y")

        if rl_agreements:
            print(f"Generating Agreement Plots in {output_dir}")
            plot_agreement_vs_shock(rl_agreements, y, output_dir)

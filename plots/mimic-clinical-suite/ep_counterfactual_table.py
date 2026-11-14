#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

from src.app.api.plotting import BasePlotter
from .ep_cache import get_ep_eval_data
from src.app.api.packages import provider_export
write_counterfactual_table = provider_export('mimic-early-prediction', 'write_counterfactual_table')


class EpCounterfactualTablePlotter(BasePlotter):
    def __init__(self, *, paths):
        super().__init__("ep_counterfactual_table", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        if not cfg.get("enabled", True):
            return

        data = get_ep_eval_data(exp_id, cfg, group, output_dir, paths=self.workspace)
        if not data:
            return

        cf_data = data.get("cf_data")

        if cf_data:
            print(f"Generating Counterfactual Tables in {output_dir}")
            csv_path = output_dir / "counterfactual_summary.csv"
            txt_path = output_dir / "counterfactual_summary.txt"
            write_counterfactual_table(cf_data, csv_path, txt_path)

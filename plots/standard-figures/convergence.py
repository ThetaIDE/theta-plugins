#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.app.core.plotting.base import BasePlotter, clean_label, get_style_info, moving_average


class ConvergencePlotter(BasePlotter):
    def __init__(self):
        super().__init__("convergence")

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        metrics = cfg.get("metrics", ["eval/reward", "train/reward", "train/length"])
        self.plot_metric_series(exp_id, group, output_dir, metrics, cfg)

#!/usr/bin/env python3
"""
plot/clinical_alignment.py — MIMIC Clinical Policy Alignment & Septic Shock Plotter.

Evaluates trained RL policies against clinician treatment decisions and patient
septic shock outcomes in ICU time-series trajectories.

Outputs:
  - Clinical action alignment: precision, recall, F1, windowed F1, AUC-ROC, AUPRC
  - Septic shock outcome analysis: agreement_vs_shock.png
  - Interval progression analysis: intervals/interagreement_vs_shock_[method]_progression.png
  - Tabular metrics: method_comparison.csv
  - Visual figures: clinical_agreement.png (opt-in)
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


import argparse

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.app.api.plotting import BasePlotter, clean_label, get_canonical_method_name, get_method_aliases
from src.app.api.plotting import get_style as get_method_style


def compute_trajectory_agreement(
    p_clin: np.ndarray,
    p_pol: np.ndarray,
    metric: str = "windowed_jaccard",
    window_hours: int = 3,
    empty_union_score: float = 1.0,
) -> float:
    """Compute per-patient trajectory agreement score between clinician and policy.

    Args:
        p_clin: Clinician binary treatment actions for this patient trajectory.
        p_pol: Policy binary treatment actions for this patient trajectory.
        metric: 'windowed_jaccard', 'jaccard', or 'accuracy'.
        window_hours: Matching window (±hours) for windowed Jaccard.
        empty_union_score: Score to assign when neither clinician nor policy intervened.

    Returns:
        float agreement score in [0.0, 1.0].
    """
    if metric == "accuracy":
        return float((p_clin == p_pol).mean()) if len(p_clin) > 0 else 0.0

    clin_pos = np.where(p_clin == 1)[0]
    pol_pos = np.where(p_pol == 1)[0]

    # Both withheld treatment across entire trajectory
    if len(clin_pos) == 0 and len(pol_pos) == 0:
        return empty_union_score
    # One administered while other never did
    if len(clin_pos) == 0 or len(pol_pos) == 0:
        return 0.0

    if metric == "jaccard" or window_hours == 0:
        tp = int(np.isin(pol_pos, clin_pos).sum())
        fp = len(pol_pos) - tp
        fn = len(clin_pos) - tp
    else:  # windowed_jaccard
        tp = sum(1 for c in clin_pos if np.min(np.abs(pol_pos - c)) <= window_hours)
        fn = len(clin_pos) - tp
        fp = sum(1 for p in pol_pos if np.min(np.abs(clin_pos - p)) > window_hours)

    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else empty_union_score


class ClinicalAlignmentPlotter(BasePlotter):
    def __init__(self, *, paths):
        super().__init__("clinical_alignment", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        self._run_mimic_eval(exp_id, cfg, group, clean_exp, output_dir)

    def _discover_checkpoints(self, exp_id: str, group: str, clean_exp: str):
        checkpoints = self.workspace.results / "checkpoints"
        ckpt_root = checkpoints / group / clean_exp
        if not ckpt_root.exists():
            ckpt_root = checkpoints / clean_exp
        if not ckpt_root.exists():
            return {}, {}

        exp_cfg = self.get_experiment_config(exp_id)
        active_aliases, has_active_filter = self.get_active_aliases(exp_cfg)

        method_ckpts = {}
        method_interval_ckpts = {}
        import re

        for method_dir in sorted(ckpt_root.iterdir()):
            if method_dir.is_dir():
                m_name = method_dir.name
                if not self.is_method_active(m_name, active_aliases, has_active_filter):
                    continue

                canon = get_canonical_method_name(m_name)

                # 1. Best checkpoint: the tuner's promoted winner, else any saved best model.
                best_ckpt = None
                promoted = method_dir / "best_model.ckpt"
                if promoted.is_file():
                    best_ckpt = promoted

                if not best_ckpt:
                    ckpts = list(method_dir.rglob("best_model*.ckpt"))
                    if ckpts:
                        best_ckpt = ckpts[0]

                if best_ckpt:
                    if canon not in method_ckpts or m_name == canon:
                        method_ckpts[canon] = best_ckpt

                # 2. Interval checkpoints
                seen_epochs = set()
                intervals = []
                candidate_files = list(method_dir.rglob("epoch_*.ckpt"))
                for ifile in candidate_files:
                    match_eq = re.search(r"epoch=(\d+)", ifile.name)
                    match_us = re.search(r"epoch_(\d+)", ifile.name)
                    if match_eq:
                        ep = int(match_eq.group(1)) + 1
                    elif match_us:
                        ep = int(match_us.group(1))
                    else:
                        continue
                    if ep not in seen_epochs:
                        seen_epochs.add(ep)
                        intervals.append((ep, ifile))
                intervals.sort(key=lambda x: x[0])
                if intervals:
                    if canon not in method_interval_ckpts or m_name == canon:
                        method_interval_ckpts[canon] = intervals

        return method_ckpts, method_interval_ckpts

    def _load_agent(self, path, dev):
        from src.app.api.agents import get_agent_class, list_registered_agents

        candidates = []
        preferred = ("cql", "iql", "blendrl_cql", "blendrl_iql", "cew")
        for name in (*preferred, *[n for n in list_registered_agents() if n not in preferred]):
            try:
                cand = get_agent_class(name)
            except Exception:
                continue
            if cand not in candidates and hasattr(cand, "load_from_checkpoint"):
                candidates.append(cand)

        last_error = None
        for cls in candidates:
            try:
                ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False)
                ag.to(dev)
                ag.eval()
                return ag
            except Exception as e:
                last_error = e
                try:
                    ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False, strict=False)
                    ag.to(dev)
                    ag.eval()
                    return ag
                except Exception as e2:
                    last_error = e2
                    continue
        if last_error is not None:
            print(f"  [clinical_alignment] Checkpoint load error for {path}: {last_error}")
        return None

    def _get_probs_and_actions(self, ag, obs_b):
        if hasattr(ag, "get_action_probs"):
            probs = ag.get_action_probs(obs_b)
            acts = ag.get_action(obs_b) if hasattr(ag, "get_action") else torch.argmax(probs, dim=-1)
            return probs, acts

        # Fallback for unmigrated or raw models
        is_cql = "cql" in ag.__class__.__name__.lower() or "cql" in str(getattr(ag, "algorithm", "")).lower()
        use_actor = bool(ag.get_cfg("use_actor", False)) if hasattr(ag, "get_cfg") else getattr(ag, "use_actor", False)

        if hasattr(ag, "is_modular") and ag.is_modular:
            logic_obs = (
                ag._prepare_logic_obs(obs_b)
                if hasattr(ag, "_prepare_logic_obs")
                else obs_b.unsqueeze(1).repeat(1, 2, 1)
            )
            if is_cql and not use_actor and hasattr(ag.model, "get_q_values"):
                q_vals = ag.model.get_q_values(obs_b, logic_obs)
                probs = torch.softmax(q_vals, dim=-1)
                acts = torch.argmax(q_vals, dim=-1)
                return probs, acts
            elif use_actor and hasattr(ag.model, "actor"):
                probs, _ = ag.model.actor(obs_b, logic_obs)
                acts = torch.argmax(probs, dim=-1)
                return probs, acts
            elif hasattr(ag.model, "get_q_values"):
                q_vals = ag.model.get_q_values(obs_b, logic_obs)
                probs = torch.softmax(q_vals, dim=-1)
                acts = torch.argmax(q_vals, dim=-1)
                return probs, acts
        elif is_cql and not use_actor and hasattr(ag, "q_network"):
            q = ag.q_network.get_q_values(obs_b) if hasattr(ag.q_network, "get_q_values") else ag.q_network(obs_b)
            probs = torch.softmax(q, dim=-1)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "actor") and hasattr(ag.actor, "get_action_probs"):
            probs = ag.actor.get_action_probs(obs_b)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "fuzzy_model") and ag.fuzzy_model is not None:
            q = ag.fuzzy_model(obs_b.to("cpu"))
            probs = torch.softmax(q, dim=-1).to(obs_b.device)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "q_network"):
            if hasattr(ag.q_network, "get_action_probs"):
                probs = ag.q_network.get_action_probs(obs_b)
            else:
                q = ag.q_network(obs_b)
                probs = torch.softmax(q, dim=-1)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "model") and hasattr(ag.model, "get_q_values"):
            q = ag.model.get_q_values(obs_b)
            probs = torch.softmax(q, dim=-1)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        else:
            out = ag.get_action_and_value(obs_b)
            act = out[0] if isinstance(out, (tuple, list)) else out
            n_acts = 3 if obs_b.shape[-1] >= 123 else 2
            probs = torch.zeros((obs_b.shape[0], n_acts), device=obs_b.device)
            probs.scatter_(1, act.unsqueeze(1).long(), 1.0)
            return probs, act

    def _run_mimic_eval(self, exp_id: str, cfg: dict, group: str, clean_exp: str, output_dir: Path):
        """Clinical policy alignment and septic shock evaluation for MIMIC datasets."""
        env_ds = (
            cfg.get("env", {}).get("dataset_name", "mimic_lazy_0_interventions_balanced.npz")
            if isinstance(cfg.get("env"), dict)
            else "mimic_lazy_0_interventions_balanced.npz"
        )
        npz_candidate = self.workspace.datasets / "mimic" / env_ds
        if not npz_candidate.exists():
            mode_path = cfg.get("mode", {}).get("dataset_path", "")
            if mode_path:
                cand = Path(mode_path).with_suffix(".npz")
                if cand.exists():
                    npz_candidate = cand

        if not npz_candidate.exists():
            raise FileNotFoundError(f"[clinical_alignment]: MIMIC dataset file '{npz_candidate}' not found.")

        print("\n==========================================================================================")
        print(f"=== Running MIMIC Clinical Alignment Evaluation for '{exp_id}' ===")
        print("==========================================================================================")
        print(
            f"  Loading dataset '{npz_candidate.name}' ({npz_candidate.stat().st_size / (1024 * 1024):.1f} MB)...",
            flush=True,
        )
        data = np.load(npz_candidate, allow_pickle=True)
        X = data["X"]  # (N, 240, 49)
        mask = data["mask"]  # (N, 240, 1)

        valid_mask = mask.squeeze(-1) != -1
        all_obs = X[:, :, :46][valid_mask]
        all_clin_acts = X[:, :, 47][valid_mask].astype(int)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        total_steps = len(all_clin_acts)
        print(
            f"  Loaded {total_steps:,} valid clinical transitions across {X.shape[0]:,} patient trajectories.",
            flush=True,
        )

        agreement_metric = str(cfg.get("agreement_metric", "windowed_jaccard")).lower().strip()
        window_hours = int(cfg.get("window_hours", 3))
        empty_union_score = float(cfg.get("empty_union_score", 1.0))

        if agreement_metric == "windowed_jaccard":
            metric_label = f"Clinician – RL Policy Jaccard Similarity (±{window_hours}h) (%)"
            metric_title = f"Clinician Jaccard (±{window_hours}h)"
        elif agreement_metric == "jaccard":
            metric_label = "Clinician – RL Policy Jaccard Similarity (%)"
            metric_title = "Clinician Jaccard Similarity"
        else:
            metric_label = "Clinician – RL Policy Agreement (%)"
            metric_title = "Clinician Agreement"

        print(f"  Alignment Metric: {agreement_metric} (window: ±{window_hours}h, empty_union: {empty_union_score})")

        cache_dir = output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        metric_tag = f"{agreement_metric}_w{window_hours}" if "window" in agreement_metric else agreement_metric
        cache_path = cache_dir / f"clinical_alignment_cache_{metric_tag}.npz"
        history_cache_path = cache_dir / f"clinical_alignment_history_{metric_tag}.npz"

        # Fallback to legacy cache files if accuracy requested and legacy files exist
        if not cache_path.exists() and agreement_metric == "accuracy":
            legacy_cache = cache_dir / "clinical_alignment_cache.npz"
            if legacy_cache.exists():
                cache_path = legacy_cache
        if not history_cache_path.exists() and agreement_metric == "accuracy":
            legacy_hist = cache_dir / "clinical_alignment_history.npz"
            if legacy_hist.exists():
                history_cache_path = legacy_hist

        remake = cfg.get("remake", False)
        use_cache = cfg.get("use_cache", False) and not remake
        num_patients = X.shape[0]
        outcomes = data["y"].squeeze() if "y" in data else np.zeros(num_patients)
        patient_agreements = {}

        print(f"  Scanning policy checkpoints for '{clean_exp}' in group '{group}'...", flush=True)
        method_ckpts, method_interval_ckpts = self._discover_checkpoints(exp_id, group, clean_exp)
        print(
            f"  Discovered {len(method_ckpts)} best checkpoints and {len(method_interval_ckpts)} interval checkpoint sets.",
            flush=True,
        )
        interval_agreements = {}

        if use_cache and history_cache_path.exists():
            print(f"  Loading cached clinical alignment interval history from {history_cache_path}")
            try:
                hist_data = np.load(history_cache_path, allow_pickle=True)
                for k in hist_data.files:
                    if "__ep__" in k:
                        m_part, ep_part = k.split("__ep__", 1)
                        try:
                            ep_num = int(ep_part)
                            interval_agreements.setdefault(m_part, {})[ep_num] = hist_data[k]
                        except Exception:
                            pass
            except Exception as e:
                print(f"  Notice: could not load interval history cache: {e}")

        if use_cache and cache_path.exists():
            print(f"  Loading cached clinical alignment data from {cache_path}")
            try:
                cached_data = np.load(cache_path, allow_pickle=True)
                for k in cached_data.files:
                    if k != "outcomes":
                        patient_agreements[k] = cached_data[k]
                if "outcomes" in cached_data.files:
                    outcomes = cached_data["outcomes"]
            except Exception as e:
                print(f"  Notice: could not load alignment cache: {e}")

        if not method_ckpts and not method_interval_ckpts and not interval_agreements and not patient_agreements:
            print(f"Notice [clinical_alignment]: No policy checkpoints or cached data found for '{clean_exp}'")
            return

        csv_path = output_dir / "method_comparison.csv"
        results = []
        if use_cache and csv_path.exists():
            try:
                raw_records = pd.read_csv(csv_path).to_dict("records")
                seen_methods = {}
                for r in raw_records:
                    orig_m = str(r.get("Method", "")).strip()
                    clean_m = orig_m if orig_m == "Clinician (Baseline)" else clean_label(orig_m)
                    r["Method"] = clean_m
                    # Later records for the same method replace earlier ones (e.g. replacing old CQL (Standard MLP) with DNN)
                    seen_methods[clean_m] = r
                results = list(seen_methods.values())
            except Exception:
                results = []

        win_key_f1 = f"Windowed F1 (±{window_hours}h)"
        win_key_jaccard = f"Windowed Jaccard (±{window_hours}h) %"

        if not results:
            # 1. Clinician Baseline
            clin_admin_rate = (all_clin_acts == 1).mean() * 100.0
            results.append(
                {
                    "Method": "Clinician (Baseline)",
                    "Accuracy %": 100.0,
                    "Admin Rate %": float(clin_admin_rate),
                    "AUC-ROC": 1.0000,
                    "AUPRC": 1.0000,
                    "Precision": 1.0000,
                    "Recall": 1.0000,
                    "F1 Score": 1.0000,
                    "Best F1": 1.0000,
                    "Jaccard Similarity %": 100.0,
                    win_key_f1: 1.0000,
                    win_key_jaccard: 100.0,
                    "Windowed Recall %": 100.0,
                    "Opt Threshold": 0.5000,
                }
            )

        batch_size = 10000
        method_opt_thresholds = {}
        for method_name, ckpt_path in sorted(method_ckpts.items()):
            ckpt_p = Path(ckpt_path)
            ckpt_newer = (
                cache_path.exists() and ckpt_p.exists() and (ckpt_p.stat().st_mtime > cache_path.stat().st_mtime)
            )
            if (
                use_cache
                and not ckpt_newer
                and method_name in patient_agreements
                and any(r.get("Method") == clean_label(method_name) for r in results)
            ):
                continue
            if ckpt_newer or not use_cache:
                results = [r for r in results if r.get("Method") != clean_label(method_name)]
            agent = self._load_agent(ckpt_path, device)
            if agent is None:
                print(f"  Warning [clinical_alignment]: Could not load checkpoint {ckpt_path}")
                continue

            print(f"  Evaluating {clean_label(method_name)} ({ckpt_p.name})...", flush=True)

            all_admin_probs = []
            all_policy_acts = []

            with torch.no_grad():
                for b_start in range(0, total_steps, batch_size):
                    b_end = min(b_start + batch_size, total_steps)
                    obs_batch = torch.tensor(all_obs[b_start:b_end], dtype=torch.float32).to(device)

                    probs, policy_acts_tensor = self._get_probs_and_actions(agent, obs_batch)
                    policy_acts = policy_acts_tensor.cpu().numpy()
                    if probs.shape[-1] > 1:
                        admin_probs = probs[:, 1].cpu().numpy()
                    else:
                        admin_probs = probs.squeeze().cpu().numpy()

                    all_admin_probs.extend(admin_probs)
                    all_policy_acts.extend(policy_acts)

            all_admin_probs = np.array(all_admin_probs)

            # Option A: Calibrated clinical decision thresholding
            # Determine optimal operating threshold from Precision-Recall curve
            try:
                p_thresh, r_thresh, thresholds = precision_recall_curve(all_clin_acts, all_admin_probs)
                f1_scores = 2 * (p_thresh * r_thresh) / (p_thresh + r_thresh + 1e-8)
                best_idx = np.argmax(f1_scores)
                best_f1 = float(f1_scores[best_idx])
                opt_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
            except Exception:
                best_f1 = 0.0
                opt_thresh = 0.5

            method_opt_thresholds[method_name] = opt_thresh

            # Apply calibrated threshold for clinical policy actions
            all_policy_acts = (all_admin_probs >= opt_thresh).astype(int)

            matches = (all_policy_acts == all_clin_acts).sum()
            accuracy = (matches / total_steps) * 100.0
            admin_rate = (all_policy_acts == 1).mean() * 100.0

            tp = ((all_policy_acts == 1) & (all_clin_acts == 1)).sum()
            fp = ((all_policy_acts == 1) & (all_clin_acts == 0)).sum()
            fn = ((all_policy_acts == 0) & (all_clin_acts == 1)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            try:
                auc_roc = float(roc_auc_score(all_clin_acts, all_admin_probs))
            except Exception:
                auc_roc = float("nan")

            try:
                auprc = float(average_precision_score(all_clin_acts, all_admin_probs))
            except Exception:
                auprc = float("nan")

            # Windowed agreement calculation (±window_hours)
            step_idx = 0
            patient_agrs = []
            win_tp = 0
            win_fp = 0
            win_fn = 0
            total_clin_pos = 0

            for p_idx in range(num_patients):
                p_valid = valid_mask[p_idx]
                p_len = p_valid.sum()
                if p_len == 0:
                    continue

                p_clin = all_clin_acts[step_idx : step_idx + p_len]
                p_pol = all_policy_acts[step_idx : step_idx + p_len]
                step_idx += p_len

                p_agree = compute_trajectory_agreement(
                    p_clin,
                    p_pol,
                    metric=agreement_metric,
                    window_hours=window_hours,
                    empty_union_score=empty_union_score,
                )
                patient_agrs.append(p_agree)

                clin_pos_indices = np.where(p_clin == 1)[0]
                pol_pos_indices = np.where(p_pol == 1)[0]
                total_clin_pos += len(clin_pos_indices)

                for c_pos in clin_pos_indices:
                    if len(pol_pos_indices) > 0 and np.min(np.abs(pol_pos_indices - c_pos)) <= window_hours:
                        win_tp += 1
                    else:
                        win_fn += 1

                for p_pos in pol_pos_indices:
                    if len(clin_pos_indices) == 0 or np.min(np.abs(clin_pos_indices - p_pos)) > window_hours:
                        win_fp += 1

            patient_agreements[method_name] = np.array(patient_agrs)
            jaccard = tp / (tp + fp + fn + 1e-8)
            win_precision = win_tp / (win_tp + win_fp + 1e-8)
            win_recall = win_tp / (total_clin_pos + 1e-8)
            win_f1 = 2 * (win_precision * win_recall) / (win_precision + win_recall + 1e-8)
            win_jaccard = win_tp / (win_tp + win_fp + win_fn + 1e-8)

            results.append(
                {
                    "Method": clean_label(method_name),
                    "Accuracy %": float(accuracy),
                    "Admin Rate %": float(admin_rate),
                    "AUC-ROC": auc_roc,
                    "AUPRC": auprc,
                    "Precision": float(precision),
                    "Recall": float(recall),
                    "F1 Score": float(f1),
                    "Best F1": best_f1,
                    "Jaccard Similarity %": float(jaccard * 100.0),
                    win_key_f1: float(win_f1),
                    win_key_jaccard: float(win_jaccard * 100.0),
                    "Windowed Recall %": float(win_recall * 100.0),
                    "Opt Threshold": opt_thresh,
                }
            )

        # Evaluate interval checkpoints if discovered
        new_interval_data = False
        for method_name, intervals in sorted(method_interval_ckpts.items()):
            opt_thresh = method_opt_thresholds.get(method_name, 0.5)
            for ep, ckpt_path in intervals:
                ckpt_p = Path(ckpt_path)
                hist_newer = (
                    history_cache_path.exists()
                    and ckpt_p.exists()
                    and (ckpt_p.stat().st_mtime > history_cache_path.stat().st_mtime)
                )
                if (
                    use_cache
                    and not hist_newer
                    and method_name in interval_agreements
                    and ep in interval_agreements[method_name]
                ):
                    continue
                if hist_newer or not use_cache:
                    new_interval_data = True
                agent = self._load_agent(ckpt_path, device)
                if agent is None:
                    continue
                all_admin_probs_ep = []
                with torch.no_grad():
                    for b_start in range(0, total_steps, batch_size):
                        b_end = min(b_start + batch_size, total_steps)
                        obs_batch = torch.tensor(all_obs[b_start:b_end], dtype=torch.float32).to(device)
                        probs, _ = self._get_probs_and_actions(agent, obs_batch)
                        if probs.shape[-1] > 1:
                            p_adm = probs[:, 1].cpu().numpy()
                        else:
                            p_adm = probs.squeeze().cpu().numpy()
                        all_admin_probs_ep.extend(p_adm)
                all_admin_probs_ep = np.array(all_admin_probs_ep)
                all_policy_acts = (all_admin_probs_ep >= opt_thresh).astype(int)

                step_idx = 0
                ep_agrs = []
                for p_idx in range(num_patients):
                    p_valid = valid_mask[p_idx]
                    p_len = p_valid.sum()
                    if p_len == 0:
                        continue
                    p_clin = all_clin_acts[step_idx : step_idx + p_len]
                    p_pol = all_policy_acts[step_idx : step_idx + p_len]
                    step_idx += p_len
                    p_agree = compute_trajectory_agreement(
                        p_clin,
                        p_pol,
                        metric=agreement_metric,
                        window_hours=window_hours,
                        empty_union_score=empty_union_score,
                    )
                    ep_agrs.append(p_agree)

                interval_agreements.setdefault(method_name, {})[ep] = np.array(ep_agrs)
                new_interval_data = True

        # Ensure patient_agreements models are included in interval_agreements as final if not already present
        for m_name, p_agr in patient_agreements.items():
            if m_name not in interval_agreements:
                interval_agreements[m_name] = {0: p_agr}
            else:
                max_ep = max(interval_agreements[m_name].keys()) if interval_agreements[m_name] else 0
                if max_ep not in interval_agreements[m_name]:
                    interval_agreements[m_name][max_ep + 1] = p_agr

        # And ensure interval_agreements models are included in patient_agreements (at latest available epoch)
        for m_name, ep_dict in interval_agreements.items():
            if m_name not in patient_agreements and ep_dict:
                latest_ep = max(ep_dict.keys())
                patient_agreements[m_name] = ep_dict[latest_ep]

        if results:
            df = pd.DataFrame(results)
            csv_path = output_dir / "method_comparison.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved MIMIC method comparison: {csv_path}")

        if method_ckpts or patient_agreements:
            np.savez(cache_path, outcomes=outcomes, **patient_agreements)
            print(f"  Cached clinical alignment data: {cache_path}")

        # Save interval history cache
        if interval_agreements and (new_interval_data or not history_cache_path.exists()):
            save_hist = {}
            for m, ep_dict in interval_agreements.items():
                for ep, p_agr in ep_dict.items():
                    save_hist[f"{m}__ep__{ep}"] = p_agr
            np.savez(history_cache_path, **save_hist)
            print(f"  Cached clinical alignment interval history: {history_cache_path}")

        # Color deduplication across active methods
        all_active_methods = list(set(list(patient_agreements.keys()) + list(interval_agreements.keys())))
        color_map = self._deduplicate_colors(all_active_methods)

        # 1. Clinician Agreement % Bar Chart (Opt-in only, excluded from defaults)
        requested_plots = cfg.get("plots", [])
        if isinstance(requested_plots, dict):
            requested_plots = list(requested_plots.keys())
        elif isinstance(requested_plots, (list, tuple)):
            requested_plots = [str(p).lower() for p in requested_plots]
        else:
            requested_plots = []

        if any(k in requested_plots for k in ["clinical_agreement", "agreement_bar", "agreement"]) and results:
            fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.8), 5.5))
            methods = [r["Method"] for r in results]
            if "jaccard" in agreement_metric:
                score_key = win_key_jaccard if "window" in agreement_metric else "Jaccard Similarity %"
                accuracies = [float(r.get(score_key, r.get("Accuracy %", 0.0))) for r in results]
                bar_ylabel = f"Clinician {metric_title} (%)"
                bar_title = f"MIMIC Treatment Action {metric_title} ({clean_exp})"
            else:
                accuracies = [float(r.get("Accuracy %", 0.0)) for r in results]
                bar_ylabel = "Clinician Agreement (%)"
                bar_title = f"MIMIC Treatment Action Agreement ({clean_exp})"

            bar_colors = []
            for r in results:
                m_name = r["Method"]
                if "Clinician" in m_name:
                    bar_colors.append("#756bb1")
                else:
                    bar_colors.append(color_map.get(m_name, get_method_style(m_name).get("color") or "tab:blue"))

            bars = ax.bar(
                methods, accuracies, color=bar_colors, width=0.55, edgecolor="#333333", linewidth=1.0, alpha=0.85
            )
            ax.set_ylabel(bar_ylabel, fontsize=12, fontweight="bold")
            ax.set_title(bar_title, fontsize=13, fontweight="bold")
            ax.set_ylim(0, 110)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            plt.xticks(rotation=15, ha="right", fontsize=10, fontweight="bold")

            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            fig.tight_layout()
            plot_path = output_dir / "clinical_agreement.png"
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {plot_path}")

        # 2. Clinician Agreement vs Septic Shock Outcome Analysis
        plot_all = len(requested_plots) == 0
        should_plot_shock = plot_all or any(
            k in requested_plots for k in ["agreement_vs_shock", "agreement_vs_shock_absolute", "shock", "absolute"]
        )
        should_plot_prog = plot_all or any(
            k in requested_plots
            for k in [
                "agreement_vs_shock",
                "agreement_vs_shock_progression",
                "interagreement",
                "progression",
                "interagreement_vs_shock",
                "intervals",
            ]
        )

        n_evals = int(cfg.get("n_evals", cfg.get("num_evals", 100)))
        has_valid_data = bool(patient_agreements) and len(outcomes) > 0
        if should_plot_shock and has_valid_data:
            self._plot_agreement_vs_shock(
                patient_agreements,
                outcomes,
                output_dir,
                clean_exp,
                color_map,
                n_evals=n_evals,
                metric_label=metric_label,
                metric_title=metric_title,
            )

        if should_plot_prog and interval_agreements and len(outcomes) > 0:
            for m_name, epochs_dict in sorted(interval_agreements.items()):
                self._plot_method_agreement_vs_shock_progression(
                    m_name,
                    epochs_dict,
                    outcomes,
                    output_dir,
                    clean_exp,
                    color_map,
                    n_evals=n_evals,
                    metric_label=metric_label,
                    metric_title=metric_title,
                )

        if (
            any(
                k in requested_plots
                for k in ["agreement_vs_shock_absolute_progression", "combined_progression", "all_progression"]
            )
            and interval_agreements
            and len(outcomes) > 0
        ):
            self._plot_agreement_vs_shock_absolute_progression(
                interval_agreements,
                outcomes,
                output_dir,
                clean_exp,
                color_map,
                n_evals=n_evals,
                metric_label=metric_label,
                metric_title=metric_title,
            )

        print("==========================================================================================\n")

    def _compute_binned_shock_stats(
        self, p_agr: np.ndarray, outcomes: np.ndarray, bins: np.ndarray, n_evals: int = 100, seed: int = 42
    ) -> tuple:
        """Compute mean and standard deviation of septic shock rates across multiple evaluation resamples."""
        N = len(outcomes)
        num_bins = len(bins) - 1
        if N == 0:
            return np.full(num_bins, np.nan), np.full(num_bins, np.nan)

        agr_pct = p_agr * 100.0 if np.max(p_agr) <= 1.0 else p_agr

        if n_evals <= 1:
            means, stds = [], []
            for i in range(num_bins):
                low, high = bins[i], bins[i + 1]
                m = (agr_pct >= low) & (agr_pct <= high if i == num_bins - 1 else agr_pct < high)
                pts = outcomes[m]
                if len(pts) >= 5:
                    means.append(float(np.mean(pts) * 100.0))
                    stds.append(float(np.std(pts) / np.sqrt(len(pts))) * 100.0 if len(pts) > 1 else 0.0)
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            return np.array(means), np.array(stds)

        rng = np.random.default_rng(seed)
        boot_idx = rng.choice(N, size=(n_evals, N), replace=True)
        boot_agrs = agr_pct[boot_idx]
        boot_outs = outcomes[boot_idx]

        means, stds = [], []
        for i in range(num_bins):
            low, high = bins[i], bins[i + 1]
            m = (boot_agrs >= low) & (boot_agrs <= high if i == num_bins - 1 else boot_agrs < high)
            counts = m.sum(axis=1)
            sums = (boot_outs * m).sum(axis=1)
            valid = counts >= 5
            if valid.sum() >= 5:
                rates = (sums[valid] / counts[valid]) * 100.0
                means.append(float(np.mean(rates)))
                stds.append(float(np.std(rates)))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        return np.array(means), np.array(stds)

    def _deduplicate_colors(self, methods: list[str]) -> dict[str, str]:
        """Ensures every method plotted has a distinct color, dynamically reassigning duplicates if needed."""
        fallback_palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#08519c",
            "#d95f02",
            "#018571",
            "#6a3d9a",
            "#e7298a",
        ]
        color_map = {}
        used_colors = set()
        for m in methods:
            style = get_method_style(m)
            c = style.get("color")
            if not c or c in used_colors:
                for candidate in fallback_palette:
                    if candidate not in used_colors:
                        c = candidate
                        break
                else:
                    c = f"C{len(color_map) % 10}"
            color_map[m] = c
            used_colors.add(c)
        return color_map

    def _plot_agreement_vs_shock(
        self,
        patient_agreements: dict,
        outcomes: np.ndarray,
        output_dir: Path,
        clean_exp: str,
        color_map: dict,
        n_evals: int = 100,
        metric_label: str = "Clinician – RL Policy Agreement (%)",
        metric_title: str = "Clinician Agreement",
    ):
        """Plot septic shock rate vs clinician agreement across all methods with connected error bands and background histogram."""
        bins = np.linspace(0, 100, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        method_items = sorted(patient_agreements.items())
        K = len(method_items)
        total_bar_width = 7.5
        bar_width = total_bar_width / max(K, 1)

        # Background histogram
        for k_idx, (m_name, p_agr) in enumerate(method_items):
            agr_pct = p_agr * 100.0 if np.max(p_agr) <= 1.0 else p_agr
            color = color_map.get(m_name) or get_method_style(m_name).get("color") or "tab:blue"

            counts = []
            for b_idx in range(10):
                low, high = bins[b_idx], bins[b_idx + 1]
                mask_bin = (agr_pct >= low) & (agr_pct <= high if b_idx == 9 else agr_pct < high)
                counts.append(mask_bin.sum())

            offset = (k_idx - (K - 1) / 2.0) * bar_width
            bar_x = bin_centers + offset
            ax2.bar(
                bar_x, counts, width=bar_width * 0.9, color=color, alpha=0.18, edgecolor=color, linewidth=0.8, zorder=1
            )

        ax2.set_ylabel("Patient Trajectory Count (Histogram)", fontsize=12, fontweight="bold", color="#555555")
        ax2.tick_params(axis="y", labelcolor="#555555")

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        # Foreground lines with connected error bands
        for k_idx, (m_name, p_agr) in enumerate(method_items):
            color = color_map.get(m_name) or get_method_style(m_name).get("color") or "tab:blue"
            marker = get_method_style(m_name).get("marker") or "o"
            ls = get_method_style(m_name).get("linestyle") or "-"
            label = clean_label(m_name)

            means, stds = self._compute_binned_shock_stats(p_agr, outcomes, bins, n_evals=n_evals)
            valid = ~np.isnan(means)
            if valid.sum() > 0:
                ax1.plot(
                    bin_centers[valid],
                    means[valid],
                    marker=marker,
                    color=color,
                    linestyle=ls,
                    label=label,
                    linewidth=2.5,
                    markersize=7,
                    zorder=3,
                )
                ax1.fill_between(
                    bin_centers[valid],
                    np.maximum(0, means[valid] - stds[valid]),
                    np.minimum(100, means[valid] + stds[valid]),
                    color=color,
                    alpha=0.12,
                    zorder=2,
                )

        ax1.set_xlabel(metric_label, fontsize=12, fontweight="bold")
        ax1.set_ylabel("True Septic Shock Rate (%)", fontsize=12, fontweight="bold")
        ax1.set_xticks(np.arange(0, 101, 10))
        ax1.set_xlim(-2, 102)
        ax1.set_ylim(0, 105)
        ax1.grid(True, linestyle="--", alpha=0.4, zorder=0)

        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1, labels1, fontsize=10, loc="best", framealpha=0.9)
        ax1.set_title(f"Septic Shock Rate vs. {metric_title} ({clean_exp})", fontsize=13, fontweight="bold")

        fig.tight_layout()
        plot_path = output_dir / "agreement_vs_shock.png"
        plt.savefig(plot_path, dpi=200)
        # Also save clinician_agreement_vs_shock_absolute.png for backward compatibility
        abs_path = output_dir / "clinician_agreement_vs_shock_absolute.png"
        plt.savefig(abs_path, dpi=200)
        plt.close()
        print(f"  Saved Agreement Plot:      {plot_path}")

    @staticmethod
    def _get_progression_color(base_color, ratio: float):
        """Generate a smooth chromatic color shift along training progression.

        Shifts the hue and saturation from a distinct, brighter starting tint
        at ratio=0.0 (initial training epoch) towards the method's deep, saturated signature
        base color at ratio=1.0 (final converged epoch), making the direction of convergence
        immediately distinguishable.
        """
        import matplotlib.colors as mcolors

        rgb = mcolors.to_rgb(base_color)
        h, s, v = mcolors.rgb_to_hsv(rgb)

        # Warm/cool hue rotation:
        # Warm colors (reds, oranges, h < 0.25) shift towards bright gold/yellow (h ~ 0.16)
        # Cool colors (greens, blues, purples, h >= 0.25) shift towards bright cyan/teal/lime (h - 0.12)
        if h < 0.25:
            h_start = min(0.18, h + 0.08)
        else:
            h_start = (h - 0.12) % 1.0

        h_curr = (h_start + ratio * (h - h_start)) % 1.0
        s_curr = 0.48 + (s - 0.48) * (ratio**0.8)
        v_curr = 0.88 + (v - 0.88) * (ratio**0.8)
        return mcolors.hsv_to_rgb((h_curr, s_curr, v_curr))

    def _plot_method_agreement_vs_shock_progression(
        self,
        m_name: str,
        epochs_dict: dict,
        outcomes: np.ndarray,
        output_dir: Path,
        clean_exp: str,
        color_map: dict,
        n_evals: int = 100,
        metric_label: str = "Clinician – RL Policy Agreement (%)",
        metric_title: str = "Clinician Agreement",
    ):
        """Plot septic shock rate progression for a single method across training intervals/epochs with connected error bands,
        saved to intervals/interagreement_vs_shock_[method]_progression.png."""
        from matplotlib.patches import Patch

        if not epochs_dict or len(outcomes) == 0:
            return

        intervals_dir = output_dir / "intervals"
        intervals_dir.mkdir(parents=True, exist_ok=True)

        bins = np.linspace(0, 100, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Background trajectory distribution for this method at its latest epoch
        sorted_epochs = sorted(epochs_dict.keys())
        latest_ep = sorted_epochs[-1]
        p_agr_latest = epochs_dict[latest_ep]
        agr_pct_latest = p_agr_latest * 100.0 if np.max(p_agr_latest) <= 1.0 else p_agr_latest

        counts = []
        for b_idx in range(10):
            low, high = bins[b_idx], bins[b_idx + 1]
            m = (agr_pct_latest >= low) & (agr_pct_latest <= high if b_idx == 9 else agr_pct_latest < high)
            counts.append(m.sum())

        ax2.bar(
            bin_centers, counts, width=7.5, color="#cfd8dc", alpha=0.35, edgecolor="#90a4ae", linewidth=1.0, zorder=1
        )
        ax2.set_ylabel("Patient Trajectory Count (Histogram)", fontsize=12, fontweight="bold", color="#555555")
        ax2.tick_params(axis="y", labelcolor="#555555")
        ax2.set_ylim(0, max(1, max(counts)) * 1.35)

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        base_color = color_map.get(m_name) or get_method_style(m_name).get("color") or "tab:blue"
        marker = get_method_style(m_name).get("marker") or "o"
        ls = get_method_style(m_name).get("linestyle") or "-"

        K = len(sorted_epochs)
        for idx, ep in enumerate(sorted_epochs):
            p_agr = epochs_dict[ep]
            means, stds = self._compute_binned_shock_stats(p_agr, outcomes, bins, n_evals=n_evals)

            valid = ~np.isnan(means)
            if valid.sum() > 0:
                ratio = idx / (K - 1) if K > 1 else 1.0
                c = self._get_progression_color(base_color, ratio) if K > 1 else base_color
                alpha = 0.45 + 0.55 * ratio if K > 1 else 1.0
                lw = 1.5 + 1.2 * ratio if K > 1 else 2.2
                ms = 4.5 + 2.5 * ratio if K > 1 else 6.5
                zorder = 5 + idx

                if K == 1:
                    lbl = f"Epoch {ep}"
                elif K <= 8:
                    lbl = (
                        f"Epoch {ep} (Initial)"
                        if idx == 0
                        else f"Epoch {ep} (Final)"
                        if idx == K - 1
                        else f"Epoch {ep}"
                    )
                else:
                    step = max(1, K // 5)
                    if idx == 0:
                        lbl = f"Epoch {ep} (Initial)"
                    elif idx == K - 1:
                        lbl = f"Epoch {ep} (Final)"
                    elif idx % step == 0:
                        lbl = f"Epoch {ep}"
                    else:
                        lbl = None

                ax1.plot(
                    bin_centers[valid],
                    means[valid],
                    marker=marker,
                    linestyle=ls,
                    linewidth=lw,
                    markersize=ms,
                    label=lbl,
                    color=c,
                    alpha=alpha,
                    zorder=zorder,
                )
                ax1.fill_between(
                    bin_centers[valid],
                    np.maximum(0, means[valid] - stds[valid]),
                    np.minimum(100, means[valid] + stds[valid]),
                    color=c,
                    alpha=0.14 * alpha,
                    zorder=zorder - 1,
                )

        ax1.set_xlabel(metric_label, fontsize=12, fontweight="bold")
        ax1.set_ylabel("True Septic Shock Rate (%)", fontsize=12, fontweight="bold")
        ax1.set_xticks(np.arange(0, 101, 10))
        ax1.set_xlim(-2, 102)
        ax1.set_ylim(0, 105)
        ax1.grid(True, linestyle="--", alpha=0.4, zorder=0)

        display_name = clean_label(m_name)
        ax1.set_title(
            f"Septic Shock Rate vs. {metric_title} — {display_name} Progression ({clean_exp})",
            fontsize=12.5,
            fontweight="bold",
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        dist_patch = Patch(facecolor="#90a4ae", alpha=0.35, label="Trajectory Count (Histogram)")
        ax1.legend(
            lines1 + [dist_patch],
            labels1 + ["Trajectory Count (Histogram)"],
            loc="lower left",
            fontsize=9.5,
            framealpha=0.92,
        )

        fig.tight_layout()
        safe_m_name = m_name.replace("/", "_")
        prog_path = intervals_dir / f"interagreement_vs_shock_{safe_m_name}_progression.png"
        plt.savefig(prog_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Progression Plot:    {prog_path}")

    def _plot_agreement_vs_shock_absolute_progression(
        self,
        interval_agreements: dict,
        outcomes: np.ndarray,
        output_dir: Path,
        clean_exp: str,
        color_map: dict,
        filename: str | None = None,
        n_evals: int = 100,
        metric_label: str = "Clinician – RL Policy Agreement (%)",
        metric_title: str = "Clinician Agreement",
    ):
        """Plot septic shock rate progression across training intervals for all methods with connected error bands."""
        from matplotlib.patches import Patch

        bins = np.linspace(0, 100, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax2 = ax1.twinx()

        # Background trajectory distribution across all methods at their final available interval
        counts_list = []
        for m_name, ep_dict in interval_agreements.items():
            if ep_dict:
                latest_ep = max(ep_dict.keys())
                p_agr = ep_dict[latest_ep]
                agr_pct = p_agr * 100.0 if np.max(p_agr) <= 1.0 else p_agr
                c = []
                for i in range(10):
                    low, high = bins[i], bins[i + 1]
                    m = (agr_pct >= low) & (agr_pct <= high if i == 9 else agr_pct < high)
                    c.append(m.sum())
                counts_list.append(c)
        if counts_list:
            counts = np.mean(counts_list, axis=0)
        else:
            counts = np.zeros(10)

        ax2.bar(
            bin_centers, counts, width=7.5, color="#cfd8dc", alpha=0.35, edgecolor="#90a4ae", linewidth=1.0, zorder=1
        )
        ax2.set_ylabel("Patient Trajectory Count (Histogram)", fontsize=12, fontweight="bold", color="#546e7a")
        ax2.tick_params(axis="y", labelcolor="#546e7a")
        ax2.set_ylim(0, max(1, max(counts)) * 1.35)

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        for m_name, epochs_dict in sorted(interval_agreements.items()):
            color = color_map.get(m_name) or get_method_style(m_name).get("color") or "tab:blue"
            marker = get_method_style(m_name).get("marker") or "o"
            ls = get_method_style(m_name).get("linestyle") or "-"

            sorted_epochs = sorted(epochs_dict.keys())
            K = len(sorted_epochs)

            for idx, ep in enumerate(sorted_epochs):
                p_agr = epochs_dict[ep]
                means, stds = self._compute_binned_shock_stats(p_agr, outcomes, bins, n_evals=n_evals)

                valid = ~np.isnan(means)
                if valid.sum() > 0:
                    ratio = idx / (K - 1) if K > 1 else 1.0
                    c = self._get_progression_color(color, ratio) if K > 1 else color
                    alpha = 0.40 + 0.60 * ratio if K > 1 else 1.0
                    lw = 1.3 + 1.1 * ratio if K > 1 else 2.2
                    ms = 3.5 + 2.5 * ratio if K > 1 else 6.5
                    zorder = 5 + idx

                    if K == 1:
                        lbl = clean_label(m_name)
                    elif K <= 6:
                        if idx == 0:
                            lbl = f"{clean_label(m_name)} (Ep {ep} - Initial)"
                        elif idx == K - 1:
                            lbl = f"{clean_label(m_name)} (Ep {ep} - Final)"
                        else:
                            lbl = f"{clean_label(m_name)} (Ep {ep})"
                    elif idx == 0:
                        lbl = f"{clean_label(m_name)} (Ep {ep} - Initial)"
                    elif idx == K - 1:
                        lbl = f"{clean_label(m_name)} (Ep {ep} - Final)"
                    else:
                        lbl = None

                    ax1.plot(
                        bin_centers[valid],
                        means[valid],
                        marker=marker,
                        linestyle=ls,
                        linewidth=lw,
                        markersize=ms,
                        label=lbl,
                        color=c,
                        alpha=alpha,
                        zorder=zorder,
                    )
                    ax1.fill_between(
                        bin_centers[valid],
                        np.maximum(0, means[valid] - stds[valid]),
                        np.minimum(100, means[valid] + stds[valid]),
                        color=c,
                        alpha=0.12 * alpha,
                        zorder=zorder - 1,
                    )

        ax1.set_xlabel(metric_label, fontsize=12, fontweight="bold")
        ax1.set_ylabel("True Septic Shock Rate (%)", fontsize=12, fontweight="bold")
        ax1.set_xticks(np.arange(0, 101, 10))
        ax1.set_xlim(-2, 102)
        ax1.set_ylim(0, 105)
        ax1.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax1.set_title(
            f"Septic Shock Rate vs. {metric_title} — Progression ({clean_exp})", fontsize=13, fontweight="bold"
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        dist_patch = Patch(facecolor="#90a4ae", alpha=0.35, label="Trajectory Count (Histogram)")
        ax1.legend(
            lines1 + [dist_patch],
            labels1 + ["Trajectory Count (Histogram)"],
            loc="upper left",
            bbox_to_anchor=(1.08, 1),
            fontsize=9.5,
            framealpha=0.95,
        )

        fig.tight_layout()
        prog_name = filename or "clinician_agreement_vs_shock_absolute_progression.png"
        prog_path = output_dir / prog_name
        plt.savefig(prog_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Combined Progression Plot: {prog_path}")

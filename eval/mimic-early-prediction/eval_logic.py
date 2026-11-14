"""
MIMIC Sepsis Early Prediction Evaluation

Generates:
1. Three agreement-vs-shock-rate plots (shock cohort, non-shock cohort, all patients)
   X-axis: policy-clinician trajectory agreement (%) bins
   Y-axis: true septic shock outcome rate (%)
2. One timestep graph with 5 lines (4 EP architectures + clinician)
   X-axis: lead time tau (hours before end-of-stay)
   Y-axis: average predicted shock probability (%)
3. A counterfactual summary table (CSV + formatted text)
"""

import argparse
import csv
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add root directory to path to allow importing src modules

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#  Method Styling — imported from plot.base
# ---------------------------------------------------------------------------
from src.app.core.plotting.base import clean_label
from src.app.core.plotting.base import get_style as get_method_style
from .model import (
    SepsisLSTM,
    SepsisTransformer,
    compute_volatility_features,
    evaluate_lstm_model,
    evaluate_transformer_model,
    normalize_features,
)


def pretty(name: str) -> str:
    return clean_label(name)


def color(name: str):
    return get_method_style(name).get("color") or "#1f77b4"


def marker(name: str) -> str:
    return get_method_style(name).get("marker") or "o"


def resolve_mimic_dataset(args):
    """Resolve the MIMIC .npz dataset using this component's dataset settings."""
    fname = getattr(args, "dataset_path", None) or getattr(args, "dataset_name", None)
    if not fname:
        fname = os.environ.get("MIMIC_DATASET_NAME", "mimic_lazy_0_interventions_balanced.npz")
    path = Path(fname)
    if path.is_file():
        return str(path.resolve())
    filename = path.name
    search_dirs = [
        Path(os.environ.get("MIMIC_DATASET_DIR", "")),
        Path(PROJECT_ROOT) / "in" / "datasets" / "mimic",
        Path(PROJECT_ROOT) / "in" / "datasets",
    ]
    for directory in search_dirs:
        candidate = directory / filename
        if str(directory) not in ("", ".") and candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"MIMIC dataset {filename!r} was not found in the configured dataset directories.")


def discover_single_policy(path: Path) -> dict:
    """Resolve a single RL policy checkpoint from a file or method directory.

    Accepts either:
      - A direct .ckpt file:   results/.../cql/0/best_model.ckpt
      - A method directory:    results/.../cql/   or   results/.../cql/0/

    Returns dict: { method_key: Path_to_best_ckpt }  (always length 0 or 1)
    """
    path = Path(path)
    if not path.exists():
        print(f"WARNING: Path {path} does not exist.")
        return {}

    if path.is_file():
        # e.g. .../cql/0/best_model.ckpt  → method = "cql"
        #      .../cql/best_model.ckpt    → method = "cql"
        method_name = path.parent.parent.name if path.parent.name.isdigit() else path.parent.name
        return {method_name: path}

    # Directory — pick the most-recently modified best_model*.ckpt inside it
    ckpts = sorted(
        list(path.glob("best_model*.ckpt")) + list(path.glob("*.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not ckpts:
        ckpts = sorted(list(path.rglob("best_model*.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        print(f"WARNING: No .ckpt files found under {path}.")
        return {}

    method_name = path.parent.name if path.name.isdigit() else path.name
    return {method_name: ckpts[0]}


def discover_policies_from_experiment_root(root: Path) -> dict:
    """Discover ALL RL policy checkpoints for a multi-method experiment.

    Expects the experiment checkpoint root directory, e.g.:
        results/checkpoints/mimic/mimic_tqn_all/
    which contains one subdirectory per trained method:
        cql/0/best_model.ckpt
        blendrl_cql_human_neural/0/best_model.ckpt
        ...

    Returns dict: { method_key: Path_to_best_ckpt }  (one entry per method found)
    """
    root = Path(root)
    if not root.exists():
        print(f"WARNING: Experiment checkpoint root {root} does not exist. No policies found.")
        return {}

    policies = {}
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        method_key = method_dir.name
        # Search method_dir itself, then any trial subdirs (e.g. 0/, 1/)
        ckpts = sorted(
            list(method_dir.glob("best_model*.ckpt")) + list(method_dir.glob("*.ckpt")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not ckpts:
            ckpts = sorted(list(method_dir.rglob("best_model*.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            policies[method_key] = ckpts[0]
        else:
            print(f"  [discover] Skipping '{method_key}': no .ckpt files found.")

    return policies


def discover_policy_checkpoints(checkpoint_root) -> dict:
    """Dispatcher for CLI convenience — resolves whichever checkpoint form was passed.

    Prefer calling the explicit functions directly in programmatic contexts:
      - discover_single_policy(path)               for a single file or method dir
      - discover_policies_from_experiment_root(dir) for a full experiment dir

    This wrapper handles all three CLI input forms and logs which case it resolved.
    """
    root = Path(checkpoint_root)
    if not root.exists():
        print(f"WARNING: Checkpoint path {root} does not exist. No policies found.")
        return {}

    if root.is_file():
        print("  [discover] Resolved as single .ckpt file.")
        return discover_single_policy(root)

    # Does the directory itself directly contain .ckpt files? → single method dir
    if list(root.glob("*.ckpt")) or list(root.glob("best_model*.ckpt")):
        print(f"  [discover] Resolved as single method directory: {root.name}")
        return discover_single_policy(root)

    # Otherwise assume it's an experiment root with per-method subdirs
    print("  [discover] Resolved as experiment root. Scanning method subdirs...")
    return discover_policies_from_experiment_root(root)


def load_policy_agent(ckpt_path, device):
    """Load a CQL or BlendRLCQL agent from checkpoint."""
    torch.serialization.add_safe_globals([getattr(sys.modules.get("omegaconf.dictconfig", None), "DictConfig", None)])

    try:
        from src.app.core.agent_registry import get_agent_class
        agent_cls = get_agent_class("cql")

        agent = agent_cls.load_from_checkpoint(str(ckpt_path), map_location=device, weights_only=False)
        agent.eval()
        return agent, "cql"
    except Exception as e:
        print(f"WARNING: Could not load checkpoint {ckpt_path}: {e}")
        return None, None


def get_policy_actions(agent, agent_type, obs_tensor, device):
    """Get action probabilities from an agent for a batch of observations.

    Returns (actions, admin_probs) — both numpy arrays of shape (N,).
    """
    with torch.no_grad():
        if hasattr(agent, "get_action_probs"):
            probs = agent.get_action_probs(obs_tensor)
            if hasattr(agent, "get_action"):
                actions = agent.get_action(obs_tensor).cpu().numpy()
            else:
                actions = torch.argmax(probs, dim=-1).cpu().numpy()
            admin_probs = probs[:, 1].cpu().numpy()
        elif hasattr(agent, "model") and hasattr(agent.model, "get_q_values"):
            q = agent.model.get_q_values(obs_tensor)
            probs = torch.softmax(q, dim=-1)
            actions = torch.argmax(probs, dim=-1).cpu().numpy()
            admin_probs = probs[:, 1].cpu().numpy()
        elif hasattr(agent, "q_network"):
            q = agent.q_network(obs_tensor)
            probs = torch.softmax(q, dim=-1)
            actions = torch.argmax(probs, dim=-1).cpu().numpy()
            admin_probs = probs[:, 1].cpu().numpy()
        elif hasattr(agent, "actor") and hasattr(agent.actor, "get_action_probs"):
            probs = agent.actor.get_action_probs(obs_tensor)
            actions = torch.argmax(probs, dim=-1).cpu().numpy()
            admin_probs = probs[:, 1].cpu().numpy()
        else:
            raise ValueError(f"Unknown or incompatible agent: {type(agent)}")
    return actions, admin_probs


def discover_ep_checkpoints(ep_ckpt_root):
    """Discover pre-trained Early Prediction model checkpoints from tuned_early_pred_sweep.

    Returns dict: { model_config_name: { tau: [list_of_ckpt_paths_per_split] } }
    """
    root = Path(ep_ckpt_root)
    if not root.exists():
        return {}

    ep_models = {}
    candidate_files = sorted(list(root.rglob("*.pt")) + list(root.rglob("*.ckpt")))
    seen = set()
    for pt_file in candidate_files:
        if pt_file in seen:
            continue
        seen.add(pt_file)
        name = pt_file.stem  # e.g. lstm_no_v_tau5_split3 or lstm_no_v_split3
        if "_tau" in name:
            parts = name.split("_tau")
            if len(parts) != 2:
                continue
            model_key = parts[0]
            tau_split = parts[1]
            ts = tau_split.split("_split")
            if len(ts) != 2:
                continue
            try:
                tau = int(ts[0])
            except ValueError:
                continue
        elif "_split" in name:
            parts = name.split("_split")
            if len(parts) != 2:
                continue
            model_key = parts[0]
            tau = "single_model"
        else:
            continue

        if model_key not in ep_models:
            ep_models[model_key] = {}
        if tau not in ep_models[model_key]:
            ep_models[model_key][tau] = []
        ep_models[model_key][tau].append(pt_file)

    return ep_models


# Map ep checkpoint keys back to friendly config names
EP_KEY_TO_CONFIG = {
    "lstm_no_v": ("LSTM (no V)", "lstm", False),
    "lstm_with_v": ("LSTM (with V)", "lstm", True),
    "transformer_no_v": ("Transformer (no V)", "transformer", False),
    "transformer_with_v": ("Transformer (with V)", "transformer", True),
}


def load_ep_model(ckpt_path, device):
    """Load a single EP model from a .pt or .ckpt checkpoint."""
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = data.get("hyper_parameters", {}) if isinstance(data, dict) else {}
    arch_name = hparams.get("architecture_name", "")

    m_type = data.get("model_type")
    if not m_type:
        if arch_name.startswith("transformer") or "transformer" in Path(ckpt_path).stem.lower():
            m_type = "transformer"
        else:
            m_type = "lstm"

    input_dim = data.get("input_dim", hparams.get("input_dim", 196))
    params = data.get("hyperparams", hparams)

    if m_type == "lstm":
        model = SepsisLSTM(
            input_dim=input_dim,
            hidden_dim=params.get("hidden_dim", 64),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.2),
            use_tcn_conv=params.get("use_tcn_conv", False),
            bidirectional=params.get("bidirectional", False),
        ).to(device)
    elif m_type == "transformer":
        d_model = params.get("d_model", 64)
        dim_ff = params.get("dim_feedforward", d_model * 2)
        model = SepsisTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=params.get("nhead", 4),
            num_layers=params.get("num_layers", 2),
            dim_feedforward=dim_ff,
            dropout=params.get("dropout", 0.1),
            norm_first=params.get("norm_first", True),
            pos_type=params.get("pos_type", "learned"),
            use_cls_token=params.get("use_cls_token", True),
            use_tcn_conv=params.get("use_tcn_conv", False),
        ).to(device)
    else:
        raise ValueError(f"Unknown EP model type: {m_type}")

    if "model_state_dict" in data:
        raw_sd = data["model_state_dict"]
    elif "state_dict" in data:
        raw_sd = data["state_dict"]
    else:
        raw_sd = data

    model_sd = {}
    for k, v in raw_sd.items():
        key = k[6:] if k.startswith("model.") else k
        if not key.startswith("loss_fn."):
            model_sd[key] = v

    model.load_state_dict(model_sd, strict=False)
    model.eval()
    return model, m_type, input_dim, data.get("opt_thresh", 0.5)


def predict_shock_probs_with_ep_models(ep_ckpts_for_tau, X_sequences, device):
    """Load all EP models for a given tau and average their predicted shock probabilities.

    Args:
        ep_ckpts_for_tau: list of .pt checkpoint paths (one per split)
        X_sequences: list of numpy arrays, each (seq_len, features)
        device: torch device

    Returns:
        mean_probs: numpy array of shape (N,) — average predicted P(shock) per patient
    """
    if len(X_sequences) == 0:
        return np.array([])

    # Apply z-score normalization matching the training pipeline
    all_steps = np.concatenate([s for s in X_sequences], axis=0)
    mean = np.mean(all_steps, axis=0, keepdims=True)
    std = np.std(all_steps, axis=0, keepdims=True) + 1e-6
    X_norm = [(s - mean) / std for s in X_sequences]

    all_probs = []
    for ckpt_path in ep_ckpts_for_tau:
        model, m_type, input_dim, _ = load_ep_model(ckpt_path, device)
        if m_type == "lstm":
            probs = evaluate_lstm_model(model, X_norm, input_dim, device=str(device))
        else:
            probs = evaluate_transformer_model(model, X_norm, input_dim, device=str(device))
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


# ---------------------------------------------------------------------------
#  Graph 1: Agreement vs Shock Rate (All Patients)
# ---------------------------------------------------------------------------


def plot_agreement_vs_shock(agreements, outcomes, report_dir):
    """Generate single agreement vs shock rate plot for All Patients,
    overlaying trajectory counts per bucket in a background bar plot.

    agreements: dict { method_key: np.array of per-patient agreement % }
    outcomes: np.array of true labels (1=shock, 0=non-shock)
    """
    bins = np.linspace(0, 100, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    first_method_counts = None

    for idx, (method_key, patient_agreements) in enumerate(agreements.items()):
        agr = patient_agreements
        out = outcomes

        means = []
        sems = []
        counts = []
        for b_idx in range(10):
            low, high = bins[b_idx], bins[b_idx + 1]
            if b_idx == 9:
                idx_mask = (agr >= low) & (agr <= high)
            else:
                idx_mask = (agr >= low) & (agr < high)
            pts = out[idx_mask]
            counts.append(len(pts))
            if len(pts) > 0:
                means.append(float(np.mean(pts)) * 100.0)
                sems.append(float(np.std(pts) / np.sqrt(len(pts))) * 100.0 if len(pts) > 1 else 0.0)
            else:
                means.append(np.nan)
                sems.append(0.0)

        if first_method_counts is None:
            first_method_counts = counts

        means_arr = np.array(means)
        sems_arr = np.array(sems)
        valid = ~np.isnan(means_arr)

        label = pretty(method_key)
        color = get_method_style(method_key)["color"]
        marker = get_method_style(method_key)["marker"]
        ax1.plot(
            bin_centers[valid], means_arr[valid], marker=marker, color=color, label=label, linewidth=2.5, markersize=7
        )
        ax1.fill_between(
            bin_centers[valid],
            means_arr[valid] - sems_arr[valid],
            means_arr[valid] + sems_arr[valid],
            color=color,
            alpha=0.12,
        )

    # Background bar chart for trajectory count
    if first_method_counts is not None:
        ax2.bar(
            bin_centers,
            first_method_counts,
            width=8,
            color="tab:blue",
            alpha=0.15,
            label="Trajectory Count in Bucket",
            zorder=1,
        )
        ax2.set_ylabel("Trajectory Count in Bucket", fontsize=13, fontweight="bold", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_xlabel("Clinician – RL Policy Agreement (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("True Septic Shock Rate (%)", fontsize=13, fontweight="bold")
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.grid(True, linestyle="--", alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="best")

    ax1.set_title("True Septic Shock Rate vs. Policy Agreement — All Patients", fontsize=14, fontweight="bold")

    fig.tight_layout()
    out_path = report_dir / "agreement_vs_shock.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved agreement plot: {out_path}")


# ---------------------------------------------------------------------------
#  Graph 2: EP Predicted Shock % over tau timesteps (3 cohort graphs)
# ---------------------------------------------------------------------------


def plot_ep_shock_over_tau(ep_shock_results, report_dir):
    """Plot average predicted shock% at each tau across 3 cohorts:
      1. All Patients
      2. Septic Shock Cohort (y=1)
      3. Non-Shock Cohort (y=0)

    ep_shock_results: dict { line_label: { "tau": [..], "all": {...}, "shock": {...}, "non_shock": {...} } }
    """
    cohort_configs = [
        ("All Patients", "all", "ep_shock_over_tau_all.png"),
        ("Septic Shock Cohort (y=1)", "shock", "ep_shock_over_tau_shock.png"),
        ("Non-Shock Cohort (y=0)", "non_shock", "ep_shock_over_tau_non_shock.png"),
    ]

    from src.app.core.plotting.base import QUALITATIVE_PALETTE

    all_colors = list(QUALITATIVE_PALETTE) + [
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
    ]
    all_markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for cohort_title, cohort_key, fname in cohort_configs:
        fig, ax = plt.subplots(figsize=(12, 7))

        for idx, (label, data) in enumerate(ep_shock_results.items()):
            if cohort_key not in data:
                continue
            tau_arr = np.array(data["tau"])
            mean_arr = np.array(data[cohort_key]["means"])
            sem_arr = np.array(data[cohort_key]["sems"])

            color = all_colors[idx % len(all_colors)]
            marker = all_markers[idx % len(all_markers)]

            ax.plot(tau_arr, mean_arr * 100.0, marker=marker, color=color, label=label, linewidth=2, markersize=6)
            ax.fill_between(
                tau_arr, (mean_arr - sem_arr) * 100.0, (mean_arr + sem_arr) * 100.0, color=color, alpha=0.12
            )

        ax.set_xlabel("Lead Time τ (hours before end-of-stay)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Average Predicted Septic Shock Probability (%)", fontsize=13, fontweight="bold")
        ax.set_title(f"EP Model Predicted Shock % at Each Lead Time\n({cohort_title})", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10, loc="best")

        fig.tight_layout()
        out_path = report_dir / fname
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  Saved EP shock-over-tau plot ({cohort_title}): {out_path}")


# ---------------------------------------------------------------------------
#  Counterfactual Table
# ---------------------------------------------------------------------------


def write_counterfactual_table(cf_data, csv_path, txt_path):
    """Write counterfactual summary table.

    cf_data: list of dicts with keys:
        method, pred_mort_mean, pred_mort_sem, admin_rate_mean, admin_rate_sem,
        agreement_mean, agreement_sem, precision_mean, precision_sem, recall_mean, recall_sem, f1_mean, f1_sem
    """
    header = [
        "method",
        "pred_mortality_mean",
        "pred_mortality_sem",
        "admin_rate_mean",
        "admin_rate_sem",
        "agreement_mean",
        "agreement_sem",
        "precision_mean",
        "precision_sem",
        "recall_mean",
        "recall_sem",
        "f1_mean",
        "f1_sem",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in cf_data:
            writer.writerow(
                [
                    row["method"],
                    f"{row['pred_mort_mean']:.6f}",
                    f"{row['pred_mort_sem']:.6f}",
                    f"{row['admin_rate_mean']:.6f}",
                    f"{row['admin_rate_sem']:.6f}",
                    f"{row['agreement_mean']:.6f}",
                    f"{row['agreement_sem']:.6f}",
                    f"{row.get('precision_mean', 0.0):.6f}",
                    f"{row.get('precision_sem', 0.0):.6f}",
                    f"{row.get('recall_mean', 0.0):.6f}",
                    f"{row.get('recall_sem', 0.0):.6f}",
                    f"{row.get('f1_mean', 0.0):.6f}",
                    f"{row.get('f1_sem', 0.0):.6f}",
                ]
            )

    with open(txt_path, "w") as f:
        f.write("=" * 110 + "\n")
        f.write("COUNTERFACTUAL EVALUATION SUMMARY TABLE\n")
        f.write("=" * 110 + "\n\n")
        f.write(
            f"{'Method':<30} {'Accuracy/Agr %':>15} {'Admin Rate %':>15} {'Precision':>12} {'Recall':>12} {'F1 Score':>12} {'Pred Mort %':>15}\n"
        )
        f.write("-" * 110 + "\n")
        for row in cf_data:
            mort_str = f"{row['pred_mort_mean'] * 100:.2f}±{row['pred_mort_sem'] * 100:.2f}"
            admin_str = f"{row['admin_rate_mean'] * 100:.2f}±{row['admin_rate_sem'] * 100:.2f}"
            agr_str = f"{row['agreement_mean'] * 100:.2f}±{row['agreement_sem'] * 100:.2f}"
            prec_str = f"{row.get('precision_mean', 0.0):.4f}"
            rec_str = f"{row.get('recall_mean', 0.0):.4f}"
            f1_str = f"{row.get('f1_mean', 0.0):.4f}"
            f.write(
                f"{pretty(row['method']):<30} {agr_str:>15} {admin_str:>15} {prec_str:>12} {rec_str:>12} {f1_str:>12} {mort_str:>15}\n"
            )
        f.write("-" * 110 + "\n")
        f.write("\nAll values: mean ± SEM across data splits.\n")
        f.write("Accuracy/Agr %: Fraction of timesteps policy matches clinician action.\n")
        f.write("Admin Rate %: Fraction of timesteps policy administers antibiotics (action=1).\n")
        f.write("Precision/Recall/F1: Treatment decision performance vs clinician actions.\n")
        f.write("Pred Mort %: Average predicted mortality on counterfactual trajectories.\n")

    print(f"  Saved counterfactual table: {csv_path} / {txt_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def compute_ep_eval_data(
    checkpoint_root,
    dataset_path,
    ep_ckpt_root="results/checkpoints/early_prediction",
    n_splits=20,
    tau_min=1,
    tau_max=33,
    tau_step=4,
    window_hours=12,
    use_volatility=True,
):

    if not checkpoint_root:
        raise ValueError("checkpoint_root must be provided.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    #  1. Load dataset
    # -----------------------------------------------------------------------
    if not dataset_path:
        dataset_path = resolve_mimic_dataset(argparse.Namespace())
    else:
        dataset_path = str(dataset_path)
    print(f"Loading dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]  # (N, 240, features)
    y = data["y"].squeeze()  # (N,) — 1=shock, 0=non-shock
    mask = data["mask"]  # (N, 240, 1) or (N, 240)

    N_patients = len(X)
    mask_2d = mask.squeeze(-1) if mask.ndim == 3 else mask
    patient_lengths = (mask_2d != -1).sum(axis=-1)
    print(f"Dataset: {N_patients} patients, shock rate: {y.mean():.3f}")

    # -----------------------------------------------------------------------
    #  2. Discover RL policy checkpoints
    # -----------------------------------------------------------------------
    policies = discover_policy_checkpoints(checkpoint_root)
    print(f"\nDiscovered {len(policies)} RL policy checkpoints:")
    for k, v in policies.items():
        print(f"  {k}: {v}")
    if not policies:
        print("WARNING: No policy checkpoints found. Only clinician baseline will be evaluated.")

    # -----------------------------------------------------------------------
    #  3. Discover EP model checkpoints
    # -----------------------------------------------------------------------
    ep_models = discover_ep_checkpoints(ep_ckpt_root)
    ep_model_keys = sorted(ep_models.keys())
    all_taus = set()
    for mk in ep_model_keys:
        for k in ep_models[mk].keys():
            if isinstance(k, int):
                all_taus.add(k)
    all_taus = sorted(all_taus)
    if not all_taus and ep_models:
        all_taus = list(range(tau_min, tau_max + 1, tau_step))
    print(f"\nDiscovered EP models: {ep_model_keys}")
    print(f"Available taus: {all_taus}")

    if not ep_models:
        print("WARNING: No EP model checkpoints found. EP shock-over-tau graph will be skipped.")

    # Compute the tau sweep list (intersect CLI range with available checkpoints)
    tau_sweep = [t for t in range(tau_min, tau_max + 1, tau_step)]
    tau_sweep_available = [t for t in tau_sweep if t in all_taus] if all_taus else []
    print(f"Tau sweep (CLI): {tau_sweep}")
    print(f"Tau sweep (available): {tau_sweep_available}")

    # -----------------------------------------------------------------------
    #  4. Pre-compute CQL V(s) for EP models that use "with V" feature
    # -----------------------------------------------------------------------
    v_vals_all = np.zeros((N_patients, 240, 1), dtype=np.float32)
    # Use the first CQL-type policy we find for V(s)
    cql_ckpt_for_v = None
    for method_key, ckpt_path in policies.items():
        if "cql" in method_key.lower() and "blendrl" not in method_key.lower():
            cql_ckpt_for_v = ckpt_path
            break
    if cql_ckpt_for_v is None and policies:
        cql_ckpt_for_v = list(policies.values())[0]

    if cql_ckpt_for_v:
        try:
            agent_v, agent_v_type = load_policy_agent(cql_ckpt_for_v, device)
            if agent_v is not None and (
                hasattr(agent_v, "get_q_values")
                or hasattr(agent_v, "get_value")
                or hasattr(agent_v, "q_network")
                or hasattr(agent_v, "model")
            ):
                print(f"\nPre-computing V(s) from: {cql_ckpt_for_v}")
                batch_sz = 128
                with torch.no_grad():
                    for i in range(0, N_patients, batch_sz):
                        batch_x = torch.as_tensor(X[i : i + batch_sz, :, :46], dtype=torch.float32, device=device)
                        B = batch_x.size(0)
                        flat_x = batch_x.view(-1, 46)
                        if hasattr(agent_v, "get_q_values"):
                            flat_q = agent_v.get_q_values(flat_x)
                        elif hasattr(agent_v, "model") and hasattr(agent_v.model, "get_q_values"):
                            flat_q = agent_v.model.get_q_values(flat_x)
                        elif hasattr(agent_v, "q_network"):
                            flat_q = agent_v.q_network(flat_x)
                        else:
                            raise AttributeError("Agent does not have get_q_values or q_network.")
                        q_vals = flat_q.view(B, 240, -1)
                        v = torch.max(q_vals, dim=-1)[0].unsqueeze(-1).cpu().numpy()
                        v_vals_all[i : i + B] = v
                print("V(s) pre-computation done.")
            else:
                print("WARNING: Could not compute V(s) — agent has no recognizable Q-network.")
        except Exception as e:
            print(f"WARNING: V(s) pre-computation failed: {e}")

    # -----------------------------------------------------------------------
    #  5. Counterfactual evaluation: per policy, per split
    # -----------------------------------------------------------------------
    w_steps = 2 * window_hours

    # For each policy + clinician, compute per-patient agreement and collect
    # counterfactual statistics across splits.
    all_method_keys = list(policies.keys()) + ["clinician"]
    cf_data = []  # for counterfactual table

    # We'll collect per-patient agreement for the first split only (for the agreement plots)
    # since agreements are deterministic given the policy — splits only differ in train/test
    patient_agreements = {}  # method_key -> np.array (N,)

    print(f"\n{'=' * 60}")
    print("Running counterfactual evaluation...")
    print(f"{'=' * 60}")

    for method_key in all_method_keys:
        is_clinician = method_key == "clinician"

        if not is_clinician:
            ckpt_path = policies[method_key]
            agent, agent_type = load_policy_agent(ckpt_path, device)
            if agent is None:
                print(f"  SKIP {method_key}: could not load checkpoint.")
                continue
            print(f"\n  Evaluating: {pretty(method_key)} ({ckpt_path})")
        else:
            agent, agent_type = None, None
            print("\n  Evaluating: Clinician (dataset actions)")

        # Compute per-patient agreement & classification stats (over ALL patients)
        per_patient_agr = np.zeros(N_patients, dtype=np.float64)
        per_patient_admin = np.zeros(N_patients, dtype=np.float64)
        per_patient_tp = np.zeros(N_patients, dtype=np.float64)
        per_patient_fp = np.zeros(N_patients, dtype=np.float64)
        per_patient_fn = np.zeros(N_patients, dtype=np.float64)
        per_patient_tn = np.zeros(N_patients, dtype=np.float64)

        for i in range(N_patients):
            valid_steps = np.where(mask[i].squeeze() != -1)[0]
            if len(valid_steps) == 0:
                continue

            if is_clinician:
                # Clinician always agrees with itself: 100%
                per_patient_agr[i] = 100.0
                clin_acts = X[i, valid_steps, 47].astype(int)
                per_patient_admin[i] = clin_acts.mean()
                per_patient_tp[i] = (clin_acts == 1).sum()
                per_patient_tn[i] = (clin_acts == 0).sum()
            else:
                matches = 0
                admin_count = 0
                tp, fp, fn, tn = 0, 0, 0, 0
                for t in valid_steps:
                    obs = torch.tensor(X[i, t, :46], dtype=torch.float32).unsqueeze(0).to(device)
                    action, _ = get_policy_actions(agent, agent_type, obs, device)
                    clin_act = int(X[i, t, 47])
                    act = action[0]
                    if act == clin_act:
                        matches += 1
                    if act == 1:
                        admin_count += 1
                    if act == 1 and clin_act == 1:
                        tp += 1
                    elif act == 1 and clin_act == 0:
                        fp += 1
                    elif act == 0 and clin_act == 1:
                        fn += 1
                    elif act == 0 and clin_act == 0:
                        tn += 1

                per_patient_agr[i] = (matches / len(valid_steps)) * 100.0
                per_patient_admin[i] = admin_count / len(valid_steps)
                per_patient_tp[i] = tp
                per_patient_fp[i] = fp
                per_patient_fn[i] = fn
                per_patient_tn[i] = tn

        patient_agreements[method_key] = per_patient_agr

        # Multi-split counterfactual mortality & classification evaluation
        split_morts = []
        split_admins = []
        split_agreements = []
        split_precisions = []
        split_recalls = []
        split_f1s = []

        for split_idx in range(n_splits):
            seed_val = 42 + split_idx
            _, test_indices = train_test_split(np.arange(N_patients), test_size=0.2, random_state=seed_val)

            agr_split = per_patient_agr[test_indices]
            admin_split = per_patient_admin[test_indices]
            mort_split = y[test_indices].mean()

            tp_sum = per_patient_tp[test_indices].sum()
            fp_sum = per_patient_fp[test_indices].sum()
            fn_sum = per_patient_fn[test_indices].sum()

            prec = tp_sum / (tp_sum + fp_sum + 1e-8)
            rec = tp_sum / (tp_sum + fn_sum + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)

            split_morts.append(float(mort_split))
            split_admins.append(float(admin_split.mean()))
            split_agreements.append(float(agr_split.mean()) / 100.0)
            split_precisions.append(float(prec))
            split_recalls.append(float(rec))
            split_f1s.append(float(f1))

        cf_data.append(
            {
                "method": method_key,
                "pred_mort_mean": float(np.mean(split_morts)),
                "pred_mort_sem": float(np.std(split_morts) / np.sqrt(len(split_morts))),
                "admin_rate_mean": float(np.mean(split_admins)),
                "admin_rate_sem": float(np.std(split_admins) / np.sqrt(len(split_admins))),
                "agreement_mean": float(np.mean(split_agreements)),
                "agreement_sem": float(np.std(split_agreements) / np.sqrt(len(split_agreements))),
                "precision_mean": float(np.mean(split_precisions)),
                "precision_sem": float(np.std(split_precisions) / np.sqrt(len(split_precisions))),
                "recall_mean": float(np.mean(split_recalls)),
                "recall_sem": float(np.std(split_recalls) / np.sqrt(len(split_recalls))),
                "f1_mean": float(np.mean(split_f1s)),
                "f1_sem": float(np.std(split_f1s) / np.sqrt(len(split_f1s))),
            }
        )

        agr_mean_pct = np.mean(split_agreements) * 100
        admin_pct = np.mean(split_admins) * 100
        print(f"    Agreement: {agr_mean_pct:.2f}% | Admin Rate: {admin_pct:.2f}%")

    # -----------------------------------------------------------------------
    #  6. EP Shock Prediction over tau (Graph 2)
    # -----------------------------------------------------------------------
    ep_shock_results = {}

    if tau_sweep_available and ep_models:
        print(f"\n{'=' * 60}")
        print("Computing EP predicted shock % over tau for each policy...")
        print(f"{'=' * 60}")

        # For each policy + clinician, for each tau, build counterfactual sequences,
        # run EP models, get average predicted shock.
        min_stay_steps = 2 * max(tau_sweep_available) + w_steps
        cohort_indices = np.array([i for i in range(N_patients) if patient_lengths[i] >= min_stay_steps])
        y_cohort = y[cohort_indices]
        print(f"  Cohort for tau sweep: {len(cohort_indices)} patients (stays >= {min_stay_steps} steps)")

        for method_key in all_method_keys:
            is_clinician = method_key == "clinician"

            if not is_clinician:
                if method_key not in policies:
                    continue
                ckpt_path = policies[method_key]
                agent, agent_type = load_policy_agent(ckpt_path, device)
                if agent is None:
                    continue
            else:
                agent, agent_type = None, None

            print(f"\n  Processing: {pretty(method_key)}")

            for ep_key in ep_model_keys:
                if ep_key not in EP_KEY_TO_CONFIG:
                    continue
                ep_cfg_name, ep_type, use_v = EP_KEY_TO_CONFIG[ep_key]
                line_label = f"{pretty(method_key)} × {ep_cfg_name}"

                tau_vals = []
                tau_all_means, tau_all_sems = [], []
                tau_shock_means, tau_shock_sems = [], []
                tau_non_shock_means, tau_non_shock_sems = [], []

                for tau in tau_sweep_available:
                    if tau in ep_models[ep_key]:
                        ep_ckpt_list = ep_models[ep_key][tau]
                    elif "single_model" in ep_models[ep_key]:
                        ep_ckpt_list = ep_models[ep_key]["single_model"]
                    else:
                        continue
                    steps_early = 2 * tau

                    # Build sequences with counterfactual actions
                    t_cutoffs = patient_lengths[cohort_indices] - steps_early
                    seq_data = []

                    for ci, orig_idx in enumerate(cohort_indices):
                        tc = int(t_cutoffs[ci])
                        st = max(0, tc - w_steps)

                        # Counterfactual: replace action column with policy's actions
                        raw_seq = X[orig_idx, st:tc, :49].copy()

                        if not is_clinician:
                            # Replace action at col 47 with policy's recommended action
                            for t_rel in range(raw_seq.shape[0]):
                                abs_t = st + t_rel
                                obs = torch.tensor(X[orig_idx, abs_t, :46], dtype=torch.float32).unsqueeze(0).to(device)
                                act, _ = get_policy_actions(agent, agent_type, obs, device)
                                raw_seq[t_rel, 47] = act[0]
                                # Also update action col 48 if it exists (some datasets)
                                if raw_seq.shape[-1] > 48:
                                    raw_seq[t_rel, 48] = act[0]

                        if use_volatility:
                            feat_seq = compute_volatility_features(raw_seq)
                        else:
                            feat_seq = raw_seq

                        if use_v and v_vals_all is not None:
                            v_seq = v_vals_all[orig_idx, st:tc]
                            seq_data.append(np.concatenate([feat_seq, v_seq], axis=-1))
                        else:
                            seq_data.append(feat_seq)

                    # Run EP models
                    probs = predict_shock_probs_with_ep_models(ep_ckpt_list, seq_data, device)

                    tau_vals.append(tau)

                    # All patients
                    tau_all_means.append(float(np.mean(probs)))
                    tau_all_sems.append(float(np.std(probs) / np.sqrt(len(probs))))

                    # Shock cohort (y=1)
                    s_mask = y_cohort == 1
                    if np.sum(s_mask) > 0:
                        p_s = probs[s_mask]
                        tau_shock_means.append(float(np.mean(p_s)))
                        tau_shock_sems.append(float(np.std(p_s) / np.sqrt(len(p_s))))
                    else:
                        tau_shock_means.append(0.0)
                        tau_shock_sems.append(0.0)

                    # Non-shock cohort (y=0)
                    ns_mask = y_cohort == 0
                    if np.sum(ns_mask) > 0:
                        p_ns = probs[ns_mask]
                        tau_non_shock_means.append(float(np.mean(p_ns)))
                        tau_non_shock_sems.append(float(np.std(p_ns) / np.sqrt(len(p_ns))))
                    else:
                        tau_non_shock_means.append(0.0)
                        tau_non_shock_sems.append(0.0)

                if tau_vals:
                    ep_shock_results[line_label] = {
                        "tau": tau_vals,
                        "all": {"means": tau_all_means, "sems": tau_all_sems},
                        "shock": {"means": tau_shock_means, "sems": tau_shock_sems},
                        "non_shock": {"means": tau_non_shock_means, "sems": tau_non_shock_sems},
                    }
                    print(f"    {line_label}: taus={tau_vals}")

    # -----------------------------------------------------------------------
    #  7. Generate outputs
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Generating plots and tables...")
    print("\n=== Evaluation Computation Complete ===")
    return {
        "rl_agreements": {k: v for k, v in patient_agreements.items() if k != "clinician"},
        "y": y,
        "ep_shock_results": ep_shock_results,
        "cf_data": cf_data,
    }

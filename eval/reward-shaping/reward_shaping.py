"""
EP-Based Potential Reward Shaping for Offline RL.

Uses trained Early Prediction (EP) septic shock classifiers to compute
potential-based reward shaping for CQL/IQL offline RL training.

Theory: Φ(s) = -P_EP(shock | observation window ending at s)
  r_shaped = r_original + λ * (γ * Φ(s_{t+1}) - Φ(s_t))

This is potential-based reward shaping (Ng, Harada, Russell 1999),
which provably preserves the set of optimal policies.

References:
  - Ng, Harada, Russell (1999). "Policy invariance under reward transformations."
  - Wiewiora (2003). "Potential-based shaping and Q-value initialization."
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


from src.app.api.evaluation import register_evaluator
from src.app.api.workspace import WorkspacePaths


def _results():
    return WorkspacePaths.for_process().results


def _workspace_path(value):
    """A configured path; relative paths resolve from the workspace, not the working directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else WorkspacePaths.for_process().root / path


@register_evaluator("reward_shaping", "EPRewardShaper")
class EPRewardShaper:
    """Compute potential-based reward shaping using EP model predictions.

    Given a trained EP model checkpoint, computes P(shock|s) for each state
    in the offline dataset and uses it as a potential function for reward shaping.

    Args:
        ep_ckpt_dir: Directory containing EP model checkpoints (.pt files); relative paths
                     resolve from the workspace. Default: the workspace's
                     results/checkpoints/early_prediction.
        cql_ckpt_dir: CQL checkpoint (file or directory with best_model*.ckpt) for V(s),
                      required by V-augmented EP models.
        device: Torch device
        lambda_coef: Shaping coefficient (default: 1.0)
        gamma: Discount factor (default: 0.99)
        window_hours: Observation window in hours (default: 12)
        use_volatility: Whether to compute volatility features (default: True)
        ep_architecture: Which EP architecture to use ('lstm_with_v', 'transformer_with_v', etc.)
                        If None, uses all available architectures and averages.
    """

    def __init__(
        self,
        ep_ckpt_dir: str | None = None,
        cql_ckpt_dir: str | None = None,
        device: torch.device | None = None,
        lambda_coef: float = 1.0,
        gamma: float = 0.99,
        window_hours: int = 12,
        use_volatility: bool = True,
        ep_architecture: str | None = None,
        **kwargs,
    ):
        self.ep_ckpt_dir = _workspace_path(ep_ckpt_dir) if ep_ckpt_dir else _results() / "checkpoints" / "early_prediction"
        self.cql_ckpt_dir = _workspace_path(cql_ckpt_dir) if cql_ckpt_dir else None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_coef = lambda_coef
        self.gamma = gamma
        self.window_steps = 2 * window_hours  # 30-min intervals
        self.use_volatility = use_volatility
        self.ep_architecture = ep_architecture

        self._models = []  # List of (model, model_type, input_dim) tuples
        self._load_ep_models()

    def _load_ep_models(self):
        """Load EP model checkpoints from the checkpoint directory."""
        from src.app.api.packages import provider_export
        SepsisLSTM = provider_export('mimic-early-prediction', 'SepsisLSTM')
        SepsisTransformer = provider_export('mimic-early-prediction', 'SepsisTransformer')

        if not self.ep_ckpt_dir.exists():
            print(f"WARNING: EP checkpoint directory {self.ep_ckpt_dir} does not exist.")
            return

        # Find all .pt and .ckpt files (shallow search up to 2 levels to avoid NFS storms)
        ckpt_files = sorted(
            list(self.ep_ckpt_dir.glob("*.pt"))
            + list(self.ep_ckpt_dir.glob("*/*.pt"))
            + list(self.ep_ckpt_dir.glob("*/*/*.pt"))
            + list(self.ep_ckpt_dir.glob("*.ckpt"))
            + list(self.ep_ckpt_dir.glob("*/*.ckpt"))
            + list(self.ep_ckpt_dir.glob("*/*/*.ckpt"))
        )
        seen = set()
        deduped = []
        for f in ckpt_files:
            if f not in seen:
                seen.add(f)
                deduped.append(f)
        ckpt_files = deduped

        if self.ep_architecture:
            ckpt_files = [f for f in ckpt_files if self.ep_architecture in f.stem]

        if not ckpt_files:
            print(f"WARNING: No EP checkpoints found in {self.ep_ckpt_dir}")
            return

        for ckpt_file in ckpt_files:
            try:
                data = torch.load(ckpt_file, map_location=self.device, weights_only=False)
                hparams = data.get("hyper_parameters", {}) if isinstance(data, dict) else {}
                arch_name = hparams.get("architecture_name", "")

                m_type = data.get("model_type")
                if not m_type:
                    if arch_name.startswith("transformer") or "transformer" in ckpt_file.stem.lower():
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
                    ).to(self.device)
                elif m_type == "transformer":
                    d_model = params.get("d_model", 64)
                    model = SepsisTransformer(
                        input_dim=input_dim,
                        d_model=d_model,
                        nhead=params.get("nhead", 4),
                        num_layers=params.get("num_layers", 2),
                        dim_feedforward=params.get("dim_feedforward", d_model * 2),
                        dropout=params.get("dropout", 0.1),
                        norm_first=params.get("norm_first", True),
                        pos_type=params.get("pos_type", "learned"),
                        use_cls_token=params.get("use_cls_token", True),
                        use_tcn_conv=params.get("use_tcn_conv", False),
                    ).to(self.device)
                else:
                    continue

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
                self._models.append((model, m_type, input_dim))
            except Exception as e:
                print(f"WARNING: Could not load EP checkpoint {ckpt_file}: {e}")

        print(f"EPRewardShaper: Loaded {len(self._models)} EP model(s) from {self.ep_ckpt_dir}")

    def _predict_shock_prob(self, sequences: list[np.ndarray]) -> np.ndarray:
        """Run all loaded EP models on sequences and return averaged P(shock).

        Args:
            sequences: List of numpy arrays, each (seq_len, features)

        Returns:
            probs: numpy array (N,) of averaged P(shock) predictions
        """
        from src.app.api.packages import provider_export
        evaluate_lstm_model = provider_export('mimic-early-prediction', 'evaluate_lstm_model')
        evaluate_transformer_model = provider_export('mimic-early-prediction', 'evaluate_transformer_model')

        if not self._models or not sequences:
            return np.zeros(len(sequences), dtype=np.float32)

        # Z-score normalize (matching training pipeline)
        all_steps = np.concatenate(sequences, axis=0)
        mean = np.mean(all_steps, axis=0, keepdims=True)
        std = np.std(all_steps, axis=0, keepdims=True) + 1e-6
        sequences_norm = [(s - mean) / std for s in sequences]

        all_probs = []
        for model, m_type, input_dim in self._models:
            if m_type == "lstm":
                probs = evaluate_lstm_model(model, sequences_norm, input_dim, device=str(self.device))
            else:
                probs = evaluate_transformer_model(model, sequences_norm, input_dim, device=str(self.device))
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    def compute_trajectory_potentials(
        self,
        trajectory_obs: np.ndarray,
        v_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute potential Φ(s) = -P(shock|s) for each timestep in a trajectory.

        For each timestep t, uses the observation window [max(0, t-w):t] as input
        to the EP model to get P(shock) at that point.

        Args:
            trajectory_obs: (T, 49) array of raw observations (46 features + actions)
            v_values: Optional (T, 1) array of V(s) values from CQL

        Returns:
            potentials: (T,) array of Φ(s_t) = -P(shock|s_t)
        """
        from src.app.api.packages import provider_export
        compute_volatility_features = provider_export('mimic-early-prediction', 'compute_volatility_features')

        T = trajectory_obs.shape[0]
        if T == 0:
            return np.array([], dtype=np.float32)

        # Build windowed sequences for each timestep
        sequences = []
        for t in range(1, T + 1):  # t is the cutoff (exclusive end)
            st = max(0, t - self.window_steps)
            raw_window = trajectory_obs[st:t, :49]

            if self.use_volatility:
                feat_seq = compute_volatility_features(raw_window)
            else:
                feat_seq = raw_window

            # Append V(s) if available and models expect it
            if v_values is not None and self._models_use_v():
                v_window = v_values[st:t]
                feat_seq = np.concatenate([feat_seq, v_window], axis=-1)

            sequences.append(feat_seq)

        # Batch predict
        probs = self._predict_shock_prob(sequences)

        # Φ(s) = -P(shock|s) — lower shock risk = higher potential
        return -probs

    def _models_use_v(self) -> bool:
        """Check if loaded EP models expect V(s) features (input_dim > 196)."""
        if not self._models:
            return False
        # Check if any model has input_dim > 196 (196 = 49*4 volatility features)
        return any(input_dim > 196 for _, _, input_dim in self._models)

    def shape_rewards(self, reader, cfg=None):
        """Apply EP-based potential reward shaping to an offline dataset reader.

        Modifies reader.rewards in-place with:
          r_new = r_old + λ * (γ * Φ(s') - Φ(s))

        Args:
            reader: DatasetReader with .obs, .next_obs, .rewards, .dones attributes
            cfg: Unused; the shaper's own parameters configure it.
        """
        import torch

        if not self._models:
            print("WARNING: No EP models loaded. Skipping reward shaping.")
            return

        if reader.obs.shape[-1] != 46:
            print(f"WARNING: Expected 46-dim observations for MIMIC, got {reader.obs.shape[-1]}. Skipping.")
            return

        gamma = self.gamma

        n_transitions = len(reader.obs)
        print(f"EPRewardShaper: Computing potentials for {n_transitions} transitions...")

        # Reconstruct trajectories from flat reader data
        trajectories = []  # List of (start_idx, end_idx) tuples
        start_idx = 0
        for idx in range(n_transitions):
            if reader.dones[idx] == 1.0 or idx == n_transitions - 1:
                trajectories.append((start_idx, idx + 1))
                start_idx = idx + 1

        print(f"  Found {len(trajectories)} patient trajectories")

        # Pre-compute V(s) if models expect it and a CQL checkpoint is available
        v_all = None
        if self._models_use_v():
            v_all = self._precompute_v_values(reader)

        # Compute potentials for each trajectory
        all_potentials = np.zeros(n_transitions, dtype=np.float32)

        for traj_idx, (t_start, t_end) in enumerate(trajectories):
            traj_len = t_end - t_start

            # Build the 49-dim observation array for this trajectory
            # obs is 46-dim; we need to reconstruct the 49-dim with actions
            obs_46 = reader.obs[t_start:t_end].numpy()
            actions = reader.actions[t_start:t_end].numpy().reshape(-1, 1)

            # Pad to 49 dims: [obs(46), action, action, 0]
            traj_obs_49 = np.zeros((traj_len, 49), dtype=np.float32)
            traj_obs_49[:, :46] = obs_46
            traj_obs_49[:, 47] = actions.squeeze()
            traj_obs_49[:, 48] = actions.squeeze()

            v_traj = None
            if v_all is not None:
                v_traj = v_all[t_start:t_end].reshape(-1, 1)

            potentials = self.compute_trajectory_potentials(traj_obs_49, v_traj)
            all_potentials[t_start:t_end] = potentials

            if (traj_idx + 1) % 500 == 0:
                print(f"  Processed {traj_idx + 1}/{len(trajectories)} trajectories")

        # Apply potential-based shaping: r_new = r_old + λ * (γ * Φ(s') - Φ(s))
        shaped_bonus = np.zeros(n_transitions, dtype=np.float32)
        for t_start, t_end in trajectories:
            for t in range(t_start, t_end):
                phi_s = all_potentials[t]
                if t + 1 < t_end:
                    phi_s_next = all_potentials[t + 1]
                else:
                    phi_s_next = 0.0  # Terminal state potential is 0

                shaped_bonus[t] = self.lambda_coef * (gamma * phi_s_next - phi_s)

        # Apply shaping
        original_mean = reader.rewards.mean().item()
        reader.rewards = reader.rewards + torch.tensor(shaped_bonus, dtype=reader.rewards.dtype)
        shaped_mean = reader.rewards.mean().item()

        print(f"  Reward shaping applied: mean reward {original_mean:.4f} -> {shaped_mean:.4f}")
        print(
            f"  Shaping bonus stats: mean={shaped_bonus.mean():.4f}, std={shaped_bonus.std():.4f}, "
            f"min={shaped_bonus.min():.4f}, max={shaped_bonus.max():.4f}"
        )

    def _precompute_v_values(self, reader) -> np.ndarray | None:
        """Pre-compute V(s) = max_a Q(s,a) from the cql_ckpt_dir checkpoint for V-augmented EP models."""
        if self.cql_ckpt_dir is None:
            raise ValueError("EP models require V(s): set the evaluator's cql_ckpt_dir.")

        ckpt_path = self.cql_ckpt_dir
        cql_ckpt_path = None

        if ckpt_path.is_file() and ckpt_path.suffix == ".ckpt":
            cql_ckpt_path = ckpt_path
        elif ckpt_path.is_dir():
            # Strict search for best_model.ckpt in the provided directory
            candidates = list(ckpt_path.glob("best_model*.ckpt"))
            if candidates:
                candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
                cql_ckpt_path = candidates[0]

        if not cql_ckpt_path or not cql_ckpt_path.exists():
            raise FileNotFoundError(f"Could not find a valid CQL checkpoint for V(s) computation at: {ckpt_path}")

        from src.app.api.agents import get_agent_class
        agent_cls = get_agent_class("cql")

        torch.serialization.add_safe_globals(
            [getattr(sys.modules.get("omegaconf.dictconfig", None), "DictConfig", None)]
        )
        agent = agent_cls.load_from_checkpoint(str(cql_ckpt_path), map_location=self.device, weights_only=False)
        agent.eval()

        n = len(reader.obs)
        v_vals = np.zeros((n, 1), dtype=np.float32)
        batch_sz = 512

        with torch.no_grad():
            for i in range(0, n, batch_sz):
                batch_obs = reader.obs[i : i + batch_sz].to(self.device)
                if hasattr(agent, "get_value"):
                    v = agent.get_value(batch_obs).unsqueeze(-1).cpu().numpy()
                elif hasattr(agent, "get_q_values"):
                    q = agent.get_q_values(batch_obs)
                    v = torch.max(q, dim=-1)[0].unsqueeze(-1).cpu().numpy()
                elif hasattr(agent, "model") and hasattr(agent.model, "get_q_values"):
                    q = agent.model.get_q_values(batch_obs)
                    v = torch.max(q, dim=-1)[0].unsqueeze(-1).cpu().numpy()
                elif hasattr(agent, "q_network"):
                    q = agent.q_network(batch_obs)
                    v = torch.max(q, dim=-1)[0].unsqueeze(-1).cpu().numpy()
                else:
                    raise AttributeError("CQL agent does not implement get_value or recognized Q-network.")
                v_vals[i : i + batch_obs.size(0)] = v

        print(f"  V(s) computed: mean={v_vals.mean():.4f}, std={v_vals.std():.4f}")
        return v_vals

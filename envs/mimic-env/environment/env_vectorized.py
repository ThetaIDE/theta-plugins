import os
from pathlib import Path

import numpy as np
import torch as th

from src.app.core.env_vectorized import VectorizedBaseEnv


def compute_tqn_stage_severity(obs):
    """
    Computes septic stage severity penalty for a clinical state observation.
    Penalties:
      - Infection: -1.0
      - Inflammation (SIRS): -2.0
      - Single Organ Failure L1: -5.0, L2: -10.0
      - Multiple Organ Failure L1: -20.0, L2: -30.0
      - Septic Shock: -50.0
    """
    hr = obs[0]
    rr = obs[1]
    spo2 = obs[2]
    fio2 = obs[4]
    temp = obs[7]
    bands = obs[8]
    bun = obs[9]
    lactate = obs[10]
    platelets = obs[11]
    creatinine = obs[12]
    bilirubin = obs[13]
    wbc = obs[14]
    map_bp = obs[15]
    crp = obs[16]

    severity = 0.0

    # 1. Infection (-1.0)
    is_infection = (bands > 10.0) or (wbc > 12.0) or (0 < wbc < 4.0) or (crp > 10.0)
    if is_infection:
        severity += -1.0

    # 2. Inflammation (-2.0)
    is_sirs = ((temp > 38.0) or (0 < temp < 36.0)) or (hr > 90.0) or (rr > 20.0) or (wbc > 12.0) or (0 < wbc < 4.0)
    if is_sirs:
        severity += -2.0

    # 3. Organ Failures (OF)
    l1_count, l2_count = 0, 0

    # Cardio
    if 0 < map_bp < 55: l2_count += 1
    elif 0 < map_bp < 65: l1_count += 1

    # Meta
    if lactate > 4.0: l2_count += 1
    elif lactate > 2.0: l1_count += 1

    # Hema
    if 0 < platelets < 100: l2_count += 1
    elif 0 < platelets < 150: l1_count += 1

    # Renal
    if creatinine > 2.0 or bun > 40: l2_count += 1
    elif creatinine > 1.2 or bun > 20: l1_count += 1

    # Resp
    if (0 < spo2 < 88) or fio2 > 0.6: l2_count += 1
    elif (0 < spo2 < 92) or fio2 > 0.4 or rr > 24: l1_count += 1

    # Gastro
    if bilirubin > 2.0: l2_count += 1
    elif bilirubin > 1.2: l1_count += 1

    total_of = l1_count + l2_count
    if l2_count >= 2 or (total_of >= 2 and l2_count >= 1):
        severity += -30.0  # Multiple OF Level-2
    elif total_of >= 2:
        severity += -20.0  # Multiple OF Level-1
    elif l2_count == 1:
        severity += -10.0  # Single OF Level-2
    elif l1_count == 1:
        severity += -5.0   # Single OF Level-1

    # 4. Septic Shock (-50.0)
    is_shock = (0 < map_bp < 65) and (lactate > 2.0)
    if is_shock:
        severity += -50.0

    return severity

def compute_tqn_action_cost(policy_action, obs=None):
    """
    Action costs to prevent over-treatment:
      - Oxygen control: -0.01
      - Anti-infection drug: -0.1
      - Vasopressor: -0.2
    """
    cost = 0.0
    if isinstance(policy_action, (int, np.integer)):
        if policy_action == 1:
            cost += 0.1  # Anti-infection drug
    elif isinstance(policy_action, (list, tuple, np.ndarray)):
        if len(policy_action) >= 1 and policy_action[0] == 1: cost += 0.01
        if len(policy_action) >= 2 and policy_action[1] == 1: cost += 0.1
        if len(policy_action) >= 3 and policy_action[2] == 1: cost += 0.2
    return cost

class VectorizedNudgeEnv(VectorizedBaseEnv):
    name = "mimic"
    pred2action = {
        "withhold": 0,
        "administer": 1,
    }
    
    def __init__(
        self,
        mode: str,
        n_envs: int,
        seed=None,
        dataset_name=None,
        **kwargs
    ):
        super().__init__()
        self.mode = mode
        if dataset_name is None:
            dataset_name = os.environ.get("MIMIC_DATASET_NAME", "mimic_lazy_0_interventions_balanced.npz")
        self.n_envs = n_envs
        self.seed = seed if seed is not None else 42
        
        # Load dataset
        dataset_path = Path(dataset_name)
        if not dataset_path.is_file():
            filename = dataset_path.name
            search_dirs = [
                Path(os.environ.get("MIMIC_DATASET_DIR", "")),
                Path("in/datasets/mimic"),
                Path("in/datasets"),
            ]
            dataset_path = next(
                (directory / filename for directory in search_dirs
                 if str(directory) not in ("", ".") and (directory / filename).is_file()),
                dataset_path,
            )
        path = str(dataset_path)
                    
        if not os.path.exists(path):
            raise FileNotFoundError(f"MIMIC dataset not found at {path}")
            
        data = np.load(path, allow_pickle=True)
        self.X = data['X']  # (N, 240, 49)
        self.y = data['y']  # (N, 1)
        self.mask = data['mask']  # (N, 240, 1)
        self.orig = data['orig'] if 'orig' in data else None
        
        self.n_patients = len(self.X)
        self.states = self.X[:, :, :46]
        self.actions_antibiotics = self.X[:, :, 47]  # Column 47 is AntiInfectiveAdmin_max
        
        self.n_actions = 2
        self.n_raw_actions = 2
        self.n_features = 46
        
        # Keep track of active trajectories and steps for each vectorized env
        self.rng = np.random.default_rng(self.seed)
        self.current_traj_idx = np.zeros(n_envs, dtype=np.int32)
        self.current_step_idx = np.zeros(n_envs, dtype=np.int32)
        
        # Store valid steps for each patient stay
        self.valid_steps_by_patient = {}
        for i in range(self.n_patients):
            valid = np.where(self.mask[i].squeeze() != -1)[0]
            self.valid_steps_by_patient[i] = valid
            
        # Initial trajectory selection
        for i in range(n_envs):
            self._reset_env_slot(i)

    def _reset_env_slot(self, env_idx: int):
        """Randomly select a patient trajectory and set step to 0"""
        while True:
            idx = self.rng.integers(0, self.n_patients)
            if len(self.valid_steps_by_patient[idx]) > 0:
                break
        self.current_traj_idx[env_idx] = idx
        self.current_step_idx[env_idx] = 0

    def reset(self, seed=None):
        obs_list = []
        for i in range(self.n_envs):
            self._reset_env_slot(i)
            traj = self.current_traj_idx[i]
            step_t = self.valid_steps_by_patient[traj][0]
            obs = self.states[traj, step_t]
            obs_list.append(th.tensor(obs, dtype=th.float32))
        return th.stack(obs_list)

    def step(self, actions, is_mapped: bool = False):
        rewards = []
        terminations = []
        truncations = []
        infos = []
        obs_list = []

        for i in range(self.n_envs):
            traj = self.current_traj_idx[i]
            step_idx = self.current_step_idx[i]
            valid_steps = self.valid_steps_by_patient[traj]

            t = valid_steps[step_idx]
            obs = self.states[traj, t]

            # Action taken by policy
            policy_action = actions[i]
            if hasattr(policy_action, "item"):
                policy_action = policy_action.item()

            # Historical action taken by clinician
            clinician_action = int(self.actions_antibiotics[traj, t])

            is_done = (step_idx == len(valid_steps) - 1)

            reward_type = os.environ.get("REWARD_TYPE", "tqn").lower()
            if reward_type == "outcome":
                outcome = self.y[traj, 0]
                terminal_reward = 15.0 if outcome == 0 else -15.0
                reward = 0.0
                if is_done:
                    reward = terminal_reward
            elif reward_type == "tqn":
                curr_sev = compute_tqn_stage_severity(obs)
                if step_idx > 0:
                    prev_t = valid_steps[step_idx - 1]
                    prev_obs = self.states[traj, prev_t]
                    prev_sev = compute_tqn_stage_severity(prev_obs)
                else:
                    prev_sev = 0.0

                stage_reward = curr_sev - prev_sev
                act_cost = compute_tqn_action_cost(policy_action, obs)
                reward = stage_reward - act_cost
            else:
                # Default behavioral copying reward
                T = len(valid_steps)
                outcome = self.y[traj, 0]
                if outcome == 0:  # Patient survived
                    reward = 1.0 / T if policy_action == clinician_action else 0.0
                else:  # Patient died
                    reward = 0.0

            if is_done:
                terminated = True
                self._reset_env_slot(i)
                new_traj = self.current_traj_idx[i]
                new_t = self.valid_steps_by_patient[new_traj][0]
                next_obs = self.states[new_traj, new_t]
            else:
                terminated = False
                self.current_step_idx[i] += 1
                next_t = valid_steps[self.current_step_idx[i]]
                next_obs = self.states[traj, next_t]

            obs_list.append(th.tensor(next_obs, dtype=th.float32))
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(False)
            infos.append({})

        return (
            th.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terminations, dtype=bool),
            np.array(truncations, dtype=bool),
            infos,
        )

    def get_action_meanings(self):
        return ["withhold", "administer"]

    def close(self):
        pass


VectorizedEnv = VectorizedNudgeEnv


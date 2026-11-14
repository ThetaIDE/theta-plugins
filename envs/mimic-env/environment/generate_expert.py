import numpy as np
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Dynamic dataset directory detection
from src.app.api.workspace import WorkspacePaths
PROJECT_ROOT = WorkspacePaths.for_process().root
dataset_name = "mimic_lazy_0_interventions_balanced.npz"
search_dirs = [
    Path(os.environ.get("MIMIC_DATASET_DIR", "")),
    PROJECT_ROOT / "in" / "datasets" / "mimic",
    PROJECT_ROOT / "in" / "datasets",
]
src_path = next(
    (str(directory / dataset_name) for directory in search_dirs
     if str(directory) not in ("", ".") and (directory / dataset_name).is_file()),
    "",
)
out_dir = os.path.dirname(src_path)
out_path = os.path.join(out_dir, "mimic_expert_demonstrations.npz")

if not os.path.exists(src_path):
    print(f"Error: Source dataset not found at {src_path}")
    sys.exit(1)

print(f"Loading full dataset from {src_path}...")
data = np.load(src_path, allow_pickle=True)
X = data['X']  # (N, 240, 49)
y = data['y']  # (N, 1)
mask = data['mask']  # (N, 240, 1)
vids = data['vids']
orig = data['orig'] if 'orig' in data else None

# Step 1: Detect pre-intervention segments
print("Detecting pre-intervention segments...")
X_pre = []
y_pre = []
masks_pre = []
valid_indices = []

for i in range(len(X)):
    # Actions are columns 47 (antibiotics) and 48 (vasopressors)
    actions = X[i, :, 47] + X[i, :, 48]
    active_mask = (mask[i].squeeze() != -1)
    active_steps = np.where(active_mask)[0]
    
    if len(active_steps) == 0:
        continue
        
    # Find first step where an intervention was given
    intervention_steps = np.where((actions > 0) & active_mask)[0]
    if len(intervention_steps) > 0:
        T_first = intervention_steps[0]
    else:
        T_first = active_steps[-1] + 1
        
    # We need some history before the first intervention to predict
    # Let's say at least 4 steps (2 hours)
    pre_active_steps = [t for t in active_steps if t < T_first]
    if len(pre_active_steps) >= 4:
        # Create a pre-intervention sequence
        p_X = X[i].copy()
        p_mask = np.full((240, 1), -1.0)
        p_mask[pre_active_steps] = 1.0
        
        # Zero out features after T_first
        for t in range(240):
            if t >= T_first:
                p_X[t] = 0.0
                
        X_pre.append(p_X[:, :46]) # Only clinical features
        y_pre.append(y[i])
        masks_pre.append(p_mask)
        valid_indices.append(i)

X_pre = np.array(X_pre, dtype=np.float32)
y_pre = np.array(y_pre, dtype=np.float32)
masks_pre = np.array(masks_pre, dtype=np.float32)

print(f"Prepared {len(X_pre)} pre-intervention sequences for training.")

# PyTorch Dataset for LSTM
class SequenceDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]

class SepsisClassifierLSTM(nn.Module):
    def __init__(self, input_dim=46, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, mask):
        out, _ = self.lstm(x)
        # Get output at the last valid timestep of the mask
        valid_mask = (mask.squeeze(-1) != -1).float()
        lengths = valid_mask.sum(dim=1).long()
        batch_size = x.size(0)
        idx = (lengths - 1).clamp(min=0)
        last_out = out[torch.arange(batch_size), idx]
        return self.fc(last_out)

# Step 2: Train Ensemble of LSTMs
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Training ensemble on device: {device}")

ensemble_size = 5
models = []

# Split train/val
train_idx, val_idx = train_test_split(np.arange(len(X_pre)), test_size=0.2, random_state=42)

for m_idx in range(ensemble_size):
    print(f"Training ensemble model {m_idx+1}/{ensemble_size}...")
    
    # Bootstrap sample the training indices
    boot_idx = np.random.choice(train_idx, size=len(train_idx), replace=True)
    
    train_ds = SequenceDataset(X_pre[boot_idx], y_pre[boot_idx], masks_pre[boot_idx])
    val_ds = SequenceDataset(X_pre[val_idx], y_pre[val_idx], masks_pre[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = SepsisClassifierLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(40):
        model.train()
        for bx, by, bm in train_loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            optimizer.zero_grad()
            logits = model(bx, bm)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by, bm in val_loader:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                logits = model(bx, bm)
                preds = (logits > 0).float()
                correct += (preds == by).sum().item()
                total += by.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    print(f"  Model {m_idx+1} trained. Best val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    models.append(model)

# Step 3: Run Ensemble on all survivors to identify expert trajectories
print("\nRunning ensemble predictions to filter expert trajectories...")
expert_indices = []

# Prepare all sequences in the format expected by the predictor
X_all_pre = []
masks_all_pre = []
for i in range(len(X)):
    actions = X[i, :, 47] + X[i, :, 48]
    active_mask = (mask[i].squeeze() != -1)
    active_steps = np.where(active_mask)[0]
    
    if len(active_steps) == 0:
        T_first = 0
    else:
        intervention_steps = np.where((actions > 0) & active_mask)[0]
        T_first = intervention_steps[0] if len(intervention_steps) > 0 else active_steps[-1] + 1
        
    p_X = X[i].copy()
    p_mask = np.full((240, 1), -1.0)
    pre_active = [t for t in active_steps if t < T_first]
    p_mask[pre_active] = 1.0
    
    for t in range(240):
        if t >= T_first:
            p_X[t] = 0.0
            
    X_all_pre.append(p_X[:, :46])
    masks_all_pre.append(p_mask)

X_all_pre = torch.tensor(np.array(X_all_pre, dtype=np.float32)).to(device)
masks_all_pre = torch.tensor(np.array(masks_all_pre, dtype=np.float32)).to(device)

all_probs = []
with torch.no_grad():
    for model in models:
        logits = model(X_all_pre, masks_all_pre)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        all_probs.append(probs)
        
mean_probs = np.mean(all_probs, axis=0)

# Filter criteria: y == 0 (survivor) and mean_probs > 0.5 (predicted to crash)
for i in range(len(X)):
    if y[i, 0] == 0 and mean_probs[i] > 0.5:
        expert_indices.append(i)

print(f"\nFiltering results:")
print(f"  Total survivors: {len(np.where(y == 0)[0])}")
print(f"  Expert trajectories identified (survived but predicted to crash): {len(expert_indices)}")

if len(expert_indices) == 0:
    print("Warning: No expert trajectories found. Try lowering the threshold to 0.4.")
    expert_indices = [i for i in range(len(X)) if y[i, 0] == 0 and mean_probs[i] > 0.4]
    print(f"  Threshold 0.4 filter count: {len(expert_indices)}")

# Step 4: Save the expert trajectories
# For early prediction, we also slice off the last 24 steps (12 hours) of these expert stays
# to make sure the evaluation is a true early prediction task!
print(f"Saving expert trajectories (slicing off last 12 hours from stays)...")

exp_X = []
exp_y = []
exp_mask = []
exp_vids = []
exp_orig = []

for idx in expert_indices:
    mask_patient = mask[idx].squeeze()
    active_steps = np.where(mask_patient != -1)[0]
    
    # Slice off last 12 hours (24 steps)
    if len(active_steps) > 24:
        keep_indices = active_steps[:-24]
        
        p_mask = np.full((240, 1), -1.0)
        p_mask[keep_indices] = 1.0
        
        p_X = X[idx].copy()
        for t in range(240):
            if t not in keep_indices:
                p_X[t] = 0.0
                
        exp_X.append(p_X)
        exp_y.append(y[idx])
        exp_mask.append(p_mask)
        exp_vids.append(vids[idx])
        
        if orig is not None:
            p_orig = orig[idx].copy()
            for t in range(240):
                if t not in keep_indices:
                    p_orig[t] = 0.0
            exp_orig.append(p_orig)

exp_X = np.array(exp_X, dtype=X.dtype)
exp_y = np.array(exp_y, dtype=y.dtype)
exp_mask = np.array(exp_mask, dtype=mask.dtype)
exp_vids = np.array(exp_vids, dtype=vids.dtype)

save_dict = {
    'X': exp_X,
    'y': exp_y,
    'mask': exp_mask,
    'vids': exp_vids
}

if orig is not None:
    exp_orig = np.array(exp_orig, dtype=orig.dtype)
    save_dict['orig'] = exp_orig

for key in ['feature_names_x', 'feature_names_orig']:
    if key in data:
        save_dict[key] = data[key]

np.savez_compressed(out_path, **save_dict)
print(f"Expert demonstrations dataset saved successfully to {out_path} ({len(exp_X)} patients)")

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# --- Positional Encoding for Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=240):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        sin_vals = torch.sin(position * div_term)
        cos_vals = torch.cos(position * div_term)

        pe[:, 0::2] = sin_vals[:, : pe[:, 0::2].shape[1]]
        pe[:, 1::2] = cos_vals[:, : pe[:, 1::2].shape[1]]

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# --- Soft Temporal Attention Pooling ---
class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 2)), nn.Tanh(), nn.Linear(max(16, d_model // 2), 1)
        )

    def forward(self, x, padding_mask=None):
        # x: (B, L, H), padding_mask: (B, L) where True indicates padding
        scores = self.attn(x).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        context = (x * weights).sum(dim=1)
        return context


# --- Focal Loss for Class Imbalance ---
class FocalLoss(nn.Module):
    def __init__(self, pos_weight=1.0, gamma=2.0):
        super().__init__()
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        pos_w = self.pos_weight.to(targets.device)
        weight_factor = targets * pos_w + (1 - targets)
        loss = focal_factor * weight_factor * bce_loss
        return loss.mean()


def compute_volatility_features(seq):
    # seq: (L, D)
    L, D = seq.shape
    delta = np.zeros_like(seq, dtype=np.float32)
    if L > 1:
        delta[1:] = seq[1:] - seq[:-1]

    seq_prev = np.zeros_like(seq, dtype=np.float32)
    seq_prev[0] = seq[0]
    if L > 1:
        seq_prev[1:] = seq[:-1]

    rolling_min = np.minimum(seq, seq_prev)
    rolling_max = np.maximum(seq, seq_prev)

    return np.concatenate([seq, delta, rolling_min, rolling_max], axis=-1)


# --- Improved Transformer Classifier Model with Pre-LN, Learned Pos Embeddings & CLS Token & TCN Conv ---
class SepsisTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        use_dual_pooling=True,
        norm_first=True,
        pos_type="learned",
        max_len=240,
        use_cls_token=True,
        use_tcn_conv=False,
    ):
        super().__init__()
        self.use_dual_pooling = use_dual_pooling
        self.use_cls_token = use_cls_token
        self.pos_type = pos_type
        self.d_model = d_model

        if use_tcn_conv:
            self.tcn_conv = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1), nn.BatchNorm1d(input_dim), nn.GELU()
            )
        else:
            self.tcn_conv = None

        self.embedding = nn.Linear(input_dim, d_model)

        if pos_type == "learned":
            self.pos_encoder = nn.Embedding(max_len, d_model)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.input_layer_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = TemporalAttentionPooling(d_model)

        in_features = d_model
        if use_dual_pooling:
            in_features += d_model
        if use_cls_token:
            in_features += d_model

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, padding_mask):
        if self.tcn_conv is not None:
            x_conv = x.transpose(1, 2)
            x = x + self.tcn_conv(x_conv).transpose(1, 2)

        B, L, _ = x.shape
        x_emb = self.embedding(x)

        if self.pos_type == "learned":
            pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            x_emb = x_emb + self.pos_encoder(pos_ids)
        else:
            x_emb = self.pos_encoder(x_emb)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls_tokens, x_emb], dim=1)  # (B, L+1, d_model)
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, L+1)

        x_emb = self.input_layer_norm(x_emb)
        out = self.transformer_encoder(x_emb, src_key_padding_mask=padding_mask)

        if self.use_cls_token:
            cls_repr = out[:, 0]
            seq_out = out[:, 1:]
            seq_mask = padding_mask[:, 1:]
        else:
            cls_repr = None
            seq_out = out
            seq_mask = padding_mask

        valid_lens = (~seq_mask).sum(dim=1).clamp(min=1)
        last_indices = valid_lens - 1
        last_repr = seq_out[torch.arange(seq_out.size(0)), last_indices]

        to_pool = []
        if self.use_cls_token:
            to_pool.append(cls_repr)
        to_pool.append(last_repr)

        if self.use_dual_pooling:
            attn_repr = self.attn_pool(seq_out, seq_mask)
            to_pool.append(attn_repr)

        pooled = torch.cat(to_pool, dim=-1)
        logits = self.classifier(pooled)
        return logits


# --- Improved PyTorch LSTM Model with Temporal Attention, TCN Conv & Bidirectional Option ---
class SepsisLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        use_dual_pooling=True,
        use_tcn_conv=False,
        bidirectional=False,
    ):
        super().__init__()
        self.use_dual_pooling = use_dual_pooling
        self.bidirectional = bidirectional
        if use_tcn_conv:
            self.tcn_conv = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1), nn.BatchNorm1d(input_dim), nn.GELU()
            )
        else:
            self.tcn_conv = None

        num_dirs = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.attn_pool = TemporalAttentionPooling(hidden_dim * num_dirs)

        in_features = (hidden_dim * num_dirs) * 2 if use_dual_pooling else (hidden_dim * num_dirs)
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, lengths):
        if self.tcn_conv is not None:
            x_conv = x.transpose(1, 2)
            x = x + self.tcn_conv(x_conv).transpose(1, 2)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, (hn, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        if self.bidirectional:
            last_hn = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            last_hn = hn[-1]

        if self.use_dual_pooling:
            B, L, H = out.size()
            lengths_dev = lengths.to(x.device)
            mask_t = torch.arange(L, device=x.device).unsqueeze(0) < lengths_dev.unsqueeze(1)
            padding_mask = ~mask_t
            attn_out = self.attn_pool(out, padding_mask)
            pooled = torch.cat([last_hn, attn_out], dim=-1)
        else:
            pooled = last_hn

        logits = self.classifier(pooled)
        return logits


def normalize_features(X_train_list, X_test_list):
    all_steps = np.concatenate([s for s in X_train_list], axis=0)
    mean = np.mean(all_steps, axis=0, keepdims=True)
    std = np.std(all_steps, axis=0, keepdims=True) + 1e-6
    return [(s - mean) / std for s in X_train_list], [(s - mean) / std for s in X_test_list]


# --- Evaluation Helper Functions ---


def evaluate_lstm_model(model, X_test, input_dim, device="cpu"):
    from .data_module import EPSepsisDataset, collate_ep_batch

    model.eval()
    model.to(device)
    dataset = EPSepsisDataset(X_test, [0] * len(X_test), input_dim)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_ep_batch)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            x, _, lengths, _ = batch
            logits = model(x.to(device), lengths=lengths.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(1)
            all_probs.append(probs)

    return np.concatenate(all_probs)


def evaluate_transformer_model(model, X_test, input_dim, device="cpu"):
    from .data_module import EPSepsisDataset, collate_ep_batch

    model.eval()
    model.to(device)
    dataset = EPSepsisDataset(X_test, [0] * len(X_test), input_dim)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_ep_batch)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            x, _, _, padding_mask = batch
            logits = model(x.to(device), padding_mask=padding_mask.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(1)
            all_probs.append(probs)

    return np.concatenate(all_probs)

import torch
import torch.nn as nn
from src.app.api.models import register_model


class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 2)), nn.Tanh(), nn.Linear(max(16, d_model // 2), 1)
        )

    def forward(self, x, padding_mask=None):
        scores = self.attn(x).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        context = (x * weights).sum(dim=1)
        return context


@register_model("lstm", "sepsis_lstm")
class LSTMModel(nn.Module):
    """LSTM sequence policy / model with temporal pooling."""

    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        use_dual_pooling=True,
        use_tcn_conv=False,
        bidirectional=False,
        n_actions=2,
        **kwargs,
    ):
        super().__init__()
        obs_dim = kwargs.get("obs_dim") or kwargs.get("n_inputs") or input_dim
        self.input_dim = obs_dim
        self.use_dual_pooling = use_dual_pooling
        self.bidirectional = bidirectional
        num_dirs = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            self.input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.attn_pool = TemporalAttentionPooling(hidden_dim * num_dirs)
        in_features = (hidden_dim * num_dirs) * 2 if use_dual_pooling else (hidden_dim * num_dirs)
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, (hn, _) = self.lstm(x)
        if self.use_dual_pooling:
            attn_repr = self.attn_pool(out)
            last_repr = hn[-1] if not self.bidirectional else torch.cat([hn[-2], hn[-1]], dim=-1)
            pooled = torch.cat([last_repr, attn_repr], dim=-1)
        else:
            pooled = hn[-1] if not self.bidirectional else torch.cat([hn[-2], hn[-1]], dim=-1)
        return self.head(pooled)

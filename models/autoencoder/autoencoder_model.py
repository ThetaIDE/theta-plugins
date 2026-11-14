"""Autoencoder architectures for the unsupervised paradigm.

A sequence autoencoder for the padded (B, L, D) batches the MIMIC-style loaders
produce, and a plain feed-forward one for flat (B, D) features. Both expose
`encode` so an evaluation protocol can score the latent space without knowing
the architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.app.api.models import register_model


@register_model("autoencoder", "sequence_autoencoder", "seq_autoencoder")
class SequenceAutoencoder(nn.Module):
    """LSTM encoder/decoder that reconstructs its own input sequence.

    The encoder's final hidden state is the representation; the decoder repeats
    it across the sequence and maps back to input width. `lengths` and
    `padding_mask` are accepted because the shared loaders supply them, and the
    mask is what keeps padding out of the reconstruction loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        enc_out = hidden_dim * (2 if bidirectional else 1)

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.to_latent = nn.Linear(enc_out, latent_dim)
        self.decoder = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_input = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        """Return the (B, latent_dim) representation of each sequence."""
        outputs, _ = self.encoder(x)
        if lengths is not None:
            # Take the last real step per sequence rather than the last padded one.
            idx = (lengths.to(outputs.device).long() - 1).clamp(min=0)
            idx = idx.view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
            pooled = outputs.gather(1, idx).squeeze(1)
        else:
            pooled = outputs[:, -1, :]
        return self.to_latent(pooled)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x, lengths=lengths)
        repeated = latent.unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = self.decoder(repeated)
        recon = self.to_input(decoded)
        if padding_mask is not None:
            # True marks padding; zero it on both sides so it cannot carry loss.
            keep = (~padding_mask).unsqueeze(-1).to(recon.dtype)
            recon = recon * keep
        return recon, latent


@register_model("dense_autoencoder", "mlp_autoencoder")
class DenseAutoencoder(nn.Module):
    """Symmetric feed-forward autoencoder for flat (B, D) feature vectors."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.encoder(x.reshape(x.size(0), -1))

    def forward(self, x: torch.Tensor, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        return self.decoder(latent).reshape(x.shape), latent

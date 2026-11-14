"""Lightning execution adapter owned by the autoencoder package."""
import lightning as L
import torch
from src.app.api.models import build_model


class AutoencoderLightningModule(L.LightningModule):
    def __init__(self, architecture_name, input_dim, lr=1e-3, **params):
        super().__init__()
        if architecture_name not in {"autoencoder","sequence_autoencoder","seq_autoencoder","dense_autoencoder","mlp_autoencoder","basics/autoencoder","basics/dense_autoencoder"}:
            raise ValueError(f"Unknown unsupervised architecture: {architecture_name}")
        self.save_hyperparameters()
        if 'hidden_dims' in params and 'hidden_dim' not in params:
            params['hidden_dim'] = list(params.pop('hidden_dims'))[0]
        self.model = build_model(architecture_name, obs_dim=input_dim, **params)
        self.lr = lr

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def encode(self, x, **kwargs):
        return self.model.encode(x, **kwargs)

    def _step(self, batch, stage):
        x = batch['obs'] if isinstance(batch, dict) else batch[0]
        options = {}
        if isinstance(batch, (list, tuple)) and len(batch) >= 4:
            options = {'lengths':batch[2], 'padding_mask':batch[3]}
        reconstruction, _ = self(x, **options)
        error = (reconstruction - x).square()
        mask = options.get('padding_mask')
        if mask is not None:
            keep = (~mask).unsqueeze(-1).expand_as(error)
            loss = error.masked_select(keep).mean()
        else:
            loss = error.mean()
        self.log(stage + '/loss', loss, on_epoch=True, batch_size=x.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

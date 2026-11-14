"""Lightning classification adapter owned by the MLP package."""
import lightning as L
import torch
import torch.nn.functional as F
from src.app.api.models import build_model


class MLPClassifierLightningModule(L.LightningModule):
    def __init__(self, architecture_name, input_dim, lr=1e-3, **params):
        super().__init__()
        self.save_hyperparameters()
        if 'hidden_dims' in params and 'hidden_sizes' not in params:
            params['hidden_sizes'] = params.pop('hidden_dims')
        self.model = build_model(architecture_name, obs_dim=input_dim, **params)
        self.lr = lr

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _step(self, batch, stage):
        if isinstance(batch, dict):
            x, y = batch['obs'], batch['target']
        else:
            x, y = batch[:2]
        logits = self(x)
        if logits.ndim == 1 or logits.shape[-1] == 1:
            loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), y.float().reshape(-1))
        else:
            loss = F.cross_entropy(logits, y.long().reshape(-1))
        self.log(stage + '/loss', loss, on_epoch=True, batch_size=x.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

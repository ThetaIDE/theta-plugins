"""Time-series and sampled-state visualizations for BlendRL routing."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.app.api.plotting import BasePlotter


class BlendRoutingPlotter(BasePlotter):
    def __init__(self, *, paths):
        super().__init__('blend_routing', paths=paths)

    def run(self, exp_id, cli_overrides=None):
        cfg, group, output = self.get_effective_config(exp_id, cli_overrides)
        runs = self.load_metrics(group, exp_id)
        metrics = sorted({str(c) for versions in runs.values() for df in versions.values()
                          for c in df.columns if str(c).startswith('blend/')})
        if metrics:
            self.plot_metric_series(exp_id, group, output, metrics, cfg)
        root = self.workspace.results / 'logs' / group / Path(exp_id).stem
        sample_files = list(root.rglob('model_monitoring_samples.npz')) if root.exists() else []
        for index, path in enumerate(sorted(sample_files)):
            self.plot_samples(path, output / 'blend_routing' / f'sample_{index}', cfg)

    @staticmethod
    def plot_samples(path, output, cfg):
        data = np.load(path, allow_pickle=False)
        states = np.asarray(data['states'])
        weights = np.asarray(data['weights'])
        if 'expert_names' not in data.files:
            raise ValueError(f'{path} predates BlendRL 1.4.1 and has no expert_names; rerun to plot routing samples')
        names = np.asarray(data['expert_names']).astype(str)
        if states.ndim < 2 or weights.ndim != 2 or len(states) != len(weights) or weights.shape[1] != len(names) or len(states) == 0:
            raise ValueError(f'Invalid routing samples: {path}')
        output.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, name in enumerate(names):
            ax.hist(weights[:, i], bins=20, range=(0, 1), alpha=.45, label=name)
        ax.set(xlabel='Routing weight', ylabel='Sample count', title='Routing weight distribution')
        ax.legend()
        fig.tight_layout(); fig.savefig(output / 'weights.png', dpi=150); plt.close(fig)
        threshold = float(cfg.get('authority_threshold', .9))
        if not .5 < threshold <= 1:
            raise ValueError('authority_threshold must be in (0.5, 1]')
        exclusive = [(weights[:, i] >= threshold).mean() for i in range(len(names))]
        labels = list(names) + ['Mixed']
        shares = exclusive + [float((weights.max(axis=1) < threshold).mean())]
        fig, ax = plt.subplots(figsize=(8, 2.5))
        left = 0.0
        for label, share in zip(labels, shares):
            ax.barh(['Samples'], [share], left=left, label=label)
            left += share
        ax.set(xlim=(0, 1), xlabel='Fraction of sampled observations', title=f'Routing authority (threshold {threshold:g})')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -0.25), ncol=3)
        fig.tight_layout(); fig.savefig(output / 'authority.png', dpi=150); plt.close(fig)
        # Bound PCA cost for high-dimensional observations and large samples.
        indices = None
        if len(states) > 500:
            indices = np.linspace(0, len(states) - 1, 500, dtype=int)
            states, weights = states[indices], weights[indices]
        flat = states.reshape(len(states), -1).astype(float)
        if flat.shape[1] > 2048:
            flat = flat[:, np.linspace(0, flat.shape[1] - 1, 2048, dtype=int)]
        if len(states) >= 2 and flat.shape[1] >= 2:
            flat -= flat.mean(axis=0)
            _, _, directions = np.linalg.svd(flat, full_matrices=False)
            projected = flat @ directions[:2].T
            for i, name in enumerate(names):
                fig, ax = plt.subplots(figsize=(7, 5))
                points = ax.scatter(projected[:, 0], projected[:, 1], c=weights[:, i], vmin=0, vmax=1, s=12, cmap='viridis')
                fig.colorbar(points, ax=ax, label='Routing weight')
                ax.set(xlabel='Principal component 1', ylabel='Principal component 2', title=f'Routing by state: {name}')
                fig.tight_layout(); fig.savefig(output / f'state_{i}.png', dpi=150); plt.close(fig)
            if cfg.get('tsne', False) and len(states) >= 4:
                from sklearn.manifold import TSNE
                embedding = TSNE(n_components=2, perplexity=min(30, len(states) - 1), random_state=42).fit_transform(flat)
                for i, name in enumerate(names):
                    fig, ax = plt.subplots(figsize=(7, 5))
                    points = ax.scatter(embedding[:, 0], embedding[:, 1], c=weights[:, i], vmin=0, vmax=1, s=12, cmap='viridis')
                    fig.colorbar(points, ax=ax, label='Routing weight')
                    ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2', title=f'Routing manifold: {name}')
                    fig.tight_layout(); fig.savefig(output / f'manifold_{i}.png', dpi=150); plt.close(fig)
        # Optional experiment-supplied cohort labels or reference distance scores.
        for key in ('cohort', 'distance'):
            if key not in data:
                continue
            x = np.asarray(data[key])
            if indices is not None and len(x) == len(data['states']):
                x = x[indices]
            if len(x) != len(weights):
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            if key == 'cohort':
                labels = np.unique(x)
                for i, name in enumerate(names):
                    ax.plot(labels.astype(str), [weights[x == label, i].mean() for label in labels], marker='o', label=name)
            else:
                order = np.argsort(x)
                bins = [chunk for chunk in np.array_split(order, min(5, len(order))) if len(chunk)]
                for i, name in enumerate(names):
                    ax.plot([x[chunk].mean() for chunk in bins], [weights[chunk, i].mean() for chunk in bins], marker='o', label=name)
            ax.set(xlabel=key.title(), ylabel='Mean routing weight', ylim=(0, 1), title=f'Routing by {key}')
            ax.legend()
            fig.tight_layout(); fig.savefig(output / f'{key}.png', dpi=150); plt.close(fig)

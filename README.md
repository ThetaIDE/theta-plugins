# Theta Plugins & Components

Maintained research providers, environments, model architectures, data modules,
evaluators, plots, application extensions, experiment profiles, dependency
suites, the Optuna tuner (`tuners/optuna`) and the Slurm scheduler (`schedulers/slurm`)
for Theta-IDE. The IDE separately ships its generic framework, four base
paradigms and bundled providers.

All 39 maintained packages use native manifests: `storage_version: 3`,
`api_version: 1`, explicit capabilities, package-relative entry points, resources,
config mappings and dependencies. Config destinations are logical selectors
relative to the workspace config root (`model/nsfr.yaml`). The authoritative record is each package's
`.theta_component.json`. App packages also retain `plugin.json` for panel IDs,
settings and default activation. A source folder need not mirror its installation
location.

## Build locally

Use the sibling IDE's Python environment and shared validation:

```sh
../Theta-IDE/venv/bin/python scripts/package_plugins.py --validate-only
../Theta-IDE/venv/bin/python scripts/package_plugins.py --output /tmp/theta-packages
# Optional: --only dnn sb3-suite, or --ide-root /path/to/Theta-IDE
```

The builder validates all maintained manifests, provider/config selectors,
dependency ownership, cycles and versions before writing output. It also rejects
package code that imports host modules outside the Theta SDK (`src.app.api`) or
reaches into the host window. Archive names
and versions come from source metadata. `index.json` preserves manifest fields
and records each local archive's SHA-256. Archives contain declared payloads and
runtime assets; development caches and duplicate embedded suite sources are
excluded. Builds are deterministic for identical input bytes.

This command only builds local artifacts. It never copies archives, rewrites a
registry, publishes, or pushes to a Hub. The obsolete per-model `pyproject.toml`
files that advertised historical `src.usr` entry points have been retired;
maintained packages install through native archives. Third-party Python
distribution entry points remain supported by the IDE.

## Install and edit

From Theta-IDE, choose the actual target workspace:

```sh
python -m src.app.components --root /path/to/workspace install /tmp/theta-packages/dnn-1.1.0.zip
python -m src.app.components --root /path/to/workspace copy dnn my_dnn
python -m src.app.components --root /path/to/workspace export my_dnn --out /tmp/my_dnn.zip
```

Install component dependencies first when using local archives. The existing
Hub dependency workflow is separate from local verification. `--no-dependencies`
suppresses Python dependency installation; it still checks required component
ownership. Installed packages go under `src/plugins/{components,app}/installed`,
editable copies under `src/usr`, and editable profiles under `src/usr/config`.
Older archive formats are rejected rather than adapted or guessed. Existing user/installed implementations are never
rewritten by source conversion.

## Ownership decisions

- DNN owns `dnn`; MLP owns `mlp` and `standard_mlp`. Their aliases no longer collide.
- Thetalib and the basic/deep-learning/robotics suites compose independent
  dependency packages. MIMIC profiles depend on independent environment, data,
  evaluator and plotting packages instead of embedding copies.
- `standard-figures` supplies profiles and depends on the IDE's bundled
  `standard_figures`; it does not install duplicate plotter implementations.
- BlendRL owns its routing plot. NSFR owns shared CartPole rules; NEUMANN depends
  on NSFR. MIMIC owns its experiment rule assets.
- `doom-sim` depends on `doom-env`. Activating the panel does not copy an
  environment into the workspace. Panel identities remain `doom` and `obsidian`,
  while package identities remain `doom-sim` and `obsidian-notes`.
- Autoencoder and MLP classifier profiles explicitly select their package-owned
  Lightning training components. SB3/d3rlpy runners resolve workspace/config
  roots themselves; the generic worker supplies package context and leases.
- SB3 algorithm capabilities declare `episode_outcomes: true` for objective/replicate planning; filenames do not establish that contract.
- Slurm owns submission, allocation validation, job headers and allocation-side
  supervision; it ships no site profiles. Clusters are workspace-authored site packages.
- Optuna owns its Hydra sweeper profile, study storage, pruning and parallel trials.
- d3rlpy targets its 2.x API and requires an explicit transition dataset. It
  reports unsupported algorithms/APIs rather than falling back to another
  algorithm, synthesizing data, or silently omitting training.

Use relative imports inside packages and import the host only through the
Theta SDK (`src.app.api`, see the IDE's `docs/reference/SDK.md`). A package
publishes objects for others in its manifest `exports`; consumers declare the
dependency and load them with `src.app.api.packages.provider_export`. Resources
use `package://ID/path` and the environment/rules APIs. Profiles live under `configs/<group>/...`, with
workspace overrides taking precedence. Do not introduce source-depth calculations,
`sys.path` provider roots or global import aliases.

## Verification

```sh
../Theta-IDE/venv/bin/python -m pytest tests/test_native_integration.py -q
```

The tests build the real archives and install them into disposable workspaces
with separate state directories. They cover all declarations/imports, headless
listing, provider ownership, resources/rules, all four paradigm workers,
SB3/BlendRL checkpoints, editable copy/edit/export/reinstall/update/rollback,
scaffolding, experiment exports, objective/tuner selection, Slurm generation and
app-plugin identity/Obsidian activation. They never install into the IDE checkout.
Tests use dependencies already present in the chosen Python environment; they do
not download Python packages or publish artifacts. Full MIMIC experiments need
external clinical datasets/checkpoints; Atari/MuJoCo need their runtime assets;
interactive DOOM/Antigravity browser services are not exercised headlessly.

See the IDE's `docs/reference/PLUGIN_STORAGE.md` for the package format.

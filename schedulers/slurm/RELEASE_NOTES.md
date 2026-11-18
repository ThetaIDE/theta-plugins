# 3.0.0

Ships no site profiles. The ARC template is gone; author each cluster as a site package in
the workspace (`run_pipeline.py site configure SITE --scheduler slurm …`). Source moved to
theta-plugins `schedulers/slurm`.

# 2.0.0

Implements scheduler contract v2: remote jobs run from explicit experiment bundles.
Declares its `packaging` dependency. Builds the job script's environment block and
Python command itself (`shell.py`); the IDE no longer provides those helpers
(Theta SDK 3.1).

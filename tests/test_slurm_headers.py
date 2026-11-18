from package_fixtures import IDE


import pytest
from omegaconf import OmegaConf

from package_fixtures import package_module

# The scheduler implementation is the optional Slurm component, loaded from an installed copy.
_headers = package_module('slurm', 'headers.py')
generate_sbatch_header, generate_sbatch_script = _headers.generate_sbatch_header, _headers.generate_sbatch_script


@pytest.fixture(scope='module')
def site_configs(tmp_path_factory):
    """Config root of the fixture two-partition site, authored in a private workspace."""
    from package_fixtures import cluster_site
    return cluster_site(tmp_path_factory.mktemp('cluster-site')) / 'configs'


def compose_site(site, configs):
    """Compose framework defaults with an authored site package's profiles."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    defaults = IDE / 'src/app/config_defaults'
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(defaults), version_base=None):
            return compose(config_name='config', overrides=[f'site={site}', f'hydra.searchpath=[file://{configs}]'])
    finally:
        GlobalHydra.instance().clear()


def test_slurm_compute_node_defaults():
    site_cfg = OmegaConf.create({
        "name": "cluster",
        "mail_user": "test@example.com",
        "mail_type": "END,FAIL",
        "resources": {
            "partition": "gpu",
            "cores": 16,
            "memory": "32G",
            "time": "01:00:00",
            "gpus": 1,
        },
        "compute_resources": {
            "partition": "cpu",
            "cores": 4,
            "memory": "8G",
            "time": "01:00:00",
            "gpus": 0,
        },
    })

    # Test 1: Targeting compute nodes (cpu partition) without explicit cores/memory in exp
    cfg = OmegaConf.create({
        "site": site_cfg,
        "partition": "cpu",
        "consolidate": False,
        "resources": {
            "cores": None,
            "memory": None,
        },
    })
    header = generate_sbatch_header("test_job", "logs", cfg, is_consolidated=False)
    assert "#SBATCH --partition=cpu" in header
    assert "#SBATCH --cpus-per-task=4" in header
    assert "#SBATCH --mem=8G" in header
    assert "--gres" not in header

    # Test 2: Targeting GPU partition retains 16 cores and 32G
    cfg_gpu = OmegaConf.create({
        "site": site_cfg,
        "partition": "gpu",
        "consolidate": False,
        "resources": {
            "cores": None,
            "memory": None,
        },
    })
    header_gpu = generate_sbatch_header("test_gpu_job", "logs", cfg_gpu, is_consolidated=False)
    assert "#SBATCH --partition=gpu" in header_gpu
    assert "#SBATCH --cpus-per-task=16" in header_gpu
    assert "#SBATCH --mem=32G" in header_gpu
    assert "#SBATCH --gres=gpu:1" in header_gpu


def test_slurm_split_partition_sites(site_configs):
    """A site's CPU and GPU variants produce the matching sbatch headers."""
    # 1. cluster/cpu: CPU partition, 4 cores, 8G mem, consolidate=False, no GPU
    cfg_cpu = compose_site("cluster/cpu", site_configs)
    header_cpu = generate_sbatch_header("test_cpu", "logs", cfg_cpu)
    assert "#SBATCH --partition=cpu" in header_cpu
    assert "#SBATCH --cpus-per-task=4" in header_cpu
    assert "#SBATCH --mem=8G" in header_cpu
    assert "--gres" not in header_cpu
    assert cfg_cpu.site.consolidate is False
    assert "#SBATCH --time=" not in header_cpu

    # 2. cluster/gpu: GPU partition, 16 cores, 32G mem, consolidate=True, 1 GPU
    cfg_gpu = compose_site("cluster/gpu", site_configs)
    header_gpu = generate_sbatch_header("test_gpu", "logs", cfg_gpu)
    assert "#SBATCH --partition=gpu" in header_gpu
    assert "#SBATCH --cpus-per-task=16" in header_gpu
    assert "#SBATCH --mem=32G" in header_gpu
    assert "#SBATCH --gres=gpu:1" in header_gpu
    assert cfg_gpu.site.consolidate is True
    assert "#SBATCH --time=" not in header_gpu


def test_slurm_script_is_depersonalized(site_configs):
    """Verify that generated Slurm scripts contain no hardcoded home paths or user emails."""
    cfg = compose_site("cluster/default", site_configs)
    script = generate_sbatch_script(
        job_name="depersonalized_test",
        cmd_args=["+experiment=cartpole/test"],
        log_dir="src/usr/results/logs/slurm",
        cfg=cfg,
    )
    assert 'export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"' in script
    assert "/Users/" not in script
    assert "/home/" not in script
    assert "--mail-user" not in script


def test_slurm_allocation_account_and_qos():
    cfg = OmegaConf.create({'site': {'resources': {'account': 'lab', 'qos': 'normal'}},
                            'resources': {'account': 'project'}})
    header = generate_sbatch_header('job', 'logs', cfg)
    assert '#SBATCH --account=project\n' in header
    assert '#SBATCH --qos=normal\n' in header
    cfg.resources.account = 'project\n#SBATCH --nodes=100'
    import pytest
    with pytest.raises(ValueError, match='single-line'):
        generate_sbatch_header('job', 'logs', cfg)

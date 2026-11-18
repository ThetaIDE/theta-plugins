"""Slurm cluster utilities.

Provides functions for generating SBATCH scripts and submitting jobs.
All cluster-specific values (email, partition, cores) are read from site config.
"""

import re
import shlex
import subprocess

from omegaconf import OmegaConf

from src.app.api.execution import provider_worker_command

from .shell import env_block, python_cmd as shell_python_cmd


def get_gres_header(res, site_cfg):
    """Generate the appropriate --gres header.

    Emits --gres=gpu:{gpus} automatically when gpus > 0 unless explicitly disabled.
    """
    gpus = res.get("gpus")
    gres = res.get("gres")
    gpu_type = res.get("gpu_type")
    no_gres = res.get("no_gres") or getattr(site_cfg, "no_gres", False)
    partition = str(res.get("partition", "")).lower()
    is_cpu_partition = partition in ("common", "serial", "cpu", "standard", "debug_cpu")

    if no_gres or is_cpu_partition or gres in ("none", "False", "false") or not gpus or int(gpus) == 0:
        return ""
    if gres:
        return f"#SBATCH --gres={gres}\n"

    # Optional site-specific gres format template (e.g., "gpu:{gpu_type}:{gpus}")
    gres_format = getattr(site_cfg, "gres_format", None)
    if gres_format and gpu_type:
        return f"#SBATCH --gres={gres_format.format(gpu_type=gpu_type, gpus=gpus)}\n"
    elif gres_format and not gpu_type and "{gpu_type}" not in gres_format:
        return f"#SBATCH --gres={gres_format.format(gpus=gpus)}\n"

    if gpu_type:
        return f"#SBATCH --gres=gpu:{gpu_type}:{gpus}\n"
    return f"#SBATCH --gres=gpu:{gpus}\n"


def generate_sbatch_header(job_name, log_dir, cfg, dependency=None, dependency_type="afterok", is_consolidated=False, resources=None):
    """Generate standardized SBATCH script header using pure Hydra composition."""
    site_cfg = getattr(cfg, "site", None)

    from .resources import normalize
    res = resources if resources is not None else normalize(cfg, consolidated=is_consolidated)

    partition = res.get("partition")
    cfg_consolidate = cfg.get("consolidate", None)
    site_consolidate = getattr(site_cfg, "consolidate", False) if site_cfg else False
    effective_consolidate = is_consolidated or (cfg_consolidate if cfg_consolidate is not None else site_consolidate)
    cores = res.get("cores")
    if cores is None:
        cores = 16 if effective_consolidate else 4
    memory = res.get("memory")
    if memory is None:
        memory = "32G" if effective_consolidate else "8G"
    time = res.get("time")
    nodes = res.get("nodes", 1)

    mail_user = getattr(site_cfg, "mail_user", None)
    mail_type = getattr(site_cfg, "mail_type", "END,FAIL")

    script = "#!/bin/bash\n"
    script += f"#SBATCH --job-name={shlex.quote(str(job_name))}\n"
    if partition:
        script += f"#SBATCH --partition={partition}\n"
    for key in ("account", "qos"):
        if res.get(key): script += f"#SBATCH --{key}={res[key]}\n"
    script += get_gres_header(res, site_cfg)
    if time:
        script += f"#SBATCH --time={time}\n"
    if nodes:
        script += f"#SBATCH --nodes={nodes}\n"
    script += "#SBATCH --ntasks=1\n"
    if cores:
        script += f"#SBATCH --cpus-per-task={cores}\n"
    if memory:
        script += f"#SBATCH --mem={memory}\n"
    script += f"#SBATCH --output={shlex.quote(str(log_dir) + '/%x_%j.out')}\n"
    script += f"#SBATCH --error={shlex.quote(str(log_dir) + '/%x_%j.err')}\n"

    # Extra site-specific sbatch flags
    for flag in getattr(site_cfg, "extra_sbatch_flags", []) or []:
        script += f"#SBATCH {flag}\n"

    if mail_user:
        script += f"#SBATCH --mail-type={shlex.quote(str(mail_type))}\n"
        script += f"#SBATCH --mail-user={shlex.quote(str(mail_user))}\n"
    if dependency:
        script += f"#SBATCH --dependency={shlex.quote(str(dependency_type) + ':' + str(dependency))}\n"
    # Directives are one physical line each; reject newline/command injection.
    for value in [job_name, str(log_dir), mail_user, mail_type, dependency, dependency_type, *res.values(), *(getattr(site_cfg, "extra_sbatch_flags", []) or [])]:
        if value is not None and any(c in str(value) for c in "\r\n\x00"):
            raise ValueError("SLURM directive values must be single-line")
    return script


def generate_sbatch_script(job_name, cmd_args, log_dir, cfg, dependency=None, is_consolidated=False):
    """Generate an SBATCH script string for Slurm submission."""
    import shlex

    script = generate_sbatch_header(
        job_name=job_name, log_dir=log_dir, cfg=cfg, dependency=dependency, is_consolidated=is_consolidated
    )
    script += "\n"

    site_cfg = getattr(cfg, "site", None)
    script += env_block(site_cfg)
    script += "\n"

    # Construct the python command using site-aware venv path
    python_cmd = shell_python_cmd(site_cfg)
    from src.app.api.execution import resolve_engine
    engine_script, engine_python = resolve_engine({}, cfg)
    if engine_python: python_cmd = shlex.quote(engine_python)
    cmd_str = provider_worker_command(python_cmd, engine_script, cmd_args)
    script += f'echo "Running: {cmd_str}"\n'
    script += f"{cmd_str}\n"

    return script


def submit_sbatch(script_content):
    """Submit an SBATCH script to Slurm and return the job ID.

    Uses --parsable for robust job ID extraction.
    Returns None if submission fails or sbatch is not available.
    """
    try:
        from .resources import submission_environment
        environment = submission_environment()
        # Use --parsable for reliable job ID output (just the number)
        result = subprocess.run(
            ["sbatch", "--parsable"], input=script_content, env=environment, capture_output=True, text=True, check=True
        )
        job_id = result.stdout.strip().split(";")[0]  # --parsable may include cluster name after ;
        if job_id.isdigit():
            print(f" -> Job ID: {job_id}", flush=True)
            return job_id
        else:
            # Fallback: try regex parsing for non-standard sbatch output
            match = re.search(r"(\d+)", result.stdout)
            if match:
                job_id = match.group(1)
                print(f" -> Job ID: {job_id}", flush=True)
                return job_id
            print(f" -> Could not parse job ID from: {result.stdout}", flush=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f" -> Error submitting job: {e.stderr.strip()}", flush=True)
        return None
    except FileNotFoundError:
        print(" -> (sbatch not available locally)", flush=True)
        return None

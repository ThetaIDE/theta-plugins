"""One validated allocation used by preflight and SBATCH generation."""
import os
import re

from src.app.api.errors import ConfigurationError

CPU_PARTITIONS = {'common', 'serial', 'cpu', 'standard', 'debug_cpu'}
# Everything else must be represented in the managed allocation or executor.
EXTRA_FLAGS = {'constraint', 'reservation', 'licenses', 'hint', 'nice', 'mail-user', 'mail-type'}


def normalize(cfg, *, consolidated=False):
    site = cfg.get('site') or {}
    base = {k: v for k, v in (site.get('resources') or {}).items() if v is not None}
    experiment = {k: v for k, v in (cfg.get('resources') or {}).items() if v is not None}
    partition = cfg.get('partition') or experiment.get('partition') or base.get('partition')
    if str(partition).lower() in CPU_PARTITIONS:
        base.update({k: v for k, v in (site.get('compute_resources') or {}).items() if v is not None})
    result = {**base, **experiment}
    if cfg.get('partition') is not None: result['partition'] = cfg['partition']
    large = consolidated or cfg.get('consolidate', site.get('consolidate', False))
    for key, value in {'nodes': 1, 'ntasks': 1, 'cores': 16 if large else 4,
                       'memory': '32G' if large else '8G', 'gpus': 0}.items():
        result.setdefault(key, value)
    for key in ('nodes', 'ntasks', 'cores', 'gpus'):
        value = result[key]
        if type(value) is not int or value < (0 if key == 'gpus' else 1):
            raise ConfigurationError(f'resources.{key} must be a positive integer' if key != 'gpus' else 'resources.gpus must be a nonnegative integer')
    if result['ntasks'] != 1:
        raise ConfigurationError('SLURM experiment allocations support exactly one task')
    gpu_type = result.get('gpu_type')
    if gpu_type and not re.fullmatch(r'[A-Za-z0-9_-]+', str(gpu_type)):
        raise ConfigurationError('resources.gpu_type must be a GPU type token')
    gres = result.get('gres')
    if not gres and result['gpus']:
        template = site.get('gres_format')
        if not template or ('{gpu_type}' in template and not gpu_type):
            template = 'gpu:{gpu_type}:{gpus}' if gpu_type else 'gpu:{gpus}'
        try: gres = template.format(gpu_type=gpu_type, gpus=result['gpus'])
        except (KeyError, ValueError) as exc: raise ConfigurationError('Invalid site gres_format') from exc
        result['gres'] = gres
    if gres in ('none', 'False', 'false'): gres = None
    if gres:
        match = re.fullmatch(r'gpu:(?:([A-Za-z0-9_-]+):)?(\d+)', str(gres))
        if not match or int(match[2]) != result['gpus'] or (result.get('gpu_type') and match[1] != result['gpu_type']):
            raise ConfigurationError('resources.gres must match the declared GPU type/count')
    if result['gpus'] and (result.get('no_gres') or site.get('no_gres') or str(result.get('partition')).lower() in CPU_PARTITIONS):
        raise ConfigurationError('GPU request conflicts with CPU partition/no_gres')
    for flag in site.get('extra_sbatch_flags') or []:
        match = re.fullmatch(r'--([a-z-]+)(?:=[^\s\x00]+)?', str(flag))
        if not match or match[1] not in EXTRA_FLAGS:
            raise ConfigurationError(f'Unsupported extra_sbatch_flags directive: {flag}; use managed resource fields')
    for value in [*result.values(), *(site.get('extra_sbatch_flags') or [])]:
        if value is not None and any(c in str(value) for c in '\r\n\x00'):
            raise ValueError('SLURM directive values must be single-line')
    for key, value in result.items():
        if isinstance(value, str) and (any(c.isspace() for c in value) or value.startswith('-')):
            raise ConfigurationError(f'resources.{key} must be a single directive token')
    return result


def submission_environment():
    """SBATCH_* variables override script directives. Refuse rather than silently alter intent."""
    keys = sorted(key for key in os.environ if key.startswith('SBATCH_'))
    if keys:
        raise ConfigurationError('Unset inherited SBATCH_* overrides before submission: ' + ', '.join(keys))
    return os.environ.copy()

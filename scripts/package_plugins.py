#!/usr/bin/env python3
"""Validate and build native local archives. Never writes to a Hub checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

ROOT = Path(__file__).resolve().parent.parent


def inventory(root=ROOT):
    """Only maintained top-level records; nested suite content is not a package."""
    return [(p.parent, json.loads(p.read_text())) for p in sorted(root.glob('*/*/.theta_component.json'))]


def payload_files(source, meta):
    paths = {source / '.theta_component.json'}
    for section in ('implementations', 'resources'):
        for relative in meta['payload'][section]:
            path = source / relative
            paths.update(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else paths.add(path)
    paths.update(source / value for value in meta['payload']['configs'].values())
    for name in ('plugin.json', 'README.md', 'RELEASE_NOTES.md', 'requirements.txt', 'LICENSE', 'LICENSE.txt'):
        if (source / name).is_file(): paths.add(source / name)
    return sorted(p for p in paths if not any(v in p.parts for v in ('__pycache__', '.git', 'node_modules')) and p.suffix != '.pyc')


def validate_all(records):
    from src.app.components.archive import validate_native_package
    from src.app.components.lifecycle import dependencies, check_version
    from src.app.core.provider_catalog import kind_name, ProviderCatalog
    from src.app.workspace import WorkspacePaths
    import tempfile
    # Bundled providers are dependencies, never hidden/replaced by archive ordering.
    with tempfile.TemporaryDirectory(prefix='theta-build-validation-') as temp:
        catalog = ProviderCatalog(WorkspacePaths(Path(temp)))
    owners = {p.id: p.manifest for p in catalog.packages}
    selectors = {(c.kind, s.casefold()): p.id for p in catalog.packages for c in p.capabilities for s in (c.id, *c.aliases)}
    for source, meta in records:
        validate_native_package(meta, source)
        if meta['id'] in owners: raise ValueError(f"Duplicate package ID: {meta['id']}")
        owners[meta['id']] = meta
        for cap in meta['capabilities']:
            for selector in (cap['id'], *cap.get('aliases', [])):
                key = (kind_name(cap['kind']), selector.casefold())
                if key in selectors: raise ValueError(f"Duplicate {key}: {selectors[key]} and {meta['id']}")
                selectors[key] = meta['id']
    # Profile selectors must be backed by declared capabilities, including suite-owned profiles.
    import yaml
    groups = {'model':('model',('architecture','name','type')), 'env':('env',('name',)),
              'agent':('algorithm',('name','algorithm')), 'data':('datamodule',('data_module',)),
              'eval':('evaluator',('evaluator',)), 'plot':('plotter',())}
    for source,meta in records:
        for dest,relative in meta['payload']['configs'].items():
            group=dest.split('/')[2]
            if group not in groups: continue
            data=yaml.safe_load((source/relative).read_text()) or {}
            if not isinstance(data,dict): raise ValueError(f'Profile must be a mapping: {source/relative}')
            kind,fields=groups[group]
            selector=next((data[key] for key in fields if data.get(key)), Path(dest).stem if group=='plot' else None)
            if isinstance(selector,str) and (kind,selector.casefold()) not in selectors:
                raise ValueError(f'{meta["id"]}: {dest} selects undeclared {kind}:{selector}')
            if data.get('lightning_module') and ('component',data['lightning_module'].casefold()) not in selectors:
                raise ValueError(f'{dest} names undeclared training component {data["lightning_module"]}')

    def visit(meta, stack=()):
        if meta['id'] in stack: raise ValueError('Dependency cycle: ' + ' -> '.join((*stack, meta['id'])))
        for key, constraint in dependencies(meta):
            if key not in owners: raise ValueError(f"Missing dependency {key} for {meta['id']}")
            check_version(owners[key]['version'], constraint)
            visit(owners[key], (*stack, meta['id']))
    for _, meta in records: visit(meta)


def package_component(source, output):
    meta = json.loads((source / '.theta_component.json').read_text())
    from src.app.components.archive import validate_native_package
    validate_native_package(meta, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    try:
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED) as archive:
            for path in payload_files(source, meta):
                info = ZipInfo(path.relative_to(source).as_posix(), date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ide-root', type=Path, default=ROOT.parent / 'Theta-IDE', help='Framework source providing shared native validation')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/native')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.ide_root.resolve()))
    records = inventory()
    validate_all(records)
    selected = set(args.only or [meta['id'] for _, meta in records])
    unknown = selected - {meta['id'] for _, meta in records}
    if unknown: parser.error(f'Unknown package IDs: {sorted(unknown)}')
    if args.validate_only:
        print(f'Validated {len(records)} native packages and their dependency/capability ownership')
        return
    results = []
    for source, meta in records:
        if meta['id'] not in selected: continue
        output = args.output / f"{meta['id']}-{meta['version']}.zip"
        digest = package_component(source, output)
        results.append({**meta, 'archive': output.name, 'sha256': digest})
        print(f"{meta['id']} {meta['version']} {digest}")
    (args.output / 'index.json').write_text(json.dumps({'packages': results}, indent=2) + '\n')
    print(f'Built {len(results)} local archives in {args.output}')


if __name__ == '__main__':
    main()

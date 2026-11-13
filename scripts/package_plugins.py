#!/usr/bin/env python3
"""Builds .zip packages for all plugins, models, methods, and environments in theta-plugins, computing SHA-256."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent.parent
HUB_ROOT = ROOT.parent / "theta-hub"
DIST = ROOT / "dist"
HUB_PACKAGES = HUB_ROOT / "packages"
# Where the Hub serves archives; must match each release "url" the IDE downloads.
HUB_PACKAGES_URL = "https://raw.githubusercontent.com/ThetaIDE/ThetaHub/main/packages"
# Native storage-v2 fields the Hub validator requires to match the archive's .theta_component.json.
NATIVE_FIELDS = ("storage_version", "api_version", "capabilities", "payload", "config_paths")
PLUGIN_FIELDS = ("default_enabled", "icon", "entry_point", "extra")

COMPONENTS = [
    # UI Plugins
    {"id": "obsidian-notes", "name": "obsidian-0.1.0.zip", "src": ROOT / "plugins" / "obsidian", "kind": "plugin"},
    {"id": "doom-sim", "name": "doom_sim-0.1.0.zip", "src": ROOT / "plugins" / "doom-sim", "kind": "plugin"},
    {"id": "antigravity", "name": "antigravity-0.1.5.zip", "src": ROOT / "plugins" / "antigravity", "kind": "plugin"},
    {"id": "ai_assistant", "name": "ai_assistant-1.0.0.zip", "src": ROOT / "plugins" / "ai_assistant", "kind": "plugin"},
    {"id": "theta_mcp", "name": "theta_mcp-1.0.0.zip", "src": ROOT / "plugins" / "theta_mcp", "kind": "plugin"},
    # Environments
    {"id": "doom-env", "name": "doom_env-1.0.1.zip", "src": ROOT / "envs" / "doom-env", "kind": "env", "paradigms": ["online_rl"]},
    {"id": "gym_classic_control", "name": "gym_classic_control-1.0.2.zip", "src": ROOT / "envs" / "gym_classic_control", "kind": "env", "paradigms": ["online_rl"]},
    {"id": "atari-env", "name": "atari_env-1.0.0.zip", "src": ROOT / "envs" / "atari-env", "kind": "env", "paradigms": ["online_rl"]},
    {"id": "mnist", "name": "mnist-1.0.0.zip", "src": ROOT / "envs" / "mnist", "kind": "env", "paradigms": ["supervised", "unsupervised"]},
    {"id": "mimic-env", "name": "mimic_env-1.0.1.zip", "src": ROOT / "envs" / "mimic-env", "kind": "env", "suite": "mimic", "hidden": True, "paradigms": ["supervised", "offline_rl"]},
    {"id": "pyrenees-env", "name": "pyrenees_env-1.0.0.zip", "src": ROOT / "envs" / "pyrenees-env", "kind": "env", "suite": "pyrenees", "hidden": True, "paradigms": ["offline_rl"]},
    # Models
    {"id": "dnn", "name": "dnn-1.0.0.zip", "src": ROOT / "models" / "dnn", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "dueling_resnet", "name": "dueling_resnet-1.0.0.zip", "src": ROOT / "models" / "dueling_resnet", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "nature_cnn", "name": "nature_cnn-1.0.0.zip", "src": ROOT / "models" / "nature_cnn", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "cew", "name": "cew-1.0.0.zip", "src": ROOT / "models" / "cew", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "blendrl", "name": "blendrl-1.0.3.zip", "src": ROOT / "models" / "blendrl", "kind": "model", "paradigms": ["online_rl", "offline_rl"]},
    {"id": "mlp", "name": "mlp-1.0.0.zip", "src": ROOT / "models" / "mlp", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "transformer", "name": "transformer-1.0.0.zip", "src": ROOT / "models" / "transformer", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "lstm", "name": "lstm-1.0.0.zip", "src": ROOT / "models" / "lstm", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "autoencoder", "name": "autoencoder-1.0.0.zip", "src": ROOT / "models" / "autoencoder", "kind": "model", "paradigms": ["unsupervised", "supervised"]},
    {"id": "nsfr", "name": "nsfr-1.0.0.zip", "src": ROOT / "models" / "nsfr", "kind": "model", "paradigms": ["online_rl", "offline_rl"]},
    {"id": "neumann", "name": "neumann-1.0.0.zip", "src": ROOT / "models" / "neumann", "kind": "model", "paradigms": ["online_rl", "offline_rl"]},
    # Methods (Real Benchmark Suites)
    {"id": "sb3-suite", "name": "sb3_suite-1.0.3.zip", "src": ROOT / "methods" / "sb3", "kind": "method", "paradigms": ["online_rl"]},
    {"id": "d3rlpy-suite", "name": "d3rlpy_suite-1.0.2.zip", "src": ROOT / "methods" / "d3rlpy", "kind": "method", "paradigms": ["offline_rl"]},
    # Robotics suite components
    {"id": "robotics-mujoco", "name": "robotics-mujoco-1.0.1.zip", "src": ROOT / "envs" / "mujoco", "kind": "env", "paradigms": ["online_rl"]},
    {"id": "robotics", "name": "robotics-1.0.1.zip", "src": ROOT / "experiments" / "robotics", "kind": "experiment", "paradigms": ["online_rl"]},
    {"id": "robotics_suite", "name": "robotics_suite-1.0.1.zip", "src": ROOT / "suites" / "robotics_suite", "kind": "suite", "paradigms": ["online_rl"]},
    # Meta Plugins & Suites
    {"id": "thetalib", "name": "thetalib-1.0.0.zip", "src": ROOT / "suites" / "thetalib", "kind": "plugin", "paradigms": ["online_rl", "offline_rl", "supervised", "unsupervised"]},
    {"id": "basic_suite", "name": "basic_suite-1.0.0.zip", "src": ROOT / "suites" / "basic_suite", "kind": "suite", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "theta-validation-suite", "name": "theta-validation-suite-1.1.0.zip", "src": ROOT / "suites" / "theta-validation-suite", "kind": "suite", "paradigms": ["online_rl"]},
    {"id": "dl_basics_suite", "name": "dl_basics_suite-1.0.0.zip", "src": ROOT / "suites" / "dl_basics_suite", "kind": "suite", "paradigms": ["supervised", "unsupervised", "online_rl", "offline_rl"]},
    # Experiment Plugins
    {"id": "mimic", "name": "mimic-1.0.1.zip", "src": ROOT / "experiments" / "mimic", "kind": "experiment", "paradigms": ["supervised", "offline_rl"]},
    {"id": "pyrenees", "name": "pyrenees-1.0.0.zip", "src": ROOT / "experiments" / "pyrenees", "kind": "experiment", "paradigms": ["offline_rl"]},
    # Data Modules
    {"id": "image-data", "name": "image_data-1.0.0.zip", "src": ROOT / "data" / "image-data", "kind": "datamodule", "suite": "thetalib", "hidden": True, "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "replay-buffer", "name": "replay_buffer-1.0.0.zip", "src": ROOT / "data" / "replay-buffer", "kind": "datamodule", "suite": "thetalib", "hidden": True, "paradigms": ["offline_rl"]},
    {"id": "tabular-data", "name": "tabular_data-1.0.0.zip", "src": ROOT / "data" / "tabular-data", "kind": "datamodule", "suite": "thetalib", "hidden": True, "paradigms": ["supervised", "offline_rl", "unsupervised"]},
    {"id": "clinical-timeseries-data", "name": "clinical_timeseries_data-1.0.0.zip", "src": ROOT / "data" / "clinical-timeseries-data", "kind": "datamodule", "suite": "mimic", "hidden": True, "paradigms": ["supervised", "offline_rl"]},
    # Evaluators
    {"id": "pyrenees-evaluator", "name": "pyrenees_evaluator-1.0.0.zip", "src": ROOT / "eval" / "pyrenees-evaluator", "kind": "evaluator", "suite": "pyrenees", "hidden": True, "paradigms": ["offline_rl"]},
    {"id": "mimic-early-prediction", "name": "mimic_early_prediction-1.0.1.zip", "src": ROOT / "eval" / "mimic-early-prediction", "kind": "evaluator", "suite": "mimic", "hidden": True, "paradigms": ["supervised", "offline_rl"]},
    {"id": "reward-shaping", "name": "reward_shaping-1.0.0.zip", "src": ROOT / "eval" / "reward-shaping", "kind": "evaluator", "suite": "mimic", "hidden": True, "paradigms": ["offline_rl"]},
    # Plotters / Visualization Suites
    {"id": "standard-figures", "name": "standard_figures-1.0.0.zip", "src": ROOT / "plots" / "standard-figures", "kind": "plotter", "suite": "thetalib", "hidden": True, "paradigms": ["online_rl", "offline_rl", "supervised", "unsupervised"]},
    {"id": "mimic-clinical-suite", "name": "mimic_clinical_suite-1.0.0.zip", "src": ROOT / "plots" / "mimic-clinical-suite", "kind": "plotter", "suite": "mimic", "hidden": True, "paradigms": ["supervised", "offline_rl"]},
    {"id": "pyrenees-benchmark-suite", "name": "pyrenees_benchmark_suite-1.0.0.zip", "src": ROOT / "plots" / "pyrenees-benchmark-suite", "kind": "plotter", "suite": "pyrenees", "hidden": True, "paradigms": ["offline_rl"]},
]


def package_component(src_dir: Path, output_zip: Path) -> str:
    """Compress directory into zip archive and return hex SHA-256."""
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in sorted(src_dir.rglob("*")):
            if (
                file_path.is_file()
                and (not file_path.name.startswith(".") or file_path.name == ".theta_component.json")
                and "__pycache__" not in file_path.parts
                and not file_path.name.startswith("test_")
            ):
                arcname = file_path.relative_to(src_dir)
                zf.write(file_path, arcname=arcname)

    data = output_zip.read_bytes()
    return hashlib.sha256(data).hexdigest()


def update_registry_entry(comp: dict, sha256: str, zip_name: str):
    """Update or create registry metadata entry in theta-hub/registry/."""
    if not HUB_ROOT.exists():
        return

    kind_plural = comp["kind"] + "s" if not comp["kind"].endswith("s") else comp["kind"]
    reg_dir = HUB_ROOT / "registry" / kind_plural
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg_file = reg_dir / f"{comp['id']}.json"

    meta_file = comp["src"] / ".theta_component.json"
    comp_meta = {}
    if meta_file.exists():
        try:
            comp_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not comp_meta:
        plugin_file = comp["src"] / "plugin.json"
        if plugin_file.exists():
            try:
                comp_meta = json.loads(plugin_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    ver = comp_meta.get("version", "1.0.0")
    raw_url = f"{HUB_PACKAGES_URL}/{zip_name}"

    if reg_file.exists():
        try:
            entry = json.loads(reg_file.read_text(encoding="utf-8"))
        except Exception:
            entry = {}
    else:
        entry = {
            "id": comp["id"],
            "name": comp_meta.get("name", comp["id"].replace("_", " ").title()),
            "kind": comp["kind"],
            "version": ver,
            "description": comp_meta.get("description", f"{comp['id']} component for Theta-IDE."),
            "author": comp_meta.get("author", {"name": "Theta-IDE Team"}),
            "repository": comp_meta.get("repository", "https://github.com/CameronEgb/theta-plugins"),
            "license": comp_meta.get("license", "MIT"),
            "tags": [comp["kind"], comp["id"]],
            "releases": {},
        }

    if comp_meta.get("repository"):
        entry["repository"] = comp_meta["repository"]
    if comp_meta.get("license"):
        entry["license"] = comp_meta["license"]
    if comp_meta.get("tags"):
        entry["tags"] = comp_meta["tags"]
    if "test_suite" in comp_meta:
        entry["test_suite"] = comp_meta["test_suite"]

    if "name" in comp_meta:
        entry["name"] = comp_meta["name"]
    if "description" in comp_meta:
        entry["description"] = comp_meta["description"]
    if "config_path" in comp_meta:
        entry["config_path"] = comp_meta["config_path"]
    if "config_paths" in comp_meta:
        entry["config_paths"] = comp_meta["config_paths"]
    if "parameter_schema" in comp_meta:
        entry["parameter_schema"] = comp_meta["parameter_schema"]
    if "target_path" in comp_meta:
        entry["target_path"] = comp_meta["target_path"]
    if "dependencies" in comp_meta:
        entry["dependencies"] = comp_meta["dependencies"]
    if "suite" in comp_meta:
        entry["suite"] = comp_meta["suite"]
    elif "suite" in comp:
        entry["suite"] = comp["suite"]
    if "hidden" in comp_meta:
        entry["hidden"] = comp_meta["hidden"]
    elif "hidden" in comp:
        entry["hidden"] = comp["hidden"]
    if "provides" in comp_meta:
        entry["provides"] = comp_meta["provides"]
    if "installed_dirs" in comp_meta:
        entry["installed_dirs"] = comp_meta["installed_dirs"]

    for field in NATIVE_FIELDS + PLUGIN_FIELDS:
        if field in comp_meta:
            entry[field] = comp_meta[field]

    # Record supported paradigms
    paradigms = comp_meta.get("paradigms") or comp.get("paradigms")
    if paradigms:
        entry["paradigms"] = paradigms

    entry["version"] = ver
    if "releases" not in entry:
        entry["releases"] = {}

    entry["releases"][ver] = {
        "tag": f"v{ver}",
        "url": raw_url,
        "sha256": sha256,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": (HUB_PACKAGES / zip_name).stat().st_size if (HUB_PACKAGES / zip_name).exists() else None,
    }

    reg_file.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build and sync Theta plugin component archives.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(comp["id"] for comp in COMPONENTS),
        help="Package only these component IDs.",
    )
    selected = set(parser.parse_args().only or [])
    DIST.mkdir(parents=True, exist_ok=True)
    if HUB_ROOT.exists():
        HUB_PACKAGES.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Packaging Theta Plugins & Components")
    print("=" * 70)

    for comp in COMPONENTS:
        if selected and comp["id"] not in selected:
            continue
        src = comp["src"]
        if not src.exists():
            print(f"⚠️ Skipping {comp['id']}: directory {src} not found")
            continue

        out = DIST / comp["name"]
        sha = package_component(src, out)
        size_kb = out.stat().st_size / 1024
        print(f"📦 {comp['id']:<20} -> dist/{comp['name']}")
        print(f"   Size:   {size_kb:.1f} KB")
        print(f"   SHA256: {sha}")

        # Sync to theta-hub
        if HUB_ROOT.exists():
            hub_zip = HUB_PACKAGES / comp["name"]
            shutil.copy2(str(out), str(hub_zip))
            update_registry_entry(comp, sha, comp["name"])
            print(f"   Synced to theta-hub: packages/{comp['name']}")
        print()

    print(f"Artifacts created in: {DIST}")


if __name__ == "__main__":
    main()

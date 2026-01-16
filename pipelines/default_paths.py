from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists() or (parent / "README.md").exists():
            return parent
    return start.resolve()


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())

OUTPUT_DIR = "Output"
BASELINE_DIR = "Output"
SIM_DIR = "Output/sim_spectrum"
COMPARE_DIR = "Output/compare_methods"
BASELINE_NPZ = "Output/baseline_artifacts.npz"
BASELINE_META = "Output/baseline_meta.json"

SEED = 2025
SINGLE_BAND = True
DISABLE_PREAMP = True
DEFAULT_N_SAMPLES = 400
DEFAULT_BALANCED = True
SPLIT = (0.6, 0.2, 0.2)


def build_run_snapshot(output_dir: Path, extra: Dict[str, object] | None = None) -> Dict[str, object]:
    snapshot = {
        "project_root": str(PROJECT_ROOT),
        "single_band": SINGLE_BAND,
        "disable_preamp": DISABLE_PREAMP,
        "seed": SEED,
        "output_dir": str(output_dir),
    }
    if extra:
        snapshot.update(extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return snapshot

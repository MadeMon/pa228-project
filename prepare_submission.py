#!/usr/bin/env python3
"""
Prepares a submission archive from SEG_514143.

Steps:
    1. Copies SEG_514143 → SEG_514143 (skipping unwanted entries)
    2. Moves model.pt, learning_curves.png, model_architecture.png into final_files/
    3. Zips the result into SEG_514143.zip
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "SEG_514143"
DST = ROOT / "final_solution" / "SEG_514143"
ZIP_PATH = ROOT / "final_solution" / "SEG_514143"  # shutil.make_archive appends .zip

REMOVE = [
    ".venv",
    "__pycache__",
    "data_seg_public",
    ".env",
    "output_predictions",
    "models",
    ".pdm-python",
    "mlflow.db",
    "model_architecture"
]

MOVES = {
    "model.pt": "final_files/final_model.pt",
    "learning_curves.png": "final_files/final_learning_curve.png",
    "model_architecture.png": "final_files/final_model_architecture.png",
}


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Source directory not found: {SRC}")

    # 1. Copy, skipping unwanted entries
    if DST.exists():
        raise FileExistsError(f"Destination directory already exists: {DST}")
    DST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {SRC} → {DST} ...")
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns(*REMOVE))

    # 2. Move files into final_files/
    final_files_dir = DST / "final_files"
    final_files_dir.mkdir(parents=True, exist_ok=True)

    for src_name, dst_rel in MOVES.items():
        src_file = DST / src_name
        dst_file = DST / dst_rel
        if src_file.exists():
            print(f"  Moving {src_name} → {dst_rel}")
            shutil.move(str(src_file), str(dst_file))
        else:
            print(f"  Skipping (not found): {src_file}")

    # 3. Create zip archive
    print(f"Creating {ZIP_PATH}.zip ...")
    shutil.make_archive(
        base_name=str(ZIP_PATH),
        format="zip",
        root_dir=str(ROOT),
        base_dir=DST.name,
    )
    print("Done.")


if __name__ == "__main__":
    main()

"""
colab_setup.py — UniCirc Colab Environment Setup
═════════════════════════════════════════════════
Run this ONCE at the start of every new Colab session.

What it does:
  1. Installs packages from requirements.txt
  2. Downloads CMU-MOSEI processed features  (public Kaggle dataset)
  3. Downloads MELD dataset                  (public Kaggle dataset)
  4. Verifies everything is in place

Usage:
  !python colab_setup.py
"""

import os
import shutil
import subprocess
from pathlib import Path


# ─── Kaggle dataset identifier ────────────────────────────
# Public dataset — no kaggle.json or login needed
KAGGLE_DATASET = "biswajitroyiitj/unicirc-processed-cmu-mosei-meld-features"


# ═══════════════════════════════════════════════════════════
# SECTION 1 — INSTALL REQUIREMENTS
# Reads requirements.txt and installs all packages via pip.
# Falls back to a hardcoded core list if the file is missing.
# ═══════════════════════════════════════════════════════════

def install_requirements():
    print("=" * 55)
    print("STEP 1 — Installing requirements")
    print("=" * 55)

    req_file = Path("requirements.txt")

    if req_file.exists():
        print("  Found requirements.txt — installing...")
        result = subprocess.run(
            "pip install -q -r requirements.txt",
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ⚠ Some packages may have failed:\n{result.stderr[:400]}")
        else:
            print("  ✓ All packages installed")
    else:
        print("  requirements.txt not found — installing core packages...")
        core = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "transformers",
            "kagglehub",
            "pandas numpy scikit-learn matplotlib seaborn tqdm",
            "openai-whisper librosa mediapipe opencv-python-headless",
        ]
        for pkg in core:
            print(f"  installing {pkg.split()[0]}...")
            subprocess.run(f"pip install -q {pkg}", shell=True, check=False)

    print()


# ═══════════════════════════════════════════════════════════
# SECTION 2 — CREATE DIRECTORY STRUCTURE
# Creates the folder layout that unicirc.py expects:
#
#   circumplex-multimodal-affect/
#   ├── data/
#   │   ├── processed_mosei.pkl
#   │   └── MELD.Raw/MELD.Raw/
#   │       ├── train/
#   │       │   ├── train_sent_emo.csv
#   │       │   └── train_splits/        ← 9,989 mp4 clips
#   │       ├── dev_sent_emo.csv
#   │       ├── dev/
#   │       │   └── dev_splits_complete/ ← 1,112 mp4 clips
#   │       └── test_sent_emo.csv
#   ├── model/
#   │   └── unicirc.py
#   └── colab_setup.py
# ═══════════════════════════════════════════════════════════

def create_directories():
    for d in ["data", "model", "checkpoints", "outputs"]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# SECTION 3 — DOWNLOAD CMU-MOSEI
# Downloads processed_mosei.pkl from the public Kaggle dataset.
# File size: ~2.5 GB — takes ~5 min on Colab T4.
# ═══════════════════════════════════════════════════════════

def download_mosei():
    print("=" * 55)
    print("STEP 2 — CMU-MOSEI processed features")
    print("=" * 55)

    mosei_path = Path("data/processed_mosei.pkl")

    if mosei_path.exists():
        size_mb = mosei_path.stat().st_size / (1024 ** 2)
        print(f"  ✓ Already exists ({size_mb:.0f} MB) — skipping\n")
        return True

    print(f"  Downloading from: kaggle.com/datasets/{KAGGLE_DATASET}")
    print("  This may take ~5 minutes...")

    try:
        import kagglehub
        dl_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"  Download path: {dl_path}")

        # Search for processed_mosei.pkl anywhere in the downloaded folder
        for found in Path(dl_path).rglob("processed_mosei.pkl"):
            shutil.copy2(found, mosei_path)
            size_mb = mosei_path.stat().st_size / (1024 ** 2)
            print(f"  ✓ processed_mosei.pkl ready ({size_mb:.0f} MB)\n")
            return True

        print("  ⚠ processed_mosei.pkl not found in download.")
        print(f"  Files present: {[f.name for f in Path(dl_path).iterdir()]}\n")
        return False

    except Exception as e:
        print(f"  ❌ Download failed: {e}\n")
        return False


# ═══════════════════════════════════════════════════════════
# SECTION 4 — DOWNLOAD MELD
# Downloads MELD CSVs and video clips from the public dataset.
# Total size: ~3 GB (train + dev + test video clips).
# ═══════════════════════════════════════════════════════════

def count_mp4(folder):
    """Returns number of .mp4 files in a folder."""
    p = Path(folder)
    return len(list(p.glob("*.mp4"))) if p.exists() else 0


def download_meld():
    print("=" * 55)
    print("STEP 3 — MELD dataset")
    print("=" * 55)

    train_clips = count_mp4("data/MELD.Raw/MELD.Raw/train/train_splits")
    train_csv   = Path("data/MELD.Raw/MELD.Raw/train/train_sent_emo.csv")

    if train_clips > 9000 and train_csv.exists():
        dev_clips = count_mp4("data/MELD.Raw/MELD.Raw/dev/dev_splits_complete")
        print(f"  ✓ Already downloaded:")
        print(f"    train clips : {train_clips:,}")
        print(f"    dev clips   : {dev_clips:,}\n")
        return True

    print(f"  Downloading from: kaggle.com/datasets/{KAGGLE_DATASET}")
    print("  This may take ~10 minutes (video clips ~3 GB)...")

    try:
        import kagglehub
        dl_path = kagglehub.dataset_download(KAGGLE_DATASET)

        # Search for MELD.Raw folder anywhere in the download
        for meld_src in Path(dl_path).rglob("MELD.Raw"):
            if meld_src.is_dir():
                dst = Path("data/MELD.Raw")
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(meld_src, dst)
                print(f"  ✓ MELD.Raw copied")
                break

    except Exception as e:
        print(f"  ❌ Download failed: {e}\n")
        return False

    # Final counts
    train_clips = count_mp4("data/MELD.Raw/MELD.Raw/train/train_splits")
    dev_clips   = count_mp4("data/MELD.Raw/MELD.Raw/dev/dev_splits_complete")
    test_clips  = count_mp4("data/MELD.Raw/MELD.Raw/test/output_repeated_splits_test")

    print(f"  ✓ MELD ready:")
    print(f"    train clips : {train_clips:,} / 9,989 expected")
    print(f"    dev clips   : {dev_clips:,}  / 1,112 expected")
    print(f"    test clips  : {test_clips:,}  / 2,610 expected\n")
    return True


# ═══════════════════════════════════════════════════════════
# SECTION 5 — VERIFICATION
# Confirms all required files are present before training.
# ═══════════════════════════════════════════════════════════

def verify():
    print("=" * 55)
    print("STEP 4 — Verification")
    print("=" * 55)

    all_ok = True

    # Required files with expected minimum sizes (MB)
    file_checks = {
        "data/processed_mosei.pkl":                             "MOSEI features",
        "data/MELD.Raw/MELD.Raw/train/train_sent_emo.csv":     "MELD train CSV",
        "data/MELD.Raw/MELD.Raw/dev_sent_emo.csv":             "MELD dev CSV",
        "data/MELD.Raw/MELD.Raw/test_sent_emo.csv":            "MELD test CSV",
    }

    for path, label in file_checks.items():
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 ** 2)
            print(f"  ✓ {label:<28} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {label:<28} NOT FOUND at {path}")
            all_ok = False

    # Video clip counts
    clip_checks = {
        "data/MELD.Raw/MELD.Raw/train/train_splits":               ("train clips", 9989),
        "data/MELD.Raw/MELD.Raw/dev/dev_splits_complete":          ("dev clips",   1112),
        "data/MELD.Raw/MELD.Raw/test/output_repeated_splits_test": ("test clips",  2610),
    }

    for folder, (label, expected) in clip_checks.items():
        count  = count_mp4(folder)
        ok     = count >= int(expected * 0.95)   # 5% tolerance
        symbol = "✓" if ok else "⚠"
        print(f"  {symbol} MELD {label:<22} {count:,} / {expected:,}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("✅ Setup complete — ready to train!")
        print("\n  Next step:")
        print("  !python model/unicirc.py")
    else:
        print("⚠  Some files are missing — check errors above.")
        print("\n  Fixes:")
        print("  • MOSEI missing → re-run download_mosei()")
        print("  • MELD missing  → re-run download_meld()")
        print("  • Wrong paths   → !find data/ -name '*.mp4' | head -3")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    create_directories()
    install_requirements()
    download_mosei()
    download_meld()
    verify()

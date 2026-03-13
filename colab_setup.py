"""
colab_setup.py — UniCirc Colab Environment Setup
═════════════════════════════════════════════════
Run this ONCE at the start of every new Colab session.

What it does:
  1. Installs packages from requirements.txt
  2. Sets up Kaggle credentials
  3. Downloads CMU-MOSEI processed features
  4. Downloads MELD dataset (CSVs + video clips)
  5. Verifies everything is in place

Usage:
  !python colab_setup.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


# ═══════════════════════════════════════════════════════════
# SECTION 1 — INSTALL REQUIREMENTS
# Reads requirements.txt and installs all packages via pip.
# ═══════════════════════════════════════════════════════════

def install_requirements():
    print("=" * 55)
    print("STEP 1 — Installing requirements")
    print("=" * 55)

    req_file = Path("requirements.txt")

    if not req_file.exists():
        # requirements.txt not found — install core packages directly
        print("  requirements.txt not found — installing core packages...")
        core = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "transformers",
            "kagglehub",
            "pandas numpy scikit-learn matplotlib seaborn tqdm",
            "openai-whisper librosa mediapipe opencv-python-headless",
        ]
        for pkg in core:
            name = pkg.split()[0]
            print(f"  installing {name}...")
            subprocess.run(f"pip install -q {pkg}", shell=True, check=False)
    else:
        print(f"  Found requirements.txt — installing...")
        result = subprocess.run(
            "pip install -q -r requirements.txt",
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ⚠ Some packages may have failed:\n{result.stderr[:500]}")
        else:
            print("  ✓ All packages installed")

    print()


# ═══════════════════════════════════════════════════════════
# SECTION 2 — KAGGLE CREDENTIALS
# Uploads kaggle.json via Colab file picker if not present.
#
# To get your kaggle.json:
#   1. Go to https://www.kaggle.com/settings
#   2. Scroll to "API" → click "Create New Token"
#   3. kaggle.json downloads to your computer
#   4. Upload it when the file picker appears below
# ═══════════════════════════════════════════════════════════

def setup_kaggle():
    print("=" * 55)
    print("STEP 2 — Kaggle credentials")
    print("=" * 55)

    kaggle_path = Path("/root/.kaggle/kaggle.json")

    if kaggle_path.exists():
        print("  ✓ kaggle.json already present — skipping\n")
        return True

    print("  kaggle.json not found.")
    print("  Opening file picker — select your kaggle.json...")

    try:
        from google.colab import files
        uploaded = files.upload()   # opens Colab file picker

        if not uploaded:
            print("  ❌ No file uploaded — skipping Kaggle setup")
            return False

        kaggle_path.parent.mkdir(parents=True, exist_ok=True)
        for fname, content in uploaded.items():
            with open(kaggle_path, "wb") as f:
                f.write(content)

        os.chmod(kaggle_path, 0o600)   # required by Kaggle API
        print("  ✓ kaggle.json saved\n")
        return True

    except ImportError:
        print("  ⚠ Not running in Colab.")
        print("  Manually place kaggle.json at /root/.kaggle/kaggle.json\n")
        return False




def create_directories():
    dirs = ["data", "model", "checkpoints", "outputs"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# SECTION 4 — DOWNLOAD CMU-MOSEI
# Downloads processed_mosei.pkl from your Kaggle dataset.
# File size: ~2.5 GB — takes ~5 min on Colab T4 GPU runtime.
# ═══════════════════════════════════════════════════════════

KAGGLE_DATASET = "biswajitroyiitj/unicirc-processed-cmu-mosei-meld-features"

def download_mosei():
    print("=" * 55)
    print("STEP 3 — CMU-MOSEI processed features")
    print("=" * 55)

    mosei_path = Path("data/processed_mosei.pkl")

    if mosei_path.exists():
        size_mb = mosei_path.stat().st_size / (1024 ** 2)
        print(f"  ✓ processed_mosei.pkl already exists ({size_mb:.0f} MB) — skipping\n")
        return True

    print(f"  Downloading from Kaggle ({KAGGLE_DATASET})...")
    print("  This may take ~5 minutes...")

    try:
        import kagglehub
        dl_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"  Downloaded to: {dl_path}")

        # Copy processed_mosei.pkl into data/
        for item in Path(dl_path).rglob("processed_mosei.pkl"):
            shutil.copy2(item, mosei_path)
            size_mb = mosei_path.stat().st_size / (1024 ** 2)
            print(f"  ✓ processed_mosei.pkl copied ({size_mb:.0f} MB)\n")
            return True

        print("  ⚠ processed_mosei.pkl not found in downloaded files")
        print(f"  Contents: {list(Path(dl_path).iterdir())}\n")
        return False

    except Exception as e:
        print(f"  ❌ kagglehub failed: {e}")
        print("  Trying kaggle CLI fallback...")

        result = subprocess.run(
            f"kaggle datasets download -d {KAGGLE_DATASET} --path data/ --unzip",
            shell=True, capture_output=True, text=True
        )
        if mosei_path.exists():
            size_mb = mosei_path.stat().st_size / (1024 ** 2)
            print(f"  ✓ processed_mosei.pkl downloaded ({size_mb:.0f} MB)\n")
            return True
        else:
            print(f"  ❌ Download failed: {result.stderr[:300]}\n")
            return False


# ═══════════════════════════════════════════════════════════
# SECTION 5 — DOWNLOAD MELD
# Downloads MELD CSVs + video clips from Kaggle.
# Total size: ~3 GB (train + dev + test video clips)
# ═══════════════════════════════════════════════════════════

def count_mp4(folder):
    """Returns number of .mp4 files in a folder."""
    p = Path(folder)
    return len(list(p.glob("*.mp4"))) if p.exists() else 0


def download_meld():
    print("=" * 55)
    print("STEP 4 — MELD dataset")
    print("=" * 55)

    # Check if already downloaded
    train_clips = count_mp4("data/MELD.Raw/MELD.Raw/train/train_splits")
    dev_clips   = count_mp4("data/MELD.Raw/MELD.Raw/dev/dev_splits_complete")
    train_csv   = Path("data/MELD.Raw/MELD.Raw/train/train_sent_emo.csv")

    if train_clips > 9000 and train_csv.exists():
        print(f"  ✓ MELD already downloaded:")
        print(f"    train clips : {train_clips:,}")
        print(f"    dev clips   : {dev_clips:,}\n")
        return True

    print(f"  Downloading MELD from Kaggle ({KAGGLE_DATASET})...")
    print("  This may take ~10 minutes (video clips ~3 GB)...")

    try:
        import kagglehub
        dl_path = kagglehub.dataset_download(KAGGLE_DATASET)

        # Copy the entire MELD.Raw folder into data/
        for meld_src in Path(dl_path).rglob("MELD.Raw"):
            if meld_src.is_dir():
                dst = Path("data/MELD.Raw")
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(meld_src, dst)
                print(f"  ✓ MELD.Raw copied from {meld_src}")
                break

    except Exception as e:
        print(f"  kagglehub failed: {e} — trying kaggle CLI...")
        result = subprocess.run(
            f"kaggle datasets download -d {KAGGLE_DATASET} --path data/ --unzip",
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ❌ Kaggle CLI also failed: {result.stderr[:300]}\n")
            return False

    # Recount after download
    train_clips = count_mp4("data/MELD.Raw/MELD.Raw/train/train_splits")
    dev_clips   = count_mp4("data/MELD.Raw/MELD.Raw/dev/dev_splits_complete")
    test_clips  = count_mp4("data/MELD.Raw/MELD.Raw/test/output_repeated_splits_test")

    print(f"  ✓ MELD downloaded:")
    print(f"    train clips : {train_clips:,} / 9,989 expected")
    print(f"    dev clips   : {dev_clips:,}   / 1,112 expected")
    print(f"    test clips  : {test_clips:,}  / 2,610 expected\n")
    return True


# ═══════════════════════════════════════════════════════════
# SECTION 6 — VERIFICATION
# Checks all required files exist before training starts.
# ═══════════════════════════════════════════════════════════

def verify():
    print("=" * 55)
    print("STEP 5 — Verification")
    print("=" * 55)

    all_ok = True

    # Required files
    file_checks = {
        "data/processed_mosei.pkl":                              "MOSEI features",
        "data/MELD.Raw/MELD.Raw/train/train_sent_emo.csv":      "MELD train CSV",
        "data/MELD.Raw/MELD.Raw/dev_sent_emo.csv":              "MELD dev CSV",
        "data/MELD.Raw/MELD.Raw/test_sent_emo.csv":             "MELD test CSV",
    }

    for path, label in file_checks.items():
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 ** 2)
            print(f"  ✓ {label:<28} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {label:<28} NOT FOUND")
            all_ok = False

    # Video clip counts
    clip_checks = {
        "data/MELD.Raw/MELD.Raw/train/train_splits":                ("train clips", 9989),
        "data/MELD.Raw/MELD.Raw/dev/dev_splits_complete":           ("dev clips",   1112),
        "data/MELD.Raw/MELD.Raw/test/output_repeated_splits_test":  ("test clips",  2610),
    }

    for folder, (label, expected) in clip_checks.items():
        count  = count_mp4(folder)
        ok     = count >= int(expected * 0.95)   # 5% tolerance
        symbol = "✓" if ok else "⚠"
        print(f"  {symbol} MELD {label:<20} {count:,} / {expected:,}")
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
        print("  • MOSEI missing → re-run Step 3 or upload manually to data/")
        print("  • MELD missing  → re-run Step 4")
        print("  • Wrong clip folder → run: !find data/ -name '*.mp4' | head -3")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    create_directories()
    install_requirements()
    setup_kaggle()
    download_mosei()
    download_meld()
    verify()

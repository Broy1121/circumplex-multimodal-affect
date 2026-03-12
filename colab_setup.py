"""
UniCirc Colab Setup Script

This script prepares the environment for running UniCirc in Google Colab.

Steps:
1. Installs dependencies
2. Configures Kaggle API
3. Downloads the processed CMU-MOSEI dataset
4. Loads the dataset for training
"""

import pickle
import kagglehub
import shutil
import os
import pandas as pd


def install_requirements():
    print("Installing requirements...")
    os.system("pip install -r requirements.txt")


def setup_kaggle():
    print("Setting up Kaggle API...")
    os.system("mkdir -p ~/.kaggle")
    os.system("cp kaggle.json ~/.kaggle/")
    os.system("chmod 600 ~/.kaggle/kaggle.json")


def download_dataset():
    print("Downloading dataset via kagglehub...")

    path = kagglehub.dataset_download(
        "biswajitroyiitj/unicirc-processed-cmu-mosei-meld-features"
    )

    print("Dataset location:", path)

    os.makedirs("data", exist_ok=True)

    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join("data", item)

        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print("Dataset copied into data/")
    print("Contents of data/:")
    for item in os.listdir("data"):
        print(f"  {item}")


def load_mosei():
    print("\n── Loading CMU-MOSEI ──────────────────────")
    path = "data/processed_mosei.pkl"

    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return None

    with open(path, "rb") as f:
        mosei = pickle.load(f)

    print(f"MOSEI loaded — {len(mosei)} videos")
    return mosei


def load_meld():
    print("\n── Loading MELD ───────────────────────────")

    # Possible CSV locations
    paths = {
        "train": [
            "data/MELD.Raw/train/train_sent_emo.csv",
            "data/MELD.Raw/MELD.Raw/train/train_sent_emo.csv",
        ],
        "dev": [
            "data/MELD.Raw/dev_sent_emo.csv",
            "data/MELD.Raw/MELD.Raw/dev_sent_emo.csv",
        ],
        "test": [
            "data/MELD.Raw/test_sent_emo.csv",
            "data/MELD.Raw/MELD.Raw/test_sent_emo.csv",
        ],
    }

    meld = {}
    for split, candidates in paths.items():
        loaded = False
        for p in candidates:
            if os.path.exists(p):
                meld[split] = pd.read_csv(p)
                print(f"MELD {split:5s} loaded — {len(meld[split])} utterances | path: {p}")
                loaded = True
                break
        if not loaded:
            print(f"WARNING: MELD {split} CSV not found. Tried: {candidates}")

    if meld:
        # Show what columns and emotions we have
        if "train" in meld:
            print(f"\nColumns : {meld['train'].columns.tolist()}")
            print(f"Emotions:\n{meld['train']['Emotion'].value_counts().to_string()}")

    return meld


if __name__ == "__main__":
    install_requirements()
    setup_kaggle()
    download_dataset()

    mosei = load_mosei()
    meld  = load_meld()

    # Summary
    print("\n══ Dataset Summary ════════════════════════")
    if mosei:
        print(f"MOSEI  : {len(mosei)} videos loaded ")
    else:
        print("MOSEI  : failed to load")

    if meld:
        total = sum(len(v) for v in meld.values())
        print(f"MELD   : {total} utterances across {len(meld)} splits ")
    else:
        print("MELD   : failed to load")

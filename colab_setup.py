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


def install_requirements():
    print("Installing requirements...")
    os.system("pip install -r requirements.txt")


def setup_kaggle():
    print("Setting up Kaggle API...")
    os.system("mkdir -p ~/.kaggle")
    os.system("cp kaggle.json ~/.kaggle/")
    os.system("chmod 600 ~/.kaggle/kaggle.json")


#download both dataset I have uploaded in Kaggle
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




def load_dataset():
    print("Loading processed_mosei.pkl ...")
    path = "data/processed_mosei.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    print("Dataset loaded successfully!")
    print("Available keys:", data.keys())


if __name__ == "__main__":
    install_requirements()
    setup_kaggle()
    download_dataset()
    load_dataset()


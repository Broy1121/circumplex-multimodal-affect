"""
UniCirc Colab Setup Script

This script prepares the environment for running UniCirc in Google Colab.

Steps:
1. Installs dependencies
2. Configures Kaggle API
3. Downloads the processed CMU-MOSEI dataset
4. Loads the dataset for training
"""

import os
import pickle


def install_requirements():
    print("Installing requirements...")
    os.system("pip install -r requirements.txt")


def setup_kaggle():
    print("Setting up Kaggle API...")
    os.system("mkdir -p ~/.kaggle")
    os.system("cp kaggle.json ~/.kaggle/")
    os.system("chmod 600 ~/.kaggle/kaggle.json")


def download_dataset():
    print("Downloading MOSEI dataset from Kaggle...")
    os.system("kaggle datasets download -d biswajitroy/unicirc-mosei")
    os.system("unzip unicirc-mosei.zip -d data/")


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


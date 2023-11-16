import argparse
from google.colab import drive
import os
import sys
import torch
import torch.optim as optim
import warnings
from operations import *
from data_manager import *
from hyperparameters import *
from model import *
from train import *
from recommender import *

def mount_drive():
    drive.mount('/content/drive')

def collect_scripts():
    sys.path.append(os.path.abspath('/content/drive/MyDrive/Spotify_project/scripts/'))
    warnings.simplefilter("ignore", category=FutureWarning)

def run_spotify_pipeline(cur_dir, num_files):
    reset_data(cur_dir)
    add_songs(cur_dir, num_files)
    create_splits(cur_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spotify Project Pipeline')
    parser.add_argument('--run_pipeline', action='store_true', default = False, help='Run Spotify Pipeline')
    parser.add_argument('--cur_dir', type=str, help='Path to the current directory')
    parser.add_argument('--model_arch', type=str, default='GraphSAGE', help='Model architecture')
    parser.add_argument('--files_to_parse', type=int, default=0, help= 'Number of JSON files to extract')
    args = parser.parse_args()

    #mount_drive()
    collect_scripts()

    cur_dir = args.cur_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.run_pipeline:
        run_spotify_pipeline(cur_dir, args.files_to_parse)

    train(cur_dir, args.model_arch, device, return_model=True)

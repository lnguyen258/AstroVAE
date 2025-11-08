import os
import yaml
import argparse

import torch
from torch.utils.data import Subset, DataLoader

from src.config import TrainConfig, VAE_Config
from src.utils import Galaxies_ML_Dataset, plot_history
from src.model import VanillaVAE
from src.engine import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE model on galaxy images.")

    # Model and config args
    parser.add_argument(
        '--config_path',
        type=str,
        default='./config/train_vae.yaml',
        help="Path to the training configuration YAML file"
    )
    parser.add_argument(
        'checkpoint_path',
        type=str,
        default=None,
        help="Path to restore training"
    )

    # Data args
    parser.add_argument(
        'train_data_dir',
        type=str,
        default='./data/train/galaxies_ml_train.hdf5',
        help="Path to the training data HDF5 file"
    )
    parser.add_argument(
        'val_data_dir',
        type=str,
        default='./data/val/galaxies_ml_val.hdf5',
        help="Path to the validation data HDF5 file"
    )
    parser.add_argument(
        'test_data_dir',
        type=str,
        default='./data/test/galaxies_ml_test.hdf5',
        help="Path to the test data HDF5 file"
    )

    return parser.parse_args()

def main(
    config_path: str,
    checkpoint_path: str,
    train_data_dir: str,
    val_data_dir: str,
    test_data_dir: str
):
    
    # Feed yaml config into config dataclasses to use
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    model_config = VAE_Config.from_dict(config_dict['model'])
    train_config = TrainConfig.from_dict(config_dict['train'])

    # Initialize device & datasets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = Galaxies_ML_Dataset(train_data_dir)
    val_dataset = Galaxies_ML_Dataset(val_data_dir)
    test_dataset = Galaxies_ML_Dataset(test_data_dir)

    # Initialize model & trainer
    model = VanillaVAE(config=model_config).to(device)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        config=train_config,
    )

    # Start / resume training if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            trainer.load_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    # Train the model
    history, model = trainer.train()

    # Save training history
    output_path = os.path.join(trainer.outputs_dir, f"{trainer.run_name}.png") if train_config.save_fig else None
    plot_history(history, save_fig=output_path)

if __name__ == '__main__':

    args = parse_args()
    main(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        test_data_dir=args.test_data_dir
    )


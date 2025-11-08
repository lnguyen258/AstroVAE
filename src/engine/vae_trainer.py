import os
import re
import csv
from typing import List, Tuple, Dict, Callable, Optional, Union

import numpy as np
from tqdm.auto import tqdm
from rich.progress import (Progress, TextColumn, BarColumn, 
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.console import Console

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ..config import TrainConfig
from ..loss import LOSS_REGISTRY
from ..optim import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from ..utils import (
    CALLBACK_REGISTRY,
    get_loss_from_config,
    get_optim_from_config,
    get_scheduler_from_config,
    get_callbacks_from_config,
)


class Trainer:
    """
    Base class for training models.
    
    Parameters
    ----------
    model: nn.Module
        The model to train.
    train_dataset: Dataset
        The dataset to use for training.
    val_dataset: Dataset
        The dataset to use for validation.
    test_dataset: Dataset, optional
        The dataset to use for testing.
    device: torch.device or int, optional
        Device to run the training on. Overrides config if provided.
    metric: Callable, optional
        A function to compute a metric for evaluation.
    config: TrainConfig, optional
        Configuration object containing training parameters.
    batch_size: int, optional
        Batch size for training. Overrides config if provided.
    criterion: Dict, optional
        Loss function configuration. Overrides config if provided.
    optimizer: Dict, optional
        Optimizer configuration. Overrides config if provided.
    scheduler: Dict, optional
        Learning rate scheduler configuration. Overrides config if provided.
    callbacks: List[Dict], optional
        A list of callbacks to execute during training. Overrides config if provided.
    num_epochs: int, optional
        Number of epochs to train for. Overrides config if provided.
    start_epoch: int, optional
        Epoch to start training from. Overrides config if provided.
    history: Dict[str, List[float]], optional
        History of training metrics. If not provided, initializes an empty history.
    logging_dir: str, optional
        Directory to save logs. Overrides config if provided.
    logging_steps: int, optional
        Frequency of logging during training. Overrides config if provided.
    progress_bar: bool, optional
        Whether to display a tqdm progress bar. Useful to disable on HPC.
    save_best: bool, optional
        Whether to save the best model based on validation loss. Overrides config if provided.
    save_ckpt: bool, optional
        Whether to save checkpoints during training. Overrides config if provided.
    save_fig: bool, optional
        Whether to save evaluation figures. Overrides config if provided.
    num_workers: int, optional
        Number of workers for data loading. Overrides config if provided.
    pin_memory: bool, optional
        Whether to use pinned memory for data loading. Overrides config if provided.
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        device: Optional[Union[torch.device, int]] = None,
        metric: Optional[Callable] = None,
        config: Optional[TrainConfig] = None,
        # Parameters below can override config if supplied explicitly
        batch_size: Optional[int] = None,
        criterion: Optional[Dict] = None,
        optimizer: Optional[Dict] = None,
        scheduler: Optional[Dict] = None,
        callbacks: Optional[List[Dict]] = None,
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = None,
        history: Optional[Dict[str, List[float]]] = None,
        logging_dir: Optional[str] = None,
        logging_steps: Optional[int] = None,
        progress_bar: Optional[bool] = None,
        save_best: Optional[bool] = None,
        save_ckpt: Optional[bool] = None,
        save_fig: Optional[bool] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None
    ):

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        # Use config if provided, otherwise use defaults
        if config is not None:
            self.batch_size = batch_size if batch_size is not None else config.batch_size
            self.criterion = get_loss_from_config(criterion if criterion is not None else config.criterion, LOSS_REGISTRY)
            self.optimizer = get_optim_from_config(optimizer if optimizer is not None else config.optimizer, OPTIM_REGISTRY, self.model)
            self.scheduler = get_scheduler_from_config(scheduler if scheduler is not None else config.scheduler, SCHEDULER_REGISTRY, self.optimizer)
            self.callbacks = get_callbacks_from_config(callbacks if callbacks is not None else config.callbacks, CALLBACK_REGISTRY)
            self.num_epochs = num_epochs if num_epochs is not None else config.num_epochs
            self.start_epoch = start_epoch if start_epoch is not None else config.start_epoch
            self.logging_dir = logging_dir if logging_dir is not None else config.logging_dir
            self.logging_steps = logging_steps if logging_steps is not None else config.logging_steps
            self.progress_bar = progress_bar if progress_bar is not None else config.progress_bar
            self.save_best = save_best if save_best is not None else config.save_best
            self.save_ckpt = save_ckpt if save_ckpt is not None else config.save_ckpt
            self.save_fig = save_fig if save_fig is not None else config.save_fig
            self.num_workers = num_workers if num_workers is not None else config.num_workers
            self.pin_memory = pin_memory if pin_memory is not None else config.pin_memory
        else:
            self.batch_size = batch_size if batch_size is not None else 64
            if criterion is None: raise ValueError("Criterion must be provided if config is not supplied.")
            self.criterion = get_loss_from_config(criterion, LOSS_REGISTRY)
            if optimizer is None: raise ValueError("Optimizer must be provided if config is not supplied.")
            self.optimizer = get_optim_from_config(optimizer, OPTIM_REGISTRY, self.model)
            self.scheduler = get_scheduler_from_config(scheduler, SCHEDULER_REGISTRY, self.optimizer) if scheduler else None
            self.callbacks = get_callbacks_from_config(callbacks, CALLBACK_REGISTRY) if callbacks is not None else None
            self.num_epochs = num_epochs if num_epochs is not None else 20
            self.start_epoch = start_epoch if start_epoch is not None else 0
            self.logging_dir = logging_dir if logging_dir is not None else 'log'
            self.logging_steps = logging_steps if logging_steps is not None else 25
            self.progress_bar = progress_bar if progress_bar is not None else True
            self.save_best = save_best if save_best is not None else True
            self.save_ckpt = save_ckpt if save_ckpt is not None else True
            self.save_fig = save_fig if save_fig is not None else False
            self.num_workers = num_workers if num_workers is not None else 0
            self.pin_memory = pin_memory if pin_memory is not None else False
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        ) if test_dataset is not None else None

        # Initialize metrics and history
        self.metric = metric
        self.history = history or {
            'epoch': [],
            'train_loss': [],
            'train_metric': [],
            'val_loss': [],
            'val_metric': []
        }
        self.best_val_loss = min(self.history['val_loss']) if self.history['val_loss'] else float('inf')

        # Initialize the logging directory
        self.model_name = self.model.__class__.__name__
            
        os.makedirs(self.logging_dir, exist_ok=True)
        self.log_dir = os.path.join(self.logging_dir, self.model_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Subfolders
        self.best_models_dir = os.path.join(self.log_dir, 'best')
        self.checkpoints_dir = os.path.join(self.log_dir, 'checkpoints')
        self.loggings_dir = os.path.join(self.log_dir, 'logging')
        self.outputs_dir = os.path.join(self.log_dir, 'output')
        os.makedirs(self.best_models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.loggings_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Determine run index
        run_index = self._get_next_run_index(self.loggings_dir, 'run', '.csv')
        self.run_name = f"run_{run_index:02d}"

        # Logging and best model paths
        self._log_header_written = False
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.run_name}.pt") if self.save_ckpt else None
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")

    def _get_next_run_index(self, directory: str, prefix: str, suffix: str) -> int:
        os.makedirs(directory, exist_ok=True)
        existing = [
            f for f in os.listdir(directory)
            if f.startswith(prefix) and f.endswith(suffix)
        ]
        indices = [
            int(m.group(1)) for f in existing
            if (m := re.search(rf"{prefix}_(\d+)", f))
        ]

        return max(indices, default=0) + 1
    
    def _set_logging_paths(self, run_name: str):
        self.run_name = run_name
        self._log_header_written = True
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.run_name}.pt") if self.save_ckpt else None
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")

    def save_checkpoint(self, epoch: int):
        if self.checkpoint_path:
            checkpoint = {
                'run_name': self.run_name,
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'history': self.history
            }
            torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._set_logging_paths(checkpoint['run_name'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch']
        self.history = checkpoint['history']
    
    def load_best_model(self, best_model_path: str):
        run_name = os.path.splitext(os.path.basename(best_model_path))[0]
        self._set_logging_paths(run_name)
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

    def log_csv(self, log_dict: Dict[str, float]):
        write_header = not self._log_header_written
        with open(self.logging_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
            if write_header:
                writer.writeheader()
                self._log_header_written = True

            writer.writerow(log_dict)

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        try:
            # Callback before training  
            for cb in self.callbacks:
                cb.on_train_begin(trainer=self)

            total_steps = self.num_epochs * len(self.train_loader)
            start_step = self.start_epoch * len(self.train_loader)

            if self.progress_bar:
                console = Console()
                progress = Progress(
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    BarColumn(),
                    TextColumn("Epoch {task.fields[epoch]}/{task.fields[total_epoch]}"),
                    TextColumn("Step {task.fields[step]}/{task.fields[total_steps]}"),
                    TextColumn("Loss: {task.fields[avg_loss]:.4f}"),
                    TextColumn("Metric: {task.fields[avg_metric]:.4f}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=True,
                    refresh_per_second=5,
                )
                progress.start()
                task = progress.add_task(
                    "Training", 
                    total=total_steps, 
                    completed=start_step,
                    epoch=self.start_epoch + 1,
                    total_epoch=self.num_epochs,
                    step=0, 
                    total_steps=total_steps,
                    avg_loss=0.0, 
                    avg_metric=0.0
                )
            else:
                class _NoOpBar:
                    def advance(self, *args, **kwargs): pass
                    def update(self, *args, **kwargs): pass
                    def stop(self): pass
                    def update_task(self, *args, **kwargs): pass
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                progress = _NoOpBar()
                task = None
            
            for epoch in range(self.start_epoch, self.num_epochs):
                for cb in self.callbacks:
                    cb.on_epoch_begin(trainer=self, epoch=epoch)

                self.model.train()
                running_loss_sum = 0.0
                running_metric_sum = 0.0
                running_count = 0

                for batch_idx, (X,_) in enumerate(self.train_loader):
                    step = epoch * len(self.train_loader) + batch_idx + 1
                    X = X.to(self.device, non_blocking=self.pin_memory)
                    self.optimizer.zero_grad()
                    outputs = self.model(X)
                    loss_dict = self.criterion(*outputs)
                    loss = loss_dict['Loss']
                    loss.backward()
                    self.optimizer.step()
                    bsz = X.size(0)
                    running_loss_sum += float(loss.item()) * bsz

                    if self.metric:
                        running_metric_sum += float(self.metric(outputs[0], X)) * bsz

                    running_count += bsz

                    avg_loss = running_loss_sum / max(running_count, 1)
                    avg_metric = running_metric_sum / max(running_count, 1)

                    # Short summary

                    if task is not None:
                        progress.update(
                        task,
                        advance=1,
                        epoch=epoch,
                        step=step,
                        avg_loss=avg_loss,
                        avg_metric=avg_metric,
                        )

                        if step % self.logging_steps == 0 or step == total_steps:
                            message = (
                                f"step: {step}/{total_steps} | "
                                f"train_loss: {avg_loss:.4f} | "
                                f"train_metric: {avg_metric:.4f}"
                            )
                            console.log(message)
                    
                # Validation phase
                self.model.eval()
                val_loss_sum = 0.0 
                val_metric_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    for X_val, _ in self.val_loader:
                        X_val = X_val.to(self.device, non_blocking=self.pin_memory)
                        outputs_val = self.model(X_val)
                        loss_dict_val = self.criterion(*outputs_val)
                        loss_val = loss_dict_val['Loss']
                        bsz = X_val.size(0)
                        val_loss_sum += float(loss_val.item()) * bsz

                        if self.metric:
                            val_metric_sum += float(self.metric(outputs_val[0], X_val)) * bsz

                        val_count += bsz
                
                total_loss_sum = val_loss_sum
                total_metric_sum = val_metric_sum
                total_count = val_count

                # Global average
                val_loss = total_loss_sum / max(total_count, 1)
                val_metric = (total_metric_sum / max(total_count, 1)) if self.metric else 0.0

                # Short summary for validation
                tqdm.write(
                    f"epoch: {epoch + 1}/{self.num_epochs} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"val_metric: {val_metric:.4f}"
                )

                if self.scheduler:
                    self.scheduler.step()
                
                # Get learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Save best model
                if self.best_model_path and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.best_model_path)
                
                # Update history
                self.history['epoch'].append(epoch + 1)
                self.history['train_loss'].append(avg_loss)
                self.history['train_metric'].append(avg_metric)
                self.history['val_loss'].append(val_loss)
                self.history['val_metric'].append(val_metric)

                # Save a checkpoint every epoch
                self.save_checkpoint(epoch)

                # Log results
                logs = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'train_metric': avg_metric,
                    'val_loss': val_loss,
                    'val_metric': val_metric,
                    'learning_rate': current_lr,
                }
                self.log_csv(logs)

                # Callback after each epoch
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch, trainer=self, logs=logs)

                # Break if any callback says to stop
                if any(getattr(cb, 'should_stop', False) for cb in self.callbacks):
                    break

            # Callback after training
            for cb in self.callbacks:
                cb.on_train_end(trainer=self)

        except KeyboardInterrupt:
            print(f"\nTraining interrupted at epoch {epoch + 1}.")

        return self.history, self.model
    
    @torch.no_grad()
    def evaluate(self):
        if self.test_loader is None:
            raise ValueError("Test dataset not provided.")

        self.model.eval()
        test_loss_sum = 0.0
        test_metric_sum = 0.0
        test_count = 0

        with torch.no_grad():
            for X_test, _ in self.test_loader:
                X_test = X_test.to(self.device, non_blocking=self.pin_memory)
                outputs_test = self.model(X_test)
                loss_dict_test = self.criterion(*outputs_test)
                loss_test = loss_dict_test['Loss']
                bsz = X_test.size(0)
                test_loss_sum += float(loss_test.item()) * bsz

                if self.metric:
                    test_metric_sum += float(self.metric(outputs_test[0], X_test)) * bsz

                test_count += bsz

        test_loss = test_loss_sum / max(test_count, 1)
        test_metric = (test_metric_sum / max(test_count, 1)) if self.metric else 0.0

        print(f"Test Loss: {test_loss:.4f}")
        if self.metric:
            print(f"Test Metric: {test_metric:.4f}")

        return {
            'test_loss': test_loss,
            'test_metric': test_metric
        }

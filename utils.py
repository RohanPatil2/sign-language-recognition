from __future__ import annotations
import torch
import random
import os
import numpy as np
import json
from pathlib import Path
from typing import Any, Optional, Dict, Union


def seed_everything(seed: int) -> None:
    """Set seeds for reproducibility across various libraries."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file from path."""
    with open(path, "r") as f:
        return json.load(f)


def load_label_map(dataset: str) -> Dict[str, Any]:
    """Load label map for specified dataset."""
    file_path = Path("label_maps") / f"label_map_{dataset}.json"
    return load_json(file_path)


def get_experiment_name(args: Any) -> str:
    """Generate experiment name from configuration arguments."""
    parts = []
    if getattr(args, "use_cnn", False):
        parts.append("cnn")
    if getattr(args, "use_augs", False):
        parts.append("augs")
    parts.append(args.model)
    return "_".join(parts)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self) -> None:
        self.reset()
        
    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stops training if validation score doesn't improve after a given patience."""
    
    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        delta: float = 0.0,
        verbose: bool = False
    ) -> None:
        """
        Args:
            patience: Number of checks with no improvement
            mode: One of ['min', 'max'] for minimizing or maximizing metrics
            delta: Minimum change to qualify as improvement
            verbose: Whether to print status messages
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_score = np.Inf if mode == "min" else -np.Inf

        if mode not in ["min", "max"]:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'min' or 'max'.")

    def __call__(
        self,
        model_path: Union[str, Path],
        epoch_score: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        if self.mode == "min":
            score = -epoch_score
        else:
            score = epoch_score

        if self.best_score is None:
            self._save_checkpoint(epoch_score, model, optimizer, scheduler, model_path)
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self._save_checkpoint(epoch_score, model, optimizer, scheduler, model_path)
            self.best_score = score
            self.counter = 0

    def _save_checkpoint(
        self,
        epoch_score: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        model_path: Union[str, Path]
    ) -> None:
        """Saves model when validation score improves."""
        if np.isfinite(epoch_score):
            if self.verbose:
                print(f"Validation score improved ({self.val_score:.4f} --> {epoch_score:.4f}). Saving model...")
            self.val_score = epoch_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "score": epoch_score,
                },
                model_path,
            )
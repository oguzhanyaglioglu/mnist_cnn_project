from dataclasses import dataclass
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

@dataclass
class Config:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = str(SRC_DIR / "data")
    outputs_dir: str = str(OUTPUTS_DIR)
    ckpt_path: str = str(OUTPUTS_DIR / "best_mnist_cnn.pt")
    history_path: str = str(OUTPUTS_DIR / "training_history.json")




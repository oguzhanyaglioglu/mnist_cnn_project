from dataclasses import dataclass
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
OUTPUTS_ROOT = BASE_DIR / "outputs"


# class Config:
#     def __init__(self, seed=42, batch_size=64, lr=1e-3):
#         self.seed = seed
#         self.batch_size = batch_size
#         self.lr = lr
#         ..

# @ -> decorator, bir fonksiyon ya da sınıfın davranışını biraz değiştiren geliştiren ara.
# dataclass; ayar, bilgi, birkaç değeri bir arada toplama gibi veri tutan sınıfları daha kısa ve temiz yazmak için kullanılan "decorator"'dür
@dataclass
class Config:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = str(SRC_DIR / "data")

    # outputs_dir: str = str(OUTPUTS_DIR)
    # ckpt_path: str = str(OUTPUTS_DIR / "best_mnist_cnn.pt")
    # history_path: str = str(OUTPUTS_DIR / "training_history.json")

    outputs_root: str = str(OUTPUTS_ROOT)
    run_name: str = "baseline"

    # property bir fonksiyonu, dışarıdan bakınca(çağırınca), normal değişken/özellik gibi kullanmanı sağlayan dekoratördür.
    # cfg.output_dir() -> cfg.ouput_dir
    @property
    def outputs_dir(self) -> str:
        return str(Path(self.outputs_root) / self.run_name)

    @property
    def ckpt_path(self) -> str:
        return str(Path(self.outputs_dir) / "best_mnist_cnn.pt")

    @property
    def history_path(self) -> str:
        return str(Path(self.outputs_dir) / "training_history.json")




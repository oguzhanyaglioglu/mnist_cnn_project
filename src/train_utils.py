import torch
from torch import nn
from torch.utils.data import DataLoader
from config import Config
from model import build_model
from utils import save_checkpoint, save_json
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def train_one_epoch(cfg: Config, model: nn.Module, train_loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x,y in train_loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)

        logits = model(x)
        loss = criterion(logits, y)

        # onceki iterasyondan kalan gradientler sifirlanir
        optimizer.zero_grad()
        # backpropagation yapılır ve loss'a gore gradientler hesaplanir.
        loss.backward()
        # hesaplanan gradientler kullanilarak model parametreleri optimizer tarafindan guncellenir
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds==y).sum().item()
        total_seen += y.size(0)
        total_loss += loss.item() * y.size(0) # mean loss * batch_size

    avg_loss = total_loss / total_seen
    avg_acc = total_correct / total_seen
    return avg_loss, avg_acc

def eval_one_epoch(cfg: Config, model: nn.Module, test_loader: DataLoader,
                   criterion: nn.Module) -> tuple[float, float]:

    model.eval()# eval -> evaluation(değerlendirme), test ve validation için kullanılan mod

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            logits = model(x)
            loss = criterion(logits, y)

            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_seen += y.size(0)
            total_loss += loss.item() * y.size(0)

    avg_loss = total_loss / total_seen
    avg_acc = total_correct / total_seen
    return  avg_loss, avg_acc

def run_training(cfg: Config, train_loader: DataLoader, test_loader: DataLoader) -> dict:

    model = build_model(cfg)
    # model = SimpleCNN2().to(cfg.device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(cfg, optimizer,)

    best_test_acc = 0.0
    best_test_loss = float("inf") # küçük loss aradığımız için başlangıç lossunu "sonsuz" ayarlıyoruz
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": []
    }

    for epoch in range(1, cfg.epochs + 1):# 1, 3 + 1 = 4, range -> 1, 2, 3
        train_loss, train_acc = train_one_epoch(cfg, model, train_loader, criterion, optimizer)
        test_loss, test_acc = eval_one_epoch(cfg, model, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        current_lr = optimizer.param_groups[0]["lr"] # epoch içinde kullanılan lr
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "# 02d -> 2 basamaklı göster -> 01, 02, 03
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "# .4f - > virgülden sonra 4 hane = 0.123456 -> 0.1234
            f"test_loss={test_loss:.4f} acc={test_acc:.4f} | "
            f"current_lr: {current_lr:.6f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # torch.save(model.state_dict(), "best_mnist_cnn.pt")
            save_checkpoint(model, cfg.ckpt_path)
        else:
            patience_counter +=1

        if patience_counter >= cfg.early_stopping_patience:
            print(f"Early stopping triggered at epoch: {epoch}")
            break

        if scheduler is not None:
            if cfg.scheduler_name is not None and cfg.scheduler_name.lower() == "plateau":
                scheduler.step(test_loss)
            else:
                scheduler.step()
                # lr doğru şekilde, düşmesi gereken yerde düştü mü kontrolu
                # scheduler.get_last_lr()[0] # step() sonrası yeni lr, sonraki epoch'da kullanılacak lr

        # lr_next = optimizer.param_groups[0]["lr"]
        # print(f"[scheduler debug] current: {current_lr:.6f} | next: {lr_next}:.6f")


    print("Best test acc: ", best_test_acc)
    print("Best test loss: ", best_test_loss)
    save_json(history, cfg.history_path)
    return history


def build_scheduler(cfg, optimizer):
    if cfg.scheduler_name is None:
        return None

    if cfg.scheduler_name.lower() == "step":
        return StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma
        )

    if cfg.scheduler_name.lower() == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            # izlediğim metrik küçülürse daha iyi, loss'un küçülmesi daha iyi, örneğin izlediğim metrik accurucy olsa mode="max" olcaktı, çünkü accurucy'nin büyümesi daha iyidir.
            mode="min",
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
        )

    raise ValueError(
        f"Unknown scheduler_name: {cfg.scheduler_name}. "
        f"Choose from ['step', 'plateau']"
    )


import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from config import Config
from data_utils import build_dataloaders
from utils import load_model


# Inferance fonksiyonu -> öğrendikten sonra tahmin yapma aşaması
def predict_one_batch(cfg: Config, ckpt_path: str | None = None) -> None:
    if ckpt_path  is None:
        ckpt_path = cfg.ckpt_path

    _, test_loader = build_dataloaders(cfg)
    model = load_model(cfg, ckpt_path)

    x, y = next(iter(test_loader))
    x = x.to(cfg.device)
    y = y.to(cfg.device)

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)

    print("\n[predict_one_batch]")
    print("true labels: ", y[:10].tolist())
    print("pred labels: ",   preds[:10].tolist())

def show_misclassified_images(cfg: Config, save_dir: str, ckpt_path: str | None = None, max_show: int = 9) -> None:
    if ckpt_path is None:
        ckpt_path = cfg.ckpt_path

    os.makedirs(save_dir, exist_ok=True)

    _, test_loader = build_dataloaders(cfg)

    # model = SimpleCNN2().to(cfg.device)
    # model = build_model(cfg)
    model = load_model(cfg, ckpt_path)
    model.eval()

    images = []
    trues = []
    preds_list = []

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            wrong_mask = preds != y
            if wrong_mask.any(): # any() -> batch'de hiç yanlış yoksa boşuna işlem yapma
                idxs = wrong_mask.nonzero(as_tuple = False).squeeze(1).tolist()
                for idx in idxs:
                    images.append(x[idx].detach().cpu())
                    trues.append(y[idx].item())
                    preds_list.append(preds[idx].item())
                    if len(images) >= max_show:
                        break
            if len(images) >= max_show:
                break

    n = len(images) # max_show = 9 olduğundan en fazla 9 resim olabilir
    cols = 3 # her satırda 3 resim olacak
    # elimde resimler 3'er gruplar haline kaç satırda göstermeliyim, örn: 9 resim 3 sutun ise 3 satır, 7 resim 3 sutun  ise 3 satır, son satırda 1 resim
    rows = (n + cols - 1) // cols # // -> mutlak bölme, küsüratı yukarıya yuvarlar

    plt.figure(figsize=(cols * 3, rows * 3)) # yeni bir sayfa/pencere açar
    for i in range(n):
        plt.subplot(rows, cols, i + 1) # sayfayı row x cols hücreye ayırır, i + 1 -> hangi hücre çizeceğine karar verir
        # 28 * 28 matrise sahip image'ı pixel pixel belirlenen hücreye çizer
        plt.imshow(images[i].squeeze(0), cmap="gray") # image -> [1, 28, 28] -> squeeze(0) -> image -> [28, 28]
        plt.title(f"true={trues[i]} pred={preds_list[i]}") # o hücredeki resmin üstüne başlık yazar(true, preds)
        plt.axis("off") # resmin etrafındaki x y eksen çizgilerini kaldırır
    plt.tight_layout() # kutuların aralığını ayarlar, başlıklar üst üste binmesin diye boşlukları gösteri
    plt.savefig(os.path.join(save_dir, "misclassified_images.png"))
    plt.show() # yaratılan sayfayı ekrana basar


def build_confusion_matrix(cfg: Config, ckpt_path: str | None = None) -> torch.Tensor:
    if ckpt_path is None:
        ckpt_path = cfg.ckpt_path

    _, test_loader = build_dataloaders(cfg)

    # model = SimpleCNN2().to(cfg.device)
    # model = build_model(cfg)
    # model.eval()
    model = load_model(cfg, ckpt_path)

    # confision matris için 10x10'luk matris oluşruldu(10 sınıf olduğu için 0-9), satır doğru etiket, sutun tahmin
    cm = torch.zeros(10,10, dtype=torch.int64)

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            # zip(), listelerini eleman bazında eşler,
            # örn: y = tensor([7, 2, ...]), preds = tensor([7, 8, ...]) -> zip(y, preds) -> (7,7), (2,8), ...
            for true_label, pred_label in zip(y.view(-1), preds.view(-1)):
                cm[true_label.item(), pred_label.item()] += 1 # matriste eşleşen elemana çeltik atar
    return cm

def get_top_confusions(cm: np.ndarray, top_k: int = 3):
    cm_copy = cm.clone()
    cm_copy.fill_diagonal_(0) # bizim yanlış tahminler lazım, bu yüzden ortadaki diaganol doğruları sıfırlıyoruz

    pairs = []
    for i in range(cm_copy.shape[0]): # satırlar, true labels
        for j in range(cm_copy.shape[1]): # sutunlar, predictid labels
            if cm_copy[i, j] > 0: # eğer etiket ve tahminlerin kesişimi sıfırdan büyükse, yani yanlış tahmin varsa pairs içine ekle
                pairs.append((i, j, int(cm_copy[i, j]))) # i satır, j sutun, cm_copy[i, j] o hücredeki sayı

    # pairs'ın üçüncü elemanı yani yanlış çift sayısına göre (x[2]), büyükten küçüğe göre (reverse=True) sırala
    # örn. (i, j, count) -> (2, 7, 6) -> model 2 rakamını 6 defa 7 zannetmiş, x[0] = 2 (true), x[1] = 7 (predict),   x[2] = 6
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def show_confusion_matrix(cfg: Config, save_dir: str, ckpt_path: str | None = None, ) -> None:
    if ckpt_path is None:
        ckpt_path = cfg.ckpt_path

    os.makedirs(save_dir, exist_ok=True)
    cm = build_confusion_matrix(cfg, ckpt_path)

    print("\n[]confusion_matrix")
    print(cm)

    plt.figure(figsize=(8,8))
    plt.imshow(cm.numpy(), cmap="Blues") # matplot numpy ister
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()

    for i in range(10): # satırları dolaşır
        for j in range(10): # sutunları dolaşır
            plt.text(j, i, str(cm[i, j].item()), ha="center", va="center") # i. satır, j. sutuna, confusion matristeki sayıyı yaz

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()


def plot_history(history: dict, save_dir: str) -> None:

    os.makedirs(save_dir, exist_ok=True)

    epochs =  range(1, len(history["train_loss"]) + 1) # 3 epoch olduğu için (1, 4) epoch range'i oluşturduk, bunu x değerleri olarak kullacağım

    plt.figure(figsize=(6, 4))
    # x eksenindeki epoch'a karşılık gelen train loss değeri, bu kesişim "o" olarak işaretlenecek, bu 3 değerden eğri çizilecek, eğrinin etiketi train_loss olacak
    plt.plot(epochs, history["train_loss"], marker="o", label="train_loss")
    plt.plot(epochs, history["test_loss"], marker="o", label="test_loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(list(epochs)) # arg olan epochs parametresini grafiğe yazabilmek için listeye çevirdi
    plt.legend() # grafikte hangi eğrinin ne olduğunu gösteren küçük açıklama kutusu oluşturur
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_acc"], marker="s", label="train_acc")
    plt.plot(epochs, history["test_acc"], marker="s", label="test_acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(list(epochs))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.show()

def plot_lr_curve(history: dict, save_dir: str) -> None:

    lr_values = history.get("lr")

    if lr_values is None or len(lr_values) == 0:
        print("[plot_lr_curve] No learning rate data found")
        return

    # set() -> aynı olan değerleri tekelleştirir, örn; lr_values = [0.001, 0.001, 0.0001, 0.0001] -> set(lr_values) -> {0.001, 0.0001}
    if len(set(lr_values)) == 1: # eğer true ise, o liste sadece tek bir elemandan oluşuyor -> aynı eleman tekrar ediyor
        print("[plot_lr_curve] Learning rate did not change during this run.")
        return

    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(lr_values) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, lr_values, marker="o", label="lr")
    plt.title("Learning Rate Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.xticks(list(epochs))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lr_curve.png"))
    plt.show()

def save_final_summary_json(best_experiment: dict, cm, save_path: str) -> None:

    top_confusions = get_top_confusions(cm, top_k=5)
    summary = {
        "run_name": best_experiment["run_name"],
        "best_test_acc": best_experiment["best_test_acc"],
        "best_test_loss": best_experiment["best_test_loss"],
        "config": {
            "lr": best_experiment["lr"],
            "batch_size": best_experiment["batch_size"],
            "hidden_dim": best_experiment.get("hidden_dim"),
            "dropout_rate": best_experiment.get("dropout_rate"),
            "weight_decay": best_experiment.get("weight_decay"),
            "scheduler_name": best_experiment.get("scheduler_name"),
            "scheduler_config": best_experiment.get("scheduler_config")
        },
        "top_confusions": [
            {
            "true_label": true_label,
            "pred_label": pred_label,
            "count": count,
            }
            for true_label, pred_label, count in top_confusions
        ],
        "comment": [
            "Best model uses hidden_dim=128, dropout=0.1, a plateau scheduler, and no weight decay.",
            "Most errors occur between visually similar handwritten digits."
        ],
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


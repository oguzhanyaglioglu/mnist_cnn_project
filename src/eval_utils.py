import torch
import os
import matplotlib.pyplot as plt
from config import Config
from data_utils import build_dataloaders
from utils import load_model
from model import build_model


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

    model = build_model(cfg)
    # model = SimpleCNN2().to(cfg.device)
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

    model = build_model(cfg)
    # model = SimpleCNN2().to(cfg.device)
    model = load_model(cfg, ckpt_path)
    model.eval()

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
    # x eksenindeki epoch'a karşılık gelen train loss değeri, bu kesişim "o" olarak işaretlenecek, bu 3 değerden eğri çizilecek, etkiteki train_loss olacak
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
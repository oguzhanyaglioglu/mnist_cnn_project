import torch
from torch import nn
from config import Config
from data_utils import build_dataloaders
from model import SimpleCNN, SimpleCNN2, build_model
from utils import set_seed, load_model

def debug_config(cfg: Config) -> None:
    print("\n[debug_config]")
    print(cfg)

def debug_seed(cfg: Config) -> None:
    print("\n[debug_seed]")

    set_seed(cfg.seed)
    a1 = torch.rand(3)

    set_seed(cfg.seed)
    a2 = torch.rand(3)

    print("a1:", a1.tolist())
    print("a2:", a2.tolist())
    print("same?", torch.allclose(a1, a2))

def debug_loss_one_batch(cfg: Config) -> None:
    train_loader,_ = build_dataloaders(cfg)
    x, y = next(iter(train_loader)) # x: [64, 1, 28, 28], y: [64] -> 64 görüntünün(batch) etiket değeri

    model = SimpleCNN()
    logits = model(x) # x: [64,10]
    # LossFunction criterion = new CrossEntropyLoss(); // sadece obje oluşturma
    # Bu satır hesap yapmaz, sadece loss heplayıcı nesne(criterion) oluşturur
    # Bu nesne içerisinde şunlar saklanır; loss türü -> cross entropy, varsayılan reduction="mean" -> batch içindeki örneklerin loss'unu ortala
    criterion = nn.CrossEntropyLoss()  # reduction="mean"
    # Bu satır asıl hesabın yapıldığı satırdır.
    # Burada criterion artık bir fonksiyon gibi çağrılır ve içine, logits(modelin ham çıktısı) -> [B, 10], y(gerçek etiketler) -> [B]
    # Hesaplamada her bir görüntü için softmax ile loss bulunur, reduction="mean" olduğu için de tüm lossların ortalaması alınıp tek bir değer döndürülür
    loss = criterion(logits, y) # tek sayı (scalar)

    preds = logits.argmax(dim=1)

    acc = (preds == y).float().mean().item() # 0-1 arası
    # (preds == y)
    # preds: [8, 1, 0, 7, ...]
    # y: [8, 3, 0, 2, ...]
    # ==: [T, F, T, F, ...]

    #.float()
    #[T, F, T, F, ...] -> [1.0, 0.0, 1.0, 0.0, ...]

    # .mean()
    # mean = (1 + 0 + 1 + 0 + ...) / ... = 0.5

    # .item()
    # tensor(0.5) -> 0.5

    print("\n[debug_loss_one_batch]")
    print("logits shape", logits.shape)
    print("y shape: ", y.shape)
    print("loss: ", loss.item())
    print("batch acc", acc)
    print("first pred/true: ", preds[0].item(), "/", y[0].item())

def debug_train_one_step(cfg: Config) -> None:
    train_loader, _ = build_dataloaders(cfg)
    x, y = next(iter(train_loader))

    # Veriler dahil(x, y) herşeyi aynı cihaza(cpu/gpu) taşıyoruz, model CPU'da veriler GPU'da olursa hata alırız.
    model = SimpleCNN().to(cfg.device)
    x = x.to(cfg.device)
    y = y.to(cfg.device)

    criterion = nn.CrossEntropyLoss()

    # Adam optimezer'ı oluşturur
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Modeli train mode'a alır. Modelde Dropout/BatchNorm yok ama ilerde olursa fark eder, bu yüzden iyi bir alışkanlık.
    model.train()

    # Model, x'ten logits üretir -> logits shape: [64,10]
    logits1 = model(x)
    # Bu batch'daki ortalama loss hesaplanır, loss1 tek bir sayı verir(scaler tensor)
    # Bu noktada henüz hiçbir şey öğrenmedik, sadece şuanki model ne kadar hata yapıyoru ölçük
    loss1 = criterion(logits1, y)

    # Pytorch'da gradientler birikir, bu satır önceki adımların gradientlerini temizler, aksi halde model yanlış şekilde büyüyerek gider
    optimizer.zero_grad()
    # Backprop(geri yayılım) başlar, loss1'i düzeltmek için ağırlıkların hangi yönde değişmesi gerektiği hesaplanır, bu hesaplanan türevler
    # her parametrenin .grad alanına yazılır
    loss1.backward()
    # Adam, .grad değerine bakar ve ağırlıkları günceller, loss'u azaltacak yönde küçük bir adım atar
    # Bu adımadn sonra modelin ağırlıkları değişir, yani bir adım öğrenme gerçekleşir
    optimizer.step()

    # Aynı batch tekrar modele sokulur
    logits2 = model(x)
    # Yeni ağırlıklarla aynı batch'deki loss tekrar ölçülür, eğer learning(öğrenme) tekrar olduysa loss2, loss1'den daha küçük olur
    loss2 = criterion(logits2, y)

    print("\n[debug_train_one_step]")
    print("loss before: ", loss1.item())
    print("loss after: ", loss2.item())

# save -> load(best) -> predict zinciri testi
def debug_load_best_and_predict(cfg: Config, ckpt_path: str | None = None) -> None:
    if ckpt_path is None:
        ckpt_path = cfg.ckpt_path

    _, test_loader = build_dataloaders(cfg)


    # model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))# ckpt -> check point
    model = load_model(cfg, ckpt_path)
    model.eval()

    x,y = next(iter(test_loader))
    x = x.to(cfg.device)
    y = y.to(cfg.device)

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)

    acc = (preds == y).float().mean().item()

    print("\n[debug_load_best_and_predict]")
    print("batch acc:", acc)
    print("first 10 true", y[:10].tolist())
    print("first 10 pred", preds[:10].tolist())

def debug_misclassified(cfg: Config, ckpt_path: str | None = None, max_show: int = 15) -> None:
    if ckpt_path is None:
        cktp_path = cfg.ckpt_path

    _, test_loader = build_dataloaders(cfg)

    model = build_model(cfg)
    #model = SimpleCNN2().to(cfg.device)
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    model.eval()

    shown = 0
    total_wrong = 0
    total_seen = 0

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            logits = model(x)
            preds = logits.argmax(dim = 1)

            wrong_mask = preds != y
            wrong_idx = wrong_mask.nonzero(as_tuple=False).squeeze(1)

            total_wrong += wrong_mask.sum().item()
            total_seen += y.size(0)

            for idx in wrong_idx.tolist():
                if shown >= max_show:
                    break
                print(f"wrong #{shown+1}: true={y[idx].item()} pred={preds[idx].item()}")
                shown +=1

            if shown >= max_show:
                break

    print("\n[debug_misclassified]")
    print("wrong so far:", total_wrong, "seen:", total_seen, "wrong rate:", total_wrong/total_seen)

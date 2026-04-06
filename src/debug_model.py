from config import Config
from data_utils import build_dataloaders
from model import SimpleCNN

def debug_model_forward_shape(cfg: Config) -> None:
    train_loader, _ = build_dataloaders(cfg)
    # train_loader, test_loader = build_dataloaders(cfg)
    # ama ben test_loader'ı bu fonksiyonda kullanmayacağım
    # x, _ = (10, 20)  # sadece 10 lazım

    x,y = next(iter(train_loader)) # x : [64, 1, 28, 28]

    model = SimpleCNN()
    out = model(x)

    print("\n[debug_model_forward_shape]")
    print("input shape : ", x.shape)
    print("output shape", out.shape)

def debug_model_pool_shape(cfg: Config) -> None:
    train_loader, _ =  build_dataloaders(cfg)
    x, y = next(iter(train_loader)) # [64 , 1, 28, 28]

    model = SimpleCNN()
    out = model(x)

    print("\n[debug_model_pool_shape]")
    print("input: ", x.shape) # [64, 1, 28, 28]
    print("output: ", out.shape) # [64, 1, 14, 14]
    print("pixel before pool:", 28*28, "after: ", 14*14)

def debug_model_classifier_shape(cfg: Config) -> None:
    train_loader, _ = build_dataloaders(cfg)
    x, y = next(iter(train_loader)) # x: [64, 1, 28, 28]

    model = SimpleCNN()
    out = model(x) # out = [64,10]

    print("\n[debug_model_classifier_shape]")
    print("input shape: ", x.shape)
    print("output shape: ", out.shape)
    # Çıktı batch'inin ilk görüntüsünü( out[0, ..]) al, gradient takibinden ayır(detach()), tensoru python listesine çevir ve ekrana yazdır
    # .detach() -> bunu sadece görüntülemek için kullanıyorum, öğrenme grafiğine bağlama
    print("10 logits:", out[0, :10].detach().tolist()) # çıktıya göre büyük ihtimal "8"
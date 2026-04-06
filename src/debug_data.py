import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import Config
from data_utils import build_transforms, build_dataloaders


def debug_transforms() -> None:
    t = build_transforms()

    x = torch.rand(1, 28, 28)  # tek bir sahte görüntü (0-1 arası) -> 1 kanal(siyah-beyaz, RGB olsa 3 kanal), 28x28 pixel
    x2 = t(transforms.ToPILImage()(x))  # PIL'e çevirip transform uygula

    print("\n[debug_transforms]")
    print("Before:", x.mean().item(), x.std().item())
    print("After :", x2.mean().item(), x2.std().item())

def debug_mninst_batch_stats(cfg: Config) -> None:
    t = build_transforms()

    ds = datasets.MNIST(
        root = cfg.data_dir,
        train = True,
        download=True,
        transform =  t
    )

    # Verisetinin herhangi bir yerinden 256 örnek alıp mean/std nasıl ona bakıyoruz
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    x, y = next(iter(loader)) # 1 batch al -> x görüntü sayısı, y etiket(0 - 9 arası rakamlardan hangisi)
    print("\n[debug_mnist_batch_stats]")
    print("x shape", x.shape)# beklenen: [256, 1, 28, 28]
    print("batch mean: ", x.mean().item())# .item(tensor sayısını, python sayısına(int float) çevirir
    print("batch std: ", x.std().item())
    print("labels example: ", y[:10].tolist())

    x0 = x[0] # batch içindeki birinci görüntü
    y0 = y[0].item() # onun etiketi (python int)
    print("first sample shape:", x0.shape) # [1, 28, 28]
    # Veri seti gri olduğu için, pixel değeri 0(normalize -0.42) -> hiç ışık yok, 255 -> max ışık(normalize 2.808)
    print("first label:", y0, "| min/max", x0.min().item(), x0.max().item())

    print("\n[first 10 samples]")
    for i in range(10):
        xi = x[i]             # i. görüntü (shape : [1, 28, 28]
        yi = y[i].item()      # i. etiket
        print(f"{i} : label={yi} | min={xi.min().item():.3f} | max={xi.max().item():.3f}")

def debug_shuffle_effect(cfg: Config) -> None:
    t = build_transforms()
    ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=t)

    # Eğitimde (train) genelde shuffle=True kullanılır:
    # Veri sırası karıştığı için model sırayı ezberlemez, batch'ler daha çeşitli olur
    # ve öğrenme daha stabil hale gelebilir.

    # Doğrulama/Test (val/test) için genelde shuffle=False kullanılır:
    # sonuçlar daha tekrarlanabilir olur ve raporlama/debug daha tutarlı olur.
    # (Doğruluk gibi metrikler genelde sıradan bağımsızdır.)

    loader_no_shuffle = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    loader_shuffle = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

    _, y1 = next(iter(loader_no_shuffle))
    _, y2 = next(iter(loader_no_shuffle))

    _, ys1 = next(iter(loader_shuffle))
    _, ys2 = next(iter(loader_shuffle))

    print("\n[debug_shuffle_effect]")
    print("no shuffle batch-1 labels:", y1.tolist())
    print("no shuffle batch-2 labels", y2.tolist())
    print("shuffle batch-1 labels", ys1.tolist())
    print("shuffle batch-1 labels", ys2.tolist())

    def debug_train_test_split(cfg: Config):
        t = build_transforms()

        train_ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=t)
        test_ds = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=t)

        print("\n[debug_train_test_split]")
        print("train_size:", len(train_ds))  # beklenen 60000
        print("test_size:", len(test_ds))  # beklenen 10000

        x_tr, y_tr = train_ds[0]
        x_te, y_te = test_ds[0]

        print("train[0] label:", y_tr, "| shape:", x_tr.shape)
        print("test[0] label:", y_te, "| shape:", x_te.shape)

def debug_train_test_split(cfg: Config):
    t = build_transforms()

    train_ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=t)
    test_ds = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=t)

    print("\n[debug_train_test_split]")
    print("train_size:", len(train_ds))# beklenen 60000
    print("test_size:", len(test_ds))# beklenen 10000

    x_tr, y_tr = train_ds[0]
    x_te, y_te = test_ds[0]

    print("train[0] label:", y_tr, "| shape:", x_tr.shape)
    print("test[0] label:", y_te, "| shape:", x_te.shape)

def debug_dataloaders_one_batch(cfg : Config) -> None:
    train_loader, test_loader = build_dataloaders(cfg)

    x_tr, y_tr = next(iter(train_loader))
    x_te, y_te = next(iter(test_loader))

    print("\n[debug_data_loaders_one_batch]")
    print("train batch x:", x_tr.shape, "| y:", y_tr.shape, "| labels:", y_tr[:10].tolist())
    print("test_batch x:", x_te.shape, "| y:", y_te.shape, "| labels", y_te[:10].tolist())
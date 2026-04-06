# Genel yardımcı fonksiyonların olduğu dosya
import os
import json
import random
import torch
from torch import nn
from config import Config
from model import build_model

def ensure_output_dir(path: str) -> None:
    folder = os.path.dirname(path) # fonksiyon çağrısında verilen yolun klasör kısmını alır; "outputs/best_mnist_cnn.pt" -> "outputs"
    if folder: # verilen klasör gerçekten var mı yok mu(dosyamda "outputs" klasörü var mı yok mu)
        os.makedirs(folder, exist_ok=True) # klasör varsa hata verme, yoksa oluştur

def save_json(data:dict, path: str) -> None:
    ensure_output_dir(path) # outputs -> var mı?
    with open(path, "w", encoding="utf-8") as f: # dosyayı yazma modunda(w) açar, f olarak alır, utf-8 metin kodlamasını belirler(türkçe karakterlerde sorun yaşamamak için good practice)
        # json.dump -> python verisini alıp, json formatında bir dosyanınn("training_history.json") içine yazar.
        json.dump(data, f, indent=4) # data -> history, f -> training_history.json daha geniş okunabilir olmasını sağlar

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model: nn.Module, path: str) -> None:
    ensure_output_dir(path)
    # state.dict() -> modelin içindeki öğrenilmiş parametreleri sözlük gibi döndürür, örn;
    # {
    #     "conv1.weight": ...,
    #     "conv1.bias": ...,
    #     "conv2.weight": ...,
    #     "conv2.bias": ...,
    #     "fc.weight": ...,
    #     "fc.bias": ...
    # }
    # torch.save() -> verdiğimiz python objesini(state.dict()), diske yani "best_mnist_cnn.pt" dosyasının içine yazar
    torch.save(model.state_dict(), path) # en iyi modeli "outputs" klasörü içindeki "best_mnist_cnn.pt" içine kaydeder

def load_model(cfg: Config, ckpt_path: str | None = None) -> nn.Module:
    if ckpt_path is None:
        ckpt_path = cfg.ckpt_path

    model = build_model(cfg)
    # model = SimpleCNN2().to(cfg.device)

    # torch.load() -> disteki dosyayı("best_mnist_cnn.pt") açar, içindeki kaydedilmiş veriyi geri okur, yani torch.save() ile kaydettiğin şeyi geri getirir
    # bu satırın  sonucu genelde "state_dict" benzeri, yani paremetre isimleri ve karşılarında tensor değerleri olan sözlük yapısıdır.
    # load_state_dict() -> boş yada yeni oluşturulmuş modelin içine, dosyadan gelen ağırlıkları yerleştirir.
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    model.eval()
    return model




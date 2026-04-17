import torch
from torch import nn
from config import Config

class SimpleCNN(nn.Module):
    def __init__(self):

        # nn.Model'ın "model altyapısını" başlatır, yani alt modülleri(layerları) kaydetmeye hazır hale gelir
        # Parametre takibini açar, (model.parameters()),  .to(device), .train()/.eval(), .state_dict() gibi şeylerin düzgün çalışmasını sağlar
        super().__init__()
        # in_channels=1 (MNIST), out_channels=8(8 farklı filtre uygulanmış , biri kenarları, biri köşeleri, diğeri kalın çizgileri vs yakalar)
        # kernel_size=3 => 3 x 3 filtre, padding=1 => boyut korunur
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        # self.fc = nn.Linear(in_features, out_features)
        # 8 kanal * 14 * 14 = 1568 feature(özellik)
        # [1568] uzunlukta feature -> [10] uzunlukta skor
        # Bu 10 skor -> sınıf 0 için skor, sınıf 1 için skor ..... sınıf 9 için skor -> en büyük skor hangi sınıfa aitse model "tahminim bu" der
        # Diyelim ki bir resim için fc şunu üretti -> [-1.2, 0.3, 2.1, ... , -0.7] -> en büyük değer 2.1, index = 2 'de ise -> bu resim "2" olabilir
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self,x):
        x = self.conv1(x) # [B, 1, 28, 28] -> [B, 8, 28, 28]
        x = self.relu(x)
        x = self.pool(x) # [B, 8, 28, 28] -> [B, 8, 14, 14]

        # Flatten: [B, 8, 14, 14] -> [B,8*14*14] -> [B, 1568]
        x = torch.flatten(x, 1)
        # x = x.view(x.size(0), -1)
        # x.size(0) -> batch size(B)
        # -1 = geri kalan tüm boyutları çarp ve tek boyut yap

        # Lineer: [B, 1568(in_fatures)] -> [B, 10(out_fatures)]
        x = self.fc(x)

        return x


class SimpleCNN2(nn.Module):
    def __init__(self, hidden_dim: int = 0, dropout_rate: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(1,8,3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        if self.hidden_dim > 0:
            self.fc1 =  self.fc1 = nn.Linear(16 * 7 * 7, hidden_dim)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.fc2 = nn.Linear(hidden_dim, 10)
        else:
            self.fc_out = nn.Linear(16 * 7 * 7, 10)


    def forward(self, x):
        # Starting -> [B, 1, 28, 28]
        x = self.pool(self.relu(self.conv1(x))) # [B,8,14,14] -> conv1 + pool
        x = self.pool(self.relu(self.conv2(x))) # [B,16,7,7]  -> conv2 + pool
        x = torch.flatten(x, 1)                 # [B, 784]    -> flatten

        if self.hidden_dim > 0:
            x = self.relu(self.fc1(x)) # [B, hidden_dim]    -> fc1
            if self.dropout_rate > 0:
                x = x.dropout(x)       # droput'u config'e bağlı uyguluyoruz, yani hidden layer(dense) varsa ve dropout > 0 ise
            x = self.fc2(x)            # [B, 10]     -> fc2
        else:
            x = self.fc_out(x)

        return x


def build_model(cfg: Config) -> nn.Module:
    model = SimpleCNN2(
        hidden_dim=cfg.hidden_dim,
        dropout_rate=cfg.dropout_rate
    ).to(cfg.device)
    return model


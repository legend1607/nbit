"""
auto_cae_train_v4.py
--------------------
改进版 2D 卷积自编码器训练脚本：
- 支持任意输入大小
- 支持验证集
- 使用 MSELoss 代替 BCELoss
- 去掉 Dropout 提升重建精度
- 使用 CosineAnnealingLR 调整学习率
- 保存最新与最优模型
"""

import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ===============================
# Encoder
# ===============================
class Encoder_CNN_2D(nn.Module):
    def __init__(self, input_size=224, latent_dim=128):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )

        # 自动计算卷积输出特征图大小
        size = input_size
        for _ in range(4):
            size = math.floor((size - 1)/2 + 1)
        self.feature_map_size = size
        flatten_size = 128 * self.feature_map_size * self.feature_map_size

        self.fc_layers = nn.Sequential(
            nn.Linear(flatten_size, 512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.fc_layers(x_flat)
        return latent

# ===============================
# Decoder
# ===============================
class Decoder_CNN_2D(nn.Module):
    def __init__(self, feature_map_size=14, latent_dim=128):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.latent_dim = latent_dim

        flatten_size = 128 * feature_map_size * feature_map_size
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, flatten_size), nn.ReLU()
        )

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,2,1,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,2,1,1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,2,1,1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,2,1,1)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(x.size(0), 128, self.feature_map_size, self.feature_map_size)
        x = self.deconv_layers(x)
        return x

# ===============================
# 训练函数
# ===============================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 数据加载
    # -----------------------------
    train_data = np.load(args.dataset_path)
    train_data = torch.from_numpy(train_data).float().unsqueeze(1)
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_dataset_path:
        val_data = np.load(args.val_dataset_path)
        val_data = torch.from_numpy(val_data).float().unsqueeze(1)
        val_dataset = TensorDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_size = train_data.shape[-1]
    print(f"Input size detected: {input_size}x{input_size}")

    # -----------------------------
    # 模型初始化
    # -----------------------------
    encoder = Encoder_CNN_2D(input_size=input_size, latent_dim=args.latent_dim)
    decoder = Decoder_CNN_2D(feature_map_size=encoder.feature_map_size, latent_dim=args.latent_dim)
    encoder.to(device)
    decoder.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr/10)

    # -----------------------------
    # 保存目录
    # -----------------------------
    os.makedirs(args.model_path, exist_ok=True)
    latest_dir = os.path.join(args.model_path, "latest")
    best_dir = os.path.join(args.model_path, "best")
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    best_val_loss = float('inf')

    # -----------------------------
    # 训练循环
    # -----------------------------
    for epoch in range(1, args.num_epochs+1):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)
            latent = encoder(x)
            recon = decoder(latent)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(train_dataset)

        # -----------------------------
        # 验证集
        # -----------------------------
        if val_loader:
            encoder.eval()
            decoder.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_val = batch[0].to(device)
                    latent_val = encoder(x_val)
                    recon_val = decoder(latent_val)
                    val_loss_total += criterion(recon_val, x_val).item() * x_val.size(0)
            val_loss = val_loss_total / len(val_dataset)
        else:
            val_loss = epoch_loss

        scheduler.step()

        print(f"Epoch [{epoch}/{args.num_epochs}] - Train Loss: {epoch_loss:.6f} - Val Loss: {val_loss:.6f}")

        # -----------------------------
        # 保存最新模型
        # -----------------------------
        torch.save(encoder.state_dict(), os.path.join(latest_dir, "encoder_latest.pth"))
        torch.save(decoder.state_dict(), os.path.join(latest_dir, "decoder_latest.pth"))

        # -----------------------------
        # 保存最优模型
        # -----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), os.path.join(best_dir, "encoder_best.pth"))
            torch.save(decoder.state_dict(), os.path.join(best_dir, "decoder_best.pth"))
            print(f"✅ Best model updated at epoch {epoch} with val loss {best_val_loss:.6f}")

# ===============================
# 入口
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/random_2d/grids_train.npy')
    parser.add_argument('--val_dataset_path', type=str, default='data/random_2d/grids_val.npy')
    parser.add_argument('--model_path', type=str, default='./results')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()

    train(args)

"""
auto_cae_train_v6.py
--------------------
改进版高精度 2D 卷积自编码器：
- 更深层次的卷积结构（ResBlock + U-Net skip）
- 使用 InstanceNorm + SiLU 激活函数
- 添加感知损失 (SSIM + MSE)
- 优化器使用 AdamW
- 使用 OneCycleLR 学习率策略
- Skip connection 对齐修复，支持任意偶数输入尺寸
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import structural_similarity_index_measure as ssim


# ==================================================
# 🔹 辅助函数：裁剪或填充到目标尺寸
# ==================================================
def center_crop_or_pad(tensor, target_hw):
    """
    保证拼接时两特征图空间尺寸一致。
    tensor: (B,C,H,W)
    target_hw: (H_target, W_target)
    """
    _, _, h, w = tensor.shape
    ht, wt = target_hw
    # 裁剪
    if h > ht or w > wt:
        start_h = (h - ht) // 2
        start_w = (w - wt) // 2
        tensor = tensor[:, :, start_h:start_h+ht, start_w:start_w+wt]
        return tensor
    # 填充
    pad_h = max(0, ht - h)
    pad_w = max(0, wt - w)
    pad = (0, pad_w, 0, pad_h)
    return F.pad(tensor, pad)


# ==================================================
# 🔹 基础残差模块
# ==================================================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels)
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))


# ==================================================
# 🔹 Encoder
# ==================================================
class Encoder_CNN_2D(nn.Module):
    def __init__(self, input_size=224, latent_dim=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.enc1 = nn.Sequential(
            ResBlock(1, 32),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            ResBlock(32, 64),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            ResBlock(64, 128),
            nn.MaxPool2d(2)
        )
        self.enc4 = nn.Sequential(
            ResBlock(128, 256),
            nn.MaxPool2d(2)
        )

        size = input_size // 16
        self.feature_map_size = size
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * size * size, 1024),
            nn.SiLU(),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        e1 = self.enc1(x)  # (B,32,H/2,W/2)
        e2 = self.enc2(e1) # (B,64,H/4,W/4)
        e3 = self.enc3(e2) # (B,128,H/8,W/8)
        e4 = self.enc4(e3) # (B,256,H/16,W/16)

        latent = self.fc(self.flatten(e4))
        return latent, [e1, e2, e3, e4]


# ==================================================
# 🔹 Decoder
# ==================================================
class Decoder_CNN_2D(nn.Module):
    def __init__(self, feature_map_size=14, latent_dim=256):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 256 * feature_map_size * feature_map_size),
            nn.SiLU()
        )

        # 对齐 skip 层
        self.up1 = ResBlock(256 + 128, 128)  # concat e3
        self.up2 = ResBlock(128 + 64, 64)    # concat e2
        self.up3 = ResBlock(64 + 32, 32)     # concat e1
        self.up4 = ResBlock(32, 16)          # final up no skip

        self.final = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, latent, enc_feats):
        e1, e2, e3, e4 = enc_feats

        x = self.fc(latent)
        x = x.view(x.size(0), 256, self.feature_map_size, self.feature_map_size)

        # up1: (B,256,H/8,W/8) + e3
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = center_crop_or_pad(x, e3.shape[-2:])
        x = self.up1(torch.cat([x, e3], dim=1))

        # up2: (B,128,H/4,W/4) + e2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = center_crop_or_pad(x, e2.shape[-2:])
        x = self.up2(torch.cat([x, e2], dim=1))

        # up3: (B,64,H/2,W/2) + e1
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = center_crop_or_pad(x, e1.shape[-2:])
        x = self.up3(torch.cat([x, e1], dim=1))

        # up4: (B,32,H,W)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up4(x)

        x = self.final(x)
        return x


# ==================================================
# 🔹 训练函数
# ==================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_data = torch.from_numpy(np.load(args.dataset_path)).float().unsqueeze(1)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_dataset_path and os.path.exists(args.val_dataset_path):
        val_data = torch.from_numpy(np.load(args.val_dataset_path)).float().unsqueeze(1)
        val_loader = DataLoader(TensorDataset(val_data), batch_size=args.batch_size, shuffle=False)

    input_size = train_data.shape[-1]

    encoder = Encoder_CNN_2D(input_size=input_size, latent_dim=args.latent_dim).to(device)
    decoder = Decoder_CNN_2D(feature_map_size=encoder.feature_map_size, latent_dim=args.latent_dim).to(device)

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.num_epochs * len(train_loader))
    mse_loss = nn.MSELoss()

    best_val = float('inf')
    os.makedirs(args.model_path, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        encoder.train(), decoder.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = batch[0].to(device)
            latent, enc_feats = encoder(x)
            recon = decoder(latent, enc_feats)
            loss_mse = mse_loss(recon, x)
            loss_ssim = 1 - ssim(recon, x)
            loss = loss_mse + 0.2 * loss_ssim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_data)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f}")

        # 验证
        if val_loader:
            encoder.eval(); decoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    latent, enc_feats = encoder(x)
                    recon = decoder(latent, enc_feats)
                    val_loss += mse_loss(recon, x).item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            print(f"Val Loss: {val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(encoder.state_dict(), os.path.join(args.model_path, "encoder_best.pth"))
                torch.save(decoder.state_dict(), os.path.join(args.model_path, "decoder_best.pth"))
                print(f"✅ Best model updated: {best_val:.6f}")


# ==================================================
# 🔹 主入口
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/random_2d/grids_train.npy')
    parser.add_argument('--val_dataset_path', type=str, default='data/random_2d/grids_val.npy')
    parser.add_argument('--model_path', type=str, default='./results')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    train(args)

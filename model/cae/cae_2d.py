import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =====================================================
# Dataset
# =====================================================
class GridDataset(Dataset):
    def __init__(self, grid_path):
        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"Grid file not found: {grid_path}")

        print(f"📂 Loading grids from {grid_path} ...")
        self.grids = np.load(grid_path, mmap_mode="r")
        print(f"✅ Loaded {len(self.grids)} grids, original shape = {self.grids.shape}")

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        grid = self.grids[idx]
        grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        return grid_tensor / 1.0  # 保留 0/1


# =====================================================
# Encoder
# =====================================================
class Encoder_CNN_2D(nn.Module):
    def __init__(self, latent_dim=128, dropout_p=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_p = dropout_p

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = None  # 延迟初始化

    def forward(self, x):
        batch_size = x.size(0)
        x_conv = self.conv_layers(x)
        x_flat = x_conv.view(batch_size, -1)

        # 延迟初始化 fc_layers
        if self.fc_layers is None:
            in_features = x_flat.size(1)
            self.fc_layers = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(512, self.latent_dim)
            ).to(x.device)

        latent = self.fc_layers(x_flat)
        return latent, x_conv.size()[1:]  # 返回 feature_shape(C,H,W)


# =====================================================
# Decoder
# =====================================================
class Decoder_CNN_2D(nn.Module):
    def __init__(self, latent_dim=128, dropout_p=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_p = dropout_p

        self.fc_layers = None
        self.deconv_layers = None

    def forward(self, latent, feature_shape, target_size=None):
        C,H,W = feature_shape
        # 延迟初始化 fc_layers
        if self.fc_layers is None:
            self.fc_layers = nn.Sequential(
                nn.Linear(self.latent_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(512, int(np.prod(feature_shape))),
                nn.ReLU()
            ).to(latent.device)

        if self.deconv_layers is None:
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 1, 3, 2, 1, output_padding=1)
            ).to(latent.device)

        x = self.fc_layers(latent)
        x = x.view(x.size(0), C, H, W)
        x = self.deconv_layers(x)

        # 插值到原始尺寸
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


# =====================================================
# 训练
# =====================================================
def train_autoencoder(train_loader, val_loader, encoder, decoder, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)
    decoder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0

        for grids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            grids = grids.to(device)
            latent, feature_shape = encoder(grids)
            outputs = decoder(latent, feature_shape, target_size=grids.shape[2:])
            loss = criterion(outputs, grids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Train Loss: {avg_loss:.6f}")

        # 保存模型
        if (epoch+1) % args.test_save_epoch == 0:
            os.makedirs(args.model_path, exist_ok=True)
            en_path = os.path.join(args.model_path, f"encoder_epoch_{epoch+1}.pth")
            de_path = os.path.join(args.model_path, f"decoder_epoch_{epoch+1}.pth")
            torch.save(encoder.state_dict(), en_path)
            torch.save(decoder.state_dict(), de_path)
            print(f"✅ Saved models: {en_path}, {de_path}")

    return encoder, decoder


# =====================================================
# 重建可视化
# =====================================================
def visualize_reconstruction(encoder, decoder, dataloader, num_examples=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for grids in dataloader:
            grids = grids.to(device)
            latent, feature_shape = encoder(grids)
            recon = decoder(latent, feature_shape, target_size=grids.shape[2:])
            break

    fig, axes = plt.subplots(2, num_examples, figsize=(2*num_examples,4))
    for i in range(num_examples):
        axes[0,i].imshow(grids[i,0].cpu(), cmap='gray')
        axes[0,i].set_title("Input")
        axes[0,i].axis('off')
        axes[1,i].imshow(recon[i,0].cpu(), cmap='gray')
        axes[1,i].set_title("Reconstruction")
        axes[1,i].axis('off')
    plt.tight_layout()
    plt.show()


# =====================================================
# 主程序
# =====================================================
def main(args):
    encoder = Encoder_CNN_2D(latent_dim=args.latent_dim, dropout_p=args.dropout_p)
    decoder = Decoder_CNN_2D(latent_dim=args.latent_dim, dropout_p=args.dropout_p)

    train_dataset = GridDataset(args.train_grid_path)
    val_dataset = GridDataset(args.val_grid_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    encoder, decoder = train_autoencoder(train_loader, val_loader, encoder, decoder, args)
    visualize_reconstruction(encoder, decoder, val_loader)


# =====================================================
# 参数解析
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_grid_path', type=str, default='data/random_2d/grids_train.npy')
    parser.add_argument('--val_grid_path', type=str, default='data/random_2d/grids_val.npy')
    parser.add_argument('--model_path', type=str, default='results/cae')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--test_save_epoch', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    args = parser.parse_args()
    print(args)
    main(args)

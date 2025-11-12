"""
train_pathkeypoint_complete_masked.py
-------------------------------------
训练 PathKeypointGenerator，支持不同长度专家路径
- 输入: CAE latent + 起止点
- 输出: n_points 路径关键点
- 损失: 高斯加权路径 + 拐点 + 平滑
- 支持 TensorBoard
- 支持路径填充 (padding) + 掩码 (mask)
"""

import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.cae.cae_2d import Encoder_CNN_2D
from model.PathGeneratorModel import PathKeypointGenerator, PathLossGaussian

# =========================
# 自定义 Dataset
# =========================
class PathDataset(Dataset):
    def __init__(self, npz_path, n_points):
        data = np.load(npz_path, allow_pickle=True)
        self.grids = data['grid']      # (N,H,W)
        self.starts = data['start']    # (N,2)
        self.goals = data['goal']      # (N,2)
        self.paths_raw = data['path']  # list of (L_i,2)
        self.H, self.W = self.grids.shape[1], self.grids.shape[2]
        self.n_points = n_points

        # ========== 填充路径到统一长度 n_points ==========
        self.paths = np.zeros((len(self.paths_raw), n_points, 2), dtype=np.float32)
        self.masks = np.zeros((len(self.paths_raw), n_points), dtype=np.float32)
        for i, path in enumerate(self.paths_raw):
            L = min(len(path), n_points)
            self.paths[i, :L] = path[:L]
            self.masks[i, :L] = 1.0  # 有效位置为1，填充为0

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        grid = self.grids[idx].astype(np.float32)[None, :, :]  # (1,H,W)
        start = self.starts[idx].astype(np.float32) / np.array([self.W, self.H], dtype=np.float32)
        goal = self.goals[idx].astype(np.float32) / np.array([self.W, self.H], dtype=np.float32)
        path = self.paths[idx].astype(np.float32) / np.array([self.W, self.H], dtype=np.float32)
        mask = self.masks[idx].astype(np.float32)
        return grid, start, goal, path, mask


# =========================
# 参数设置
# =========================
def parse_args():
    parser = argparse.ArgumentParser('Train PathKeypointGenerator with Mask')
    parser.add_argument('--train_npz', type=str, default='data/random_2d/train.npz')
    parser.add_argument('--val_npz', type=str, default='data/random_2d/val.npz')
    parser.add_argument('--encoder_ckpt', type=str, default='results/encoder_best.pth')
    parser.add_argument('--save_dir', type=str, default='results/pathgenerator_masked')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--n_points', type=int, default=128)
    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()


# =========================
# 主训练函数
# =========================
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # logging
    logging.basicConfig(filename=os.path.join(args.save_dir, 'train.log'),
                        level=logging.INFO,
                        format='%(asctime)s %(message)s')
    log = lambda s: (print(s), logging.info(s))

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))

    # 随机种子
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    log(f"Random seed: {args.random_seed}")

    # 数据集
    train_ds = PathDataset(args.train_npz, args.n_points)
    val_ds = PathDataset(args.val_npz, args.n_points)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log(f"Loaded train={len(train_ds)}, val={len(val_ds)}")

    # CAE Encoder
    input_size = train_ds.grids.shape[1]
    encoder = Encoder_CNN_2D(input_size=input_size, latent_dim=args.latent_dim)
    encoder.load_state_dict(torch.load(args.encoder_ckpt))
    encoder.cuda()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    log("Loaded CAE encoder and frozen parameters")

    # PathKeypointGenerator
    model = PathKeypointGenerator(obs_latent_dim=args.latent_dim, n_points=args.n_points)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = PathLossGaussian()

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        # ========== 训练 ==========
        model.train()
        train_loss = 0.0
        for grids, starts, goals, paths, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            grids = grids.cuda()
            starts = starts.cuda()
            goals = goals.cuda()
            paths = paths.cuda()
            masks = masks.cuda()

            # CAE latent
            with torch.no_grad():
                latent, _ = encoder(grids)

            # 预测路径关键点
            pred_paths = model(latent, starts, goals)

            # 损失，支持 mask
            loss, loss_path, loss_smooth = loss_fn(pred_paths, paths, masks=masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * grids.size(0)
            global_step += 1
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

        avg_train_loss = train_loss / len(train_ds)
        writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch)
        log(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f}")

        # ========== 验证 ==========
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for grids, starts, goals, paths, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                grids = grids.cuda()
                starts = starts.cuda()
                goals = goals.cuda()
                paths = paths.cuda()
                masks = masks.cuda()

                latent, _ = encoder(grids)
                pred_paths = model(latent, starts, goals)
                loss, _, _ = loss_fn(pred_paths, paths, masks=masks)
                val_loss_total += loss.item() * grids.size(0)

        avg_val_loss = val_loss_total / len(val_ds)
        writer.add_scalar('Val/EpochLoss', avg_val_loss, epoch)
        log(f"Epoch {epoch} | Val Loss: {avg_val_loss:.6f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.save_dir, 'pathgenerator_best.pth')
            torch.save(model.state_dict(), ckpt_path)
            log(f"✅ Saved best model: {ckpt_path}")

    writer.close()
    log(f"Training complete. Best Val Loss: {best_val_loss:.6f}")


# =========================
# 运行入口
# =========================
if __name__ == '__main__':
    args = parse_args()
    main(args)

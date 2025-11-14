"""
train_pathkeypoint_npz.py
-------------------------
训练 PathKeypointGenerator，使用 npz 中 path 数据。
- 输入: CAE latent + 起止点
- 输出: n_points 无序散点路径关键点
- 损失: 覆盖 soft label 区域
- 支持 TensorBoard
- 固定 bound_min / bound_max
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
from model.PathGeneratorModel import PathKeypointGenerator, PathGenerationLoss

# ------------------------------
# 数据集处理：路径填充/掩码
# ------------------------------
def pad_paths(paths, max_len):
    padded_paths = []
    masks = []
    for p in paths:
        L = p.shape[0]
        if L < max_len:
            pad_len = max_len - L
            pad_points = np.repeat(p[-1][None, :], pad_len, axis=0)
            padded = np.vstack([p, pad_points])
        else:
            padded = p
        mask = np.zeros(max_len, dtype=np.float32)
        mask[:L] = 1.0
        padded_paths.append(padded)
        masks.append(mask)
    return np.stack(padded_paths, axis=0), np.stack(masks, axis=0)


class PathDataset(Dataset):
    """
    Dataset for PathKeypointGenerator training.
    返回: grid / start / goal / path / mask
    """
    def __init__(self, npz_path, max_len=None):
        data = np.load(npz_path, allow_pickle=True)
        self.grids = data['grid']      # (N,H,W)
        self.starts = data['start']    # (N,2)
        self.goals = data['goal']      # (N,2)
        paths = list(data['path'])           # list of (N_pc,2)

        if max_len is None:
            max_len = max([p.shape[0] for p in paths])
        self.paths, self.masks = pad_paths(paths, max_len)

        self.n_samples = len(self.grids)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        grid = torch.from_numpy(self.grids[idx].astype(np.float32))[None, :, :]
        start = torch.from_numpy(self.starts[idx].astype(np.float32))
        goal = torch.from_numpy(self.goals[idx].astype(np.float32))
        path = torch.from_numpy(self.paths[idx].astype(np.float32))
        mask = torch.from_numpy(self.masks[idx].astype(np.float32))
        return grid, start, goal, path, mask

# ------------------------------
# 参数设置
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser('Train PathKeypointGenerator with npz soft labels')
    parser.add_argument('--train_npz', type=str, default='data/random_2d/train.npz')
    parser.add_argument('--val_npz', type=str, default='data/random_2d/val.npz')
    parser.add_argument('--encoder_ckpt', type=str, default='results/cae/encoder_best.pth')
    parser.add_argument('--save_dir', type=str, default='results/pathgenerator_npz')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--n_points', type=int, default=128)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--bound_min', nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument('--bound_max', nargs=2, type=float, default=[224, 224])
    parser.add_argument('--sigma', type=float, default=0.5, help="Gaussian sigma for soft label loss")
    return parser.parse_args()

# ------------------------------
# 主训练函数
# ------------------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.save_dir, 'train.log'),
                        level=logging.INFO,
                        format='%(asctime)s %(message)s')
    log = lambda s: (print(s), logging.info(s))
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    log(f"Random seed: {args.random_seed}")

    bound_min = np.array(args.bound_min, dtype=np.float32)
    bound_max = np.array(args.bound_max, dtype=np.float32)
    log(f"Using fixed bound: min={bound_min}, max={bound_max}")

    # ------------------------------
    # 数据集
    # ------------------------------
    train_ds = PathDataset(args.train_npz)
    val_ds = PathDataset(args.val_npz)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log(f"Loaded train={len(train_ds)}, val={len(val_ds)}, max path len={train_ds.paths.shape[1]}")

    # ------------------------------
    # Encoder
    # ------------------------------
    input_size = train_ds.grids.shape[1]
    encoder = Encoder_CNN_2D(input_size=input_size, latent_dim=args.latent_dim)
    encoder.load_state_dict(torch.load(args.encoder_ckpt))
    encoder.cuda()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    log("Loaded CAE encoder and frozen parameters")

    # ------------------------------
    # PathKeypointGenerator
    # ------------------------------
    model = PathKeypointGenerator(
        obs_latent_dim=args.latent_dim,
        latent_dim=args.latent_dim,
        n_points=args.n_points,
        bound_min=bound_min,
        bound_max=bound_max
    ).cuda()

    criterion = PathGenerationLoss(sigma=args.sigma).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    # ------------------------------
    # 训练循环
    # ------------------------------
    for epoch in range(1, args.num_epochs + 1):
        # -------- Train --------
        model.train()
        total_loss = 0.0
        for grid, start, goal, path, mask in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            grid = grid.cuda()
            start = start.cuda()
            goal = goal.cuda()
            path = path.cuda()
            mask = mask.cuda()

            # 1. CAE encode
            with torch.no_grad():
                latent, _ = encoder(grid)

            # 2. Forward
            pred_points = model(latent, start, goal)

            # 3. Loss
            loss, _ = criterion(pred_points, path, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * grid.size(0)

        avg_train_loss = total_loss / len(train_ds)
        # log(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}")
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)

        # -------- Validation --------
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for grid, start, goal, path, mask in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                grid = grid.cuda()
                start = start.cuda()
                goal = goal.cuda()
                path = path.cuda()
                mask = mask.cuda()

                latent, _ = encoder(grid)
                pred_points = model(latent, start, goal)
                loss, _ = criterion(pred_points, path, mask)
                val_loss_sum += loss.item() * grid.size(0)

        avg_val_loss = val_loss_sum / len(val_ds)
        log(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)

        # -------- Save best model --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, "best_pathgenerator.pth")
            torch.save(model.state_dict(), save_path)
            log(f"Saved best model at epoch {epoch}, Val Loss: {best_val_loss:.6f}")

    writer.close()


# ------------------------------
# Entry
# ------------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args)

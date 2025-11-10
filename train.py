import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.PathPlanDataLoader import PathPlanDataset, collate_fn
from model.PathGeneratorModel import ObstacleCAE, PathKeypointGenerator, PathLossTwoGaussWeighted

# ======================
# 参数设置
# ======================
def parse_args():
    parser = argparse.ArgumentParser('Train CAE + PathKeypointGenerator')
    parser.add_argument('--dim', type=int, default=2, help='Environment dimension (2 or 3)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--npoints', type=int, default=128, help='Number of generated path points')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dim for obstacle CAE')
    return parser.parse_args()


# ======================
# 主训练函数
# ======================
def main(args):
    # ---- 文件和日志目录 ----
    experiment_dir = os.path.join('results', f'CAE_PathGen_{args.dim}d')
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # ---- logging ----
    logging.basicConfig(
        filename=os.path.join(experiment_dir, 'train.log'),
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )
    log = lambda s: (print(s), logging.info(s))

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "tensorboard"))

    # ---- 随机种子 ----
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    log(f"Training CAE+PathGenerator with seed {args.random_seed}")

    # ---- 数据集 ----
    env_type = f"random_{args.dim}d"
    train_ds = PathPlanDataset(f"data/{env_type}/train.npz")
    val_ds = PathPlanDataset(f"data/{env_type}/val.npz")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: collate_fn(x, device='cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: collate_fn(x, device='cuda'))
    log(f"Loaded train={len(train_ds)}, val={len(val_ds)}")

    # ---- 模型 ----
    cae = ObstacleCAE(n_obs=10, obs_dim=4, latent_dim=args.latent_dim).cuda()
    path_gen = PathKeypointGenerator(obs_latent_dim=args.latent_dim, point_dim=args.dim, latent_dim=args.latent_dim, n_points=args.npoints).cuda()
    criterion = PathLossTwoGaussWeighted().cuda()
    optimizer = torch.optim.Adam(list(cae.parameters()) + list(path_gen.parameters()), lr=args.learning_rate)

    best_val_loss = float('inf')
    global_step = 0

    # ======================
    # 训练循环
    # ======================
    for epoch in range(args.epoch):
        log(f"\nEpoch [{epoch+1}/{args.epoch}] --------------------------")
        cae.train()
        path_gen.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            start = batch['start'].cuda()
            goal = batch['goal'].cuda()
            path = batch['path']
            path_mask = batch['path_mask'].cuda()
            obs = batch['rectangle_obstacles'].cuda()

            optimizer.zero_grad()
            obs_latent, obs_recon = cae(obs)
            path_pred = path_gen(obs_latent, start, goal)
            loss_total, loss_path, loss_smooth, loss_cae = criterion(path_pred, path_mask, path, obs_recon, obs)
            loss_total.backward()
            optimizer.step()

            train_loss += loss_total.item()
            global_step += 1
            writer.add_scalar('Train/BatchLoss', loss_total.item(), global_step)

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch)
        log(f"Train Loss: {avg_train_loss:.6f}")

        # ======================
        # 验证阶段
        # ======================
        cae.eval()
        path_gen.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                start = batch['start'].cuda()
                goal = batch['goal'].cuda()
                path = batch['path']
                path_mask = batch['path_mask'].cuda()
                obs = batch['rectangle_obstacles'].cuda()

                obs_latent, obs_recon = cae(obs)
                path_pred = path_gen(obs_latent, start, goal)
                loss_total, _, _, _ = criterion(path_pred, path_mask, path, obs_recon, obs)
                val_loss += loss_total.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        log(f"[Val] Loss={avg_val_loss:.6f}")

        # ======================
        # 保存最优模型
        # ======================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(checkpoints_dir, f'best_model_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'cae_state_dict': cae.state_dict(),
                'pathgen_state_dict': path_gen.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss
            }, ckpt_path)
            log(f" Saved best model to {ckpt_path}")

    writer.close()
    log("Training complete.")


# ======================
# 运行入口
# ======================
if __name__ == '__main__':
    args = parse_args()
    main(args)

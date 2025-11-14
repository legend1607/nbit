import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from model.cae.cae_2d import Encoder_CNN_2D, Decoder_CNN_2D
from model.PathGeneratorModel import PathKeypointGenerator, PathGenerationLoss

# ---------------- 栅格生成 ----------------
def render_grid_from_json(env_json, size=224):
    grid = np.zeros((size, size), dtype=np.float32)
    for rect in env_json["rectangle_obstacles"]:
        x, y, w, h = rect
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
        grid[y1:y2, x1:x2] = 1.0
    return grid

# ---------------- 可视化单样本四个点 loss ----------------
def visualize_pointwise_losses(env_json, grid, recon_grid, gt_path, pred_points, sigma=5.0):
    loss_fn = PathGenerationLoss(sigma=sigma)

    pred = torch.tensor(pred_points).unsqueeze(0).cuda()
    gt   = torch.tensor(gt_path).unsqueeze(0).cuda()
    mask = torch.ones(gt.shape[1], dtype=torch.float32).unsqueeze(0).cuda()

    # 计算四种 loss 对每个点贡献
    # Chamfer
    dist_chamfer = torch.cdist(pred, gt)
    min_dist_chamfer, _ = dist_chamfer.min(dim=-1)
    loss_chamfer = (1 - torch.exp(-0.5*(min_dist_chamfer/sigma)**2)).squeeze(0).cpu().numpy()

    # Centerline
    # 对每个点计算到每段中心线最小距离
    B, N_pred, _ = pred.shape
    M = gt.shape[1]
    total_weight = torch.zeros(B, N_pred).to(pred.device)
    valid_segments = 0
    for i in range(M-1):
        seg_mask = mask[:, i] * mask[:, i+1]
        if seg_mask.sum() == 0:
            continue
        p0, p1 = gt[:, i, :], gt[:, i+1, :]
        seg = p1 - p0
        seg_len = torch.norm(seg, dim=1, keepdim=True) + 1e-8
        t = ((pred - p0.unsqueeze(1)) @ seg.unsqueeze(-1)) / (seg_len.unsqueeze(-1)**2)
        t = torch.clamp(t, 0, 1)
        proj = p0.unsqueeze(1) + t * seg.unsqueeze(1)
        dist = torch.norm(pred - proj, dim=-1)
        weight = torch.exp(-0.5 * (dist/sigma)**2) * seg_mask.unsqueeze(1)
        total_weight += weight
        valid_segments += 1
    loss_centerline = (1 - (total_weight/valid_segments)).squeeze(0).cpu().numpy()

    # Keypoint
    keypoints = gt[:, 1:, :]
    key_mask = mask[:, 1:]
    dist_key = torch.cdist(pred, keypoints)
    valid_mask_exp = key_mask.unsqueeze(1).expand(-1, N_pred, -1)
    dist_key = dist_key * valid_mask_exp + (1.0 - valid_mask_exp) * 1e6
    min_dist_key, _ = dist_key.min(dim=-1)
    loss_keypoint = (1 - torch.exp(-0.5*(min_dist_key/sigma)**2)).squeeze(0).cpu().numpy()

    # Repulsion
    dist_rep = torch.cdist(pred, pred)
    mask_rep = (dist_rep > 1e-3).float()
    repulsion = torch.exp(-(dist_rep/3.0)**2) * mask_rep
    loss_repulsion = repulsion.mean(dim=2).squeeze(0).cpu().numpy()

    # 可视化
    losses = [loss_chamfer, loss_centerline, loss_keypoint, loss_repulsion]
    loss_names = ["Chamfer", "Centerline", "Keypoint", "Repulsion"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i in range(4):
        # 上排 原始障碍
        ax = axes[0, i]
        ax.imshow(grid, cmap="gray_r", origin="lower")
        for rect in env_json["rectangle_obstacles"]:
            x, y, w, h = rect
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1))
        ax.plot(gt_path[:,0], gt_path[:,1], color='lime')
        sc = ax.scatter(pred_points[:,0], pred_points[:,1], c=losses[i], cmap='hot', s=50)
        ax.set_title(f"{loss_names[i]} Loss (原始)"); fig.colorbar(sc, ax=ax)

        # 下排 CAE 重建
        ax = axes[1, i]
        ax.imshow(recon_grid, cmap="gray_r", origin="lower")
        ax.plot(gt_path[:,0], gt_path[:,1], color='lime')
        sc = ax.scatter(pred_points[:,0], pred_points[:,1], c=losses[i], cmap='hot', s=50)
        ax.set_title(f"{loss_names[i]} Loss (CAE)"); fig.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.show()

# ---------------- 主测试函数 ----------------
# ---------------- 主测试函数 ----------------
if __name__ == "__main__":
    env_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]     # 遍历哪些环境

    json_path="data/random_2d/train/envs.json"
    cae_encoder_path="results/cae/encoder_best.pth"
    cae_decoder_path="results/cae/decoder_best.pth"
    generator_path="results/pathgenerator_npz/best_pathgenerator.pth"

    with open(json_path, "r") as f:
        envs = json.load(f)

    # 初始化 CAE
    cae_encoder = Encoder_CNN_2D(input_size=224, latent_dim=256).cuda()
    cae_decoder = Decoder_CNN_2D(feature_map_size=cae_encoder.feature_map_size, latent_dim=256).cuda()
    cae_encoder.load_state_dict(torch.load(cae_encoder_path))
    cae_decoder.load_state_dict(torch.load(cae_decoder_path))
    cae_encoder.eval(); cae_decoder.eval()

    # 初始化 Generator
    generator = PathKeypointGenerator().cuda()
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    # 遍历 env_id
    for env_id in env_ids:
        env_json = envs[env_id]

        # ⭐ 自动生成 sample_ids（根据 JSON 中 paths 的实际数量）
        total_samples = len(env_json["paths"])
        sample_ids = list(range(total_samples))

        print(f"Env {env_id} 共 {total_samples} 个 samples")

        for sample_id in sample_ids:

            # ---------- 安全检查 ----------
            if sample_id >= len(env_json["paths"]):
                print(f"跳过 env {env_id} 中不存在的 sample {sample_id}")
                continue
            if "start" not in env_json or "goal" not in env_json:
                print(f"跳过 env {env_id}（缺少 start/goal 字段）")
                continue
            if sample_id >= len(env_json["start"]) or sample_id >= len(env_json["goal"]):
                print(f"跳过 env {env_id} 的 sample {sample_id}（start/goal 数量不足）")
                continue

            print(f"Processing env {env_id}, sample {sample_id}")

            # ---------- 数据准备 ----------
            gt_path = np.array(env_json["paths"][sample_id], dtype=np.float32)
            grid = render_grid_from_json(env_json, size=224)
            grid_tensor = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0).cuda()

            # ---------- CAE 重建 ----------
            with torch.no_grad():
                latent, enc_feats = cae_encoder(grid_tensor)
                recon = cae_decoder(latent, enc_feats)
            recon_np = recon[0,0].cpu().numpy()

            # ---------- 路径预测 ----------
            with torch.no_grad():
                start = torch.tensor(env_json["start"][sample_id],
                                     dtype=torch.float32).unsqueeze(0).cuda()
                goal  = torch.tensor(env_json["goal"][sample_id],
                                     dtype=torch.float32).unsqueeze(0).cuda()
                pred = generator(latent, start, goal)
                pred_np = pred[0].cpu().numpy()

            # ---------- 可视化 ----------
            visualize_pointwise_losses(env_json, grid, recon_np, gt_path, pred_np)

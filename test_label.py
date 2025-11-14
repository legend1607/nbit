"""
test_pathlabel_visualize_bound.py
---------------------------------
可视化固定 bound 内随机生成点的高斯路径与关键点标签。
使用 PathGeneratorModel_masked_bound 版本。
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from model.PathGeneratorModel import (
    get_path_label_torch, get_keypoint_label_torch, sanitize_label_torch
)

# ===============================
# 参数配置
# ===============================
DATA_PATH = "data/random_2d/val.npz"   # 数据集路径
BOUND_MIN = (-1.0, -1.0)
BOUND_MAX = (220, 220)
SIGMA_RATIO_LINE = 0.05
SIGMA_RATIO_POINT = 0.03
NPOINTS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 主流程
# ===============================
def main():
    # ---- 读取数据 ----
    data = np.load(DATA_PATH, allow_pickle=True)
    paths = data['path']
    starts = data['start']
    goals = data['goal']

    idx = np.random.randint(len(paths))
    expert_path = np.array(paths[idx], dtype=np.float32)
    start = starts[idx]
    goal = goals[idx]
    print(f"Loaded sample {idx}: path length={len(expert_path)}")

    # ---- 固定 bound 范围 ----
    bound_min = torch.tensor(BOUND_MIN, dtype=torch.float32, device=DEVICE)
    bound_max = torch.tensor(BOUND_MAX, dtype=torch.float32, device=DEVICE)
    bound_range = bound_max - bound_min

    # ---- 随机生成预测点 ----
    pred_points = torch.rand((1, NPOINTS, 2), device=DEVICE) * bound_range + bound_min

    # ---- 准备专家路径 ----
    expert_t = torch.tensor(expert_path, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mask = torch.ones((1, expert_t.shape[1]), dtype=torch.float32, device=DEVICE)

    # ---- 固定 sigma ----
    sigma_line = (SIGMA_RATIO_LINE * bound_range).view(1, 1, 2)
    sigma_point = (SIGMA_RATIO_POINT * bound_range).view(1, 1, 2)

    # ---- 计算高斯标签 ----
    path_label = get_path_label_torch(pred_points, expert_t, mask=mask, sigma=sigma_line)
    key_label = get_keypoint_label_torch(pred_points, expert_t[:, 1:, :], mask=mask[:, 1:], sigma=sigma_point)

    combined_label = sanitize_label_torch(path_label + 1.5 * key_label)
    label_np = combined_label.squeeze(0).detach().cpu().numpy()

    # ---- 转换为 numpy 用于绘图 ----
    pred_np = pred_points.squeeze(0).detach().cpu().numpy()
    expert_np = expert_path

    # ---- 打印调试信息 ----
    print("Path label stats:", path_label.min().item(), path_label.max().item(), path_label.mean().item())
    print("Key label stats:", key_label.min().item(), key_label.max().item(), key_label.mean().item())
    print("Combined stats:", label_np.min(), label_np.max(), label_np.mean())

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Gaussian Path & Keypoint Label Visualization (Bounded)")
    ax.set_xlim(BOUND_MIN[0], BOUND_MAX[0])
    ax.set_ylim(BOUND_MIN[1], BOUND_MAX[1])
    ax.set_aspect('equal')

    # 专家路径
    ax.plot(expert_np[:, 0], expert_np[:, 1], 'b-', linewidth=2, label='Expert Path')
    ax.scatter(expert_np[:, 0], expert_np[:, 1], c='blue', s=20)

    # 预测点根据标签强度着色
    sc = ax.scatter(pred_np[:, 0], pred_np[:, 1],
                    c=label_np, cmap='plasma', s=40, alpha=0.85, label='Predicted Points')

    plt.colorbar(sc, ax=ax, label="Gaussian Label Strength")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

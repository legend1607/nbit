"""
PathGeneratorModel.py
--------------------
基于 CAE 障碍信息 + 起止点状态生成路径关键点：
- PathKeypointGenerator: 直接预测 n_points 个路径点
- PathLossGaussian: 高斯加权损失，路径中心线 + 拐点 + 平滑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================
# PathKeypointGenerator
# =========================
class PathKeypointGenerator(nn.Module):
    def __init__(self, obs_latent_dim=128, point_dim=2, latent_dim=128, n_points=128):
        super().__init__()
        self.n_points = n_points
        self.point_dim = point_dim

        input_dim = obs_latent_dim + point_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )

        # 直接输出 n_points 个坐标
        self.coord_head = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.PReLU(),
            nn.Linear(256, 512), nn.PReLU(),
            nn.Linear(512, n_points * point_dim)
        )

    def forward(self, obs_latent, start_state, goal_state):
        """
        obs_latent: (B, obs_latent_dim)
        start_state: (B, point_dim)
        goal_state: (B, point_dim)
        return: path_points (B, n_points, point_dim)
        """
        cond = torch.cat([obs_latent, start_state, goal_state], dim=1)
        latent = self.encoder(cond)
        path_points = self.coord_head(latent).view(-1, self.n_points, self.point_dim)
        return path_points


# =========================
# 高斯标签辅助函数
# =========================
def sanitize_label(label):
    label = np.clip(label, 0, 1)
    return label / (label.sum() + 1e-8)

def get_path_label(pc, path, sigma=None, sigma_ratio=0.05):
    pc = np.asarray(pc)
    path = np.asarray(path)
    if sigma is None:
        range_vec = pc.max(axis=0) - pc.min(axis=0)
        sigma = sigma_ratio * np.maximum(range_vec, 1e-8)
    sigma_eps = np.maximum(sigma, 1e-8)
    path_label = np.zeros(len(pc), dtype=np.float32)

    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i+1]
        seg_vec = p1 - p0
        seg_len2 = np.sum(seg_vec**2)
        if seg_len2 < 1e-12:
            continue
        vec = pc - p0
        t = np.clip(np.sum(vec * seg_vec, axis=1) / seg_len2, 0, 1)
        proj = p0 + t[:, None] * seg_vec
        diff = (pc - proj) / sigma_eps
        dist2 = np.sum(diff**2, axis=1)
        label = np.exp(-0.5 * dist2)
        path_label = np.maximum(path_label, label)
    return sanitize_label(path_label)

def get_keypoint_label(pc, keypoints, sigma=None, sigma_ratio=0.05):
    if len(keypoints) == 0:
        return np.zeros(len(pc), dtype=np.float32)
    pc = np.asarray(pc)
    keypoints = np.asarray(keypoints)
    if sigma is None:
        range_vec = pc.max(axis=0) - pc.min(axis=0)
        if range_vec.max() / max(range_vec.min(), 1e-8) > 3:
            sigma = sigma_ratio * range_vec
        else:
            sigma = sigma_ratio * np.mean(range_vec)
    sigma_eps = np.maximum(sigma, 1e-8)
    diff = (pc[:, None, :] - keypoints[None, :, :]) / sigma_eps
    dist2 = np.sum(diff**2, axis=2)
    label = np.exp(-0.5 * np.min(dist2, axis=1))
    return sanitize_label(label)


# =========================
# PathLossGaussian
# =========================
class PathLossGaussian(nn.Module):
    def __init__(self,
                 sigma_ratio_line=0.05,
                 sigma_ratio_point=0.05,
                 w_line=0.6,
                 w_point=0.4,
                 w_smooth=0.5,
                 alpha_keypoint=2.0):
        super().__init__()
        self.sigma_ratio_line = sigma_ratio_line
        self.sigma_ratio_point = sigma_ratio_point
        self.w_line = w_line
        self.w_point = w_point
        self.w_smooth = w_smooth
        self.alpha_keypoint = alpha_keypoint
        self.eps = 1e-8

    def forward(self, path_points, expert_path, expert_keypoints):
        """
        path_points: (B, n_points, 2)
        expert_path: (L, 2)
        expert_keypoints: (K, 2)
        """
        B, Np, D = path_points.shape
        device = path_points.device

        # 准备候选点 pc，用 path_points 本身
        pc = path_points.detach().cpu().numpy().reshape(-1, D)

        # 计算路径中心线和关键点高斯标签
        path_label = get_path_label(pc, expert_path, sigma_ratio=self.sigma_ratio_line)
        keypoint_label = get_keypoint_label(pc, expert_keypoints, sigma_ratio=self.sigma_ratio_point)

        combined_label = path_label + self.alpha_keypoint * keypoint_label
        combined_label = sanitize_label(combined_label)
        combined_label = torch.tensor(combined_label, dtype=path_points.dtype, device=device).view(B, Np)

        # -------------------------------
        # 距离加权损失（MSE + 高斯权重）
        # -------------------------------
        diff = path_points  # 这里可以用 expert_path 投影到 n_points 或插值点
        loss_path = ((diff**2) * combined_label.unsqueeze(-1)).mean()

        # -------------------------------
        # 平滑损失
        # -------------------------------
        loss_smooth = ((path_points[:, 1:, :] - path_points[:, :-1, :])**2).mean()

        total_loss = loss_path + self.w_smooth * loss_smooth
        return total_loss, loss_path, loss_smooth

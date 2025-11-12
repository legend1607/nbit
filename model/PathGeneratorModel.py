"""
PathGeneratorModel_masked.py
----------------------------
基于 CAE 障碍信息 + 起止点状态生成路径关键点
- PathKeypointGenerator: 直接预测 n_points 个路径点
- PathKeypointLoss: 高斯加权损失，路径中心线 + 拐点 + 平滑
- 支持 mask 忽略填充点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# PathKeypointGenerator
# =========================
class PathKeypointGenerator(nn.Module):
    def __init__(self, obs_latent_dim=128, point_dim=2, latent_dim=128, n_points=128):
        super().__init__()
        self.n_points = n_points
        self.point_dim = point_dim

        input_dim = obs_latent_dim + point_dim * 2  # obs + start + goal
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )

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
# 标签生成函数（高斯） 支持 mask
# =========================
def sanitize_label_torch(label):
    label = torch.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
    label = torch.clamp(label, 0.0, 1.0)
    if label.sum(dim=1, keepdim=True).max() > 0:
        label = label / (label.sum(dim=1, keepdim=True) + 1e-8)
    label[label < 1e-3] = 0.0
    return label

def get_path_label_torch(pc, path, sigma_ratio=0.05, mask=None):
    """
    高斯路径标签
    pc: (B, N, 2)
    path: (B, L, 2)
    mask: (B, L)  1=有效，0=填充
    """
    B, N, D = pc.shape
    _, L, _ = path.shape

    range_vec = pc.max(dim=1).values - pc.min(dim=1).values
    sigma = sigma_ratio * range_vec.unsqueeze(1)
    sigma_eps = torch.clamp(sigma, min=1e-8)

    label = torch.zeros(B, N, device=pc.device)

    for i in range(L - 1):
        if mask is not None:
            valid = mask[:, i] * mask[:, i+1]  # 当前段是否有效
            if valid.sum() == 0:
                continue
        else:
            valid = torch.ones(B, device=pc.device)

        p0, p1 = path[:, i, :], path[:, i + 1, :]
        seg_vec = p1 - p0
        seg_len2 = (seg_vec ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12)
        t = ((pc - p0.unsqueeze(1)) * seg_vec.unsqueeze(1)).sum(dim=2) / seg_len2
        t = torch.clamp(t, 0, 1)
        proj = p0.unsqueeze(1) + t.unsqueeze(2) * seg_vec.unsqueeze(1)
        dist2 = ((pc - proj) / sigma_eps) ** 2
        dist2 = dist2.sum(dim=2)
        label = torch.maximum(label, torch.exp(-0.5 * dist2) * valid.unsqueeze(1))

    return sanitize_label_torch(label)

def get_keypoint_label_torch(pc, keypoints, mask=None, sigma_ratio=0.05):
    """
    高斯关键点标签
    pc: (B, N, 2)
    keypoints: (B, K, 2)
    mask: (B, K) 1=有效
    """
    B, N, D = pc.shape
    _, K, _ = keypoints.shape
    if K == 0:
        return torch.zeros(B, N, device=pc.device)

    range_vec = pc.max(dim=1).values - pc.min(dim=1).values
    sigma = sigma_ratio * range_vec.unsqueeze(1)
    sigma_eps = torch.clamp(sigma, min=1e-8)

    diff = (pc.unsqueeze(2) - keypoints.unsqueeze(1)) / sigma_eps
    dist2 = (diff ** 2).sum(dim=3)
    label = torch.exp(-0.5 * dist2.min(dim=2).values)

    if mask is not None:
        mask_any = mask.sum(dim=1) > 0
        label[~mask_any] = 0.0

    return sanitize_label_torch(label)


# =========================
# PathKeypointLoss 支持 mask
# =========================
class PathKeypointLoss(nn.Module):
    """
    路径关键点联合损失：
    - 路径中心线对齐（高斯）
    - 拐点增强
    - 平滑正则化
    - 支持 mask 忽略填充点
    """

    def __init__(self,
                 sigma_ratio_line=0.05,
                 sigma_ratio_point=0.03,
                 w_line=1.0,
                 w_point=1.5,
                 w_smooth=0.5):
        super().__init__()
        self.sigma_ratio_line = sigma_ratio_line
        self.sigma_ratio_point = sigma_ratio_point
        self.w_line = w_line
        self.w_point = w_point
        self.w_smooth = w_smooth

    def forward(self, pred_path, expert_path, masks=None):
        """
        pred_path: (B, N, 2)
        expert_path: (B, L, 2)
        masks: (B, L)  1=有效路径，0=填充
        """
        B, N, D = pred_path.shape

        # 关键点：去掉起点
        keypoints = expert_path[:, 1:, :]
        key_mask = masks[:, 1:] if masks is not None else None

        # 高斯标签
        path_label = get_path_label_torch(pred_path, expert_path, self.sigma_ratio_line, mask=masks)
        keypoint_label = get_keypoint_label_torch(pred_path, keypoints, mask=key_mask, sigma_ratio=self.sigma_ratio_point)

        # 标签融合
        combined_label = self.w_line * path_label + self.w_point * keypoint_label
        combined_label = combined_label / (combined_label.sum(dim=1, keepdim=True) + 1e-8)

        # 距离加权损失
        if masks is not None:
            effective_len = masks.sum(dim=1).clamp_min(1)  # 避免除零
            dist = torch.cdist(pred_path, expert_path).min(dim=2).values
            dist = dist * masks
            loss_align = (dist.sum(dim=1) / effective_len).mean()
        else:
            dist = torch.cdist(pred_path, expert_path).min(dim=2).values
            loss_align = (dist * combined_label).mean()

        # 平滑损失
        loss_smooth = ((pred_path[:, 1:, :] - pred_path[:, :-1, :]) ** 2).mean()

        total_loss = loss_align + self.w_smooth * loss_smooth
        return total_loss, loss_align, loss_smooth

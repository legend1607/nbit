import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# CAE 编码障碍信息
# =========================
class ObstacleCAE(nn.Module):
    """
    输入: 障碍信息 (B, n_obs, 4)
    输出: 障碍 latent 特征 (B, latent_dim)
    """
    def __init__(self, n_obs=10, obs_dim=4, latent_dim=128):
        super().__init__()
        input_dim = n_obs * obs_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.PReLU(),
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.PReLU(),
            nn.Linear(128, 256), nn.PReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, obs):
        B = obs.shape[0]
        x = obs.view(B, -1)
        latent = self.encoder(x)
        recon = self.decoder(latent)
        recon = recon.view(B, obs.shape[1], obs.shape[2])
        return latent, recon


# =========================
# 双头路径+关键点生成模型（ 条件）
# =========================
class PathKeypointGenerator(nn.Module):
    """
    输入:
        - obs_latent: CAE 编码的障碍 latent (B, latent_dim)
        - start_state: 起始状态 (B, point_dim)
        - goal_state: 目标状态 (B, point_dim)
    输出:
        - path_points: 路径点云 (B, N_points, point_dim)
    """
    def __init__(self, obs_latent_dim=128, point_dim=2, latent_dim=128, n_points=128):
        super().__init__()
        self.n_points = n_points
        self.point_dim = point_dim

        # 条件输入 latent
        input_dim = obs_latent_dim + point_dim*2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )

        # 路径点生成
        self.coord_head = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.PReLU(),
            nn.Linear(256, 512), nn.PReLU(),
            nn.Linear(512, n_points * point_dim)
        )


    def forward(self, obs_latent, start_state, goal_state):
        cond = torch.cat([obs_latent, start_state, goal_state], dim=1)
        latent = self.encoder(cond)

        path_points = self.coord_head(latent).view(-1, self.n_points, self.point_dim)
        return path_points


# =========================
# 损失函数
# =========================
class PathLossTwoGaussWeighted(nn.Module):
    """
    Fully vectorized Path loss with two Gaussian components (weighted) using per-dimension sigma:
    1. Gaussian along path line segments
    2. Gaussian at path points
    Plus path smoothness and CAE reconstruction.
    """
    def __init__(self, sigma_ratio_line=0.05, sigma_ratio_point=0.05,
                 w_line=0.6, w_point=0.4, w_smooth=0.8, w_cae=0.5):
        super().__init__()
        self.sigma_ratio_line = sigma_ratio_line
        self.sigma_ratio_point = sigma_ratio_point
        self.w_line = w_line
        self.w_point = w_point
        self.w_smooth = w_smooth
        self.w_cae = w_cae

    def forward(self, path_points, path_mask, path, obs_recon, obs):
        """
        path_points: (B, Np, D)
        path_mask: (B, Np) bool
        path: list of tensors [(Ng_i, D), ...] variable length
        obs_recon, obs: (B, N_obs, obs_dim)
        """
        B, Np, D = path_points.shape
        device = path_points.device

        # -------------------------------
        # Pad target paths to max length
        # -------------------------------
        Ng_max = max([p.shape[0] for p in path])
        tgt_padded = torch.zeros(B, Ng_max, D, device=device)
        tgt_mask = torch.zeros(B, Ng_max, dtype=torch.bool, device=device)
        for b, p in enumerate(path):
            n = p.shape[0]
            tgt_padded[b, :n, :] = p.to(device)
            tgt_mask[b, :n] = 1

        # -------------------------------
        # Compute per-dimension sigma
        # -------------------------------
        max_vals, _ = path_points.max(dim=1)  # (B, D)
        min_vals, _ = path_points.min(dim=1)  # (B, D)
        sigma_line_eps = (self.sigma_ratio_line * (max_vals - min_vals)).clamp_min(1e-8)   # (B, D)
        sigma_point_eps = (self.sigma_ratio_point * (max_vals - min_vals)).clamp_min(1e-8) # (B, D)

        # -------------------------------
        # Gaussian along line segments
        # -------------------------------
        p0 = tgt_padded[:, :-1, :]          # (B, Ng_max-1, D)
        p1 = tgt_padded[:, 1:, :]           # (B, Ng_max-1, D)
        seg_vec = p1 - p0                   # (B, Ng_max-1, D)
        seg_len2 = (seg_vec**2).sum(dim=2).clamp_min(1e-8)  # (B, Ng_max-1)
        seg_mask = tgt_mask[:, :-1] & tgt_mask[:, 1:]       # (B, Ng_max-1)

        path_mask_exp = path_mask.unsqueeze(2)             # (B, Np, 1)
        pc = path_points * path_mask_exp.float()           # (B, Np, D)

        # Project all points onto all segments
        t = torch.einsum('bnd,bmd->bnm', pc - p0, seg_vec) / seg_len2.unsqueeze(1)  # (B, Np, Ng_max-1)
        t = t.clamp(0, 1)
        proj = p0.unsqueeze(1) + t.unsqueeze(3) * seg_vec.unsqueeze(1)              # (B, Np, Ng_max-1, D)

        # Apply per-dimension sigma
        diff = (pc.unsqueeze(2) - proj) / sigma_line_eps.view(B, 1, 1, D)           # (B, Np, Ng_max-1, D)
        dist2 = (diff**2).sum(dim=3)                                                # (B, Np, Ng_max-1)
        dist2 = dist2 + (~seg_mask).float().unsqueeze(1) * 1e6                       # mask invalid segments
        label_line = torch.exp(-0.5 * dist2).max(dim=2)[0]                           # (B, Np)

        # -------------------------------
        # Gaussian at path points
        # -------------------------------
        diff_point = (pc.unsqueeze(2) - tgt_padded.unsqueeze(1)) / sigma_point_eps.view(B, 1, 1, D)  # (B, Np, Ng_max, D)
        dist2_point = (diff_point**2).sum(dim=3)                                                  # (B, Np, Ng_max)
        dist2_point = dist2_point + (~tgt_mask).float().unsqueeze(1) * 1e6                         # mask padded points
        label_point = torch.exp(-0.5 * dist2_point.min(dim=2)[0])                                   # (B, Np)

        # -------------------------------
        # Combine Gaussian losses
        # -------------------------------
        label_combined = torch.clamp(self.w_line * label_line + self.w_point * label_point, 0.0, 1.0)
        loss_path = ((1 - label_combined) * path_mask.float()).sum() / path_mask.sum()

        # -------------------------------
        # Path smoothness
        # -------------------------------
        diffs = path_points[:, 1:, :] - path_points[:, :-1, :]
        mask_diff = path_mask[:, 1:] & path_mask[:, :-1]
        loss_smooth = (diffs**2).sum(dim=2)
        loss_smooth = (loss_smooth * mask_diff).sum(dim=1) / (mask_diff.sum(dim=1) + 1e-8)
        loss_smooth = loss_smooth.mean()

        # -------------------------------
        # CAE reconstruction
        # -------------------------------
        loss_cae = F.mse_loss(obs_recon, obs)

        # -------------------------------
        # Total loss
        # -------------------------------
        total_loss = self.w_smooth * loss_smooth + loss_path + self.w_cae * loss_cae

        return total_loss, loss_path, loss_smooth, loss_cae

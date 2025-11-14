# =========================
# PathGeneratorModel.py
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# PathKeypointGenerator
# =========================
class PathKeypointGenerator(nn.Module):
    """
    输入: obs_latent + start + goal
    输出: n_points 无序散点路径关键点
    """
    def __init__(self, obs_latent_dim=256, point_dim=2, latent_dim=256, n_points=128,
                 bound_min=(0.0, 0.0), bound_max=(224.0, 224.0)):
        super().__init__()
        self.n_points = n_points
        self.point_dim = point_dim
        self.bound_min = torch.tensor(bound_min, dtype=torch.float32)
        self.bound_max = torch.tensor(bound_max, dtype=torch.float32)
        self.bound_range = self.bound_max - self.bound_min

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
        cond = torch.cat([obs_latent, start_state, goal_state], dim=1)
        latent = self.encoder(cond)
        path_points = self.coord_head(latent).view(-1, self.n_points, self.point_dim)

        # 映射到固定 bounds
        device = path_points.device
        bound_min = self.bound_min.to(device)
        bound_max = self.bound_max.to(device)
        bound_range = self.bound_range.to(device)

        path_points = torch.tanh(path_points) * (bound_range / 2) + (bound_min + bound_max) / 2
        return path_points


# =========================
# PointGeneratorLoss
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F


class PathGenerationLoss(nn.Module):
    """
    无序路径散点生成损失（Path Keypoint Generation）
    包含：
      - Chamfer: 散点靠近路径区域
      - Centerline: 散点靠近路径中心线
      - Keypoint: 散点覆盖路径关键点附近
      - Repulsion: 散点不重叠
      - Bandwidth: 控制散点距离路径的平均带宽
    """
    def __init__(self,
                 sigma=4.0,
                 target_bandwidth=5.0,
                 w_chamfer=0.5,
                 w_centerline=1.0,
                 w_keypoint=1.0,
                 w_repulsion=0.1,
                 w_bandwidth=0.05):
        super().__init__()
        self.sigma = sigma
        self.target_bandwidth = target_bandwidth
        self.w_chamfer = w_chamfer
        self.w_centerline = w_centerline
        self.w_keypoint = w_keypoint
        self.w_repulsion = w_repulsion
        self.w_bandwidth = w_bandwidth

    # ---------------------------
    # Chamfer Loss
    # ---------------------------
    def chamfer_loss(self, pred, gt, mask):
        B, N_pred, _ = pred.shape
        # 跳过起点，从第二个点开始
        gt_skip_start = gt[:, 1:, :]        # [B, N_gt-1, 2]
        mask_skip_start = mask[:, 1:]      # [B, N_gt-1]

        N_gt = gt_skip_start.shape[1]
        dist = torch.cdist(pred, gt_skip_start)  # [B, N_pred, N_gt]
        valid_mask = mask_skip_start.unsqueeze(1).expand(-1, N_pred, -1).float()  # 转 float
        dist = dist * valid_mask + (1.0 - valid_mask) * 1e6

        min_pred2gt, _ = dist.min(dim=2)
        min_gt2pred, _ = dist.min(dim=1)
        loss_pred = min_pred2gt.mean()
        loss_gt = (min_gt2pred * mask_skip_start).sum() / mask_skip_start.sum()
        return loss_pred + loss_gt
    # ---------------------------
    # Centerline Loss
    # ---------------------------
    def centerline_loss(self, pred, gt, mask):
        B, N_pred, _ = pred.shape
        M = gt.shape[1]

        total_loss = 0.0
        valid_segments = 0

        for i in range(M - 1):
            seg_mask = (mask[:, i] * mask[:, i + 1]).float()  # ✅ 转 float
            if seg_mask.sum() == 0:
                continue

            p0, p1 = gt[:, i, :], gt[:, i + 1, :]
            seg = p1 - p0
            seg_len = torch.norm(seg, dim=1, keepdim=True) + 1e-8

            t = ((pred - p0.unsqueeze(1)) @ seg.unsqueeze(-1)) / (seg_len.unsqueeze(-1) ** 2)
            t = torch.clamp(t, 0, 1)
            proj = p0.unsqueeze(1) + t * seg.unsqueeze(1)
            dist = torch.norm(pred - proj, dim=-1)

            weight = torch.exp(-0.5 * (dist / self.sigma) ** 2)
            weight = weight * seg_mask.unsqueeze(1)
            total_loss += weight.sum() / (seg_mask.sum() * N_pred)
            valid_segments += 1

        if valid_segments == 0:
            return torch.tensor(0.0, device=pred.device)
        return 1.0 - total_loss / valid_segments
    # ---------------------------
    # Keypoint Loss
    # ---------------------------
    def keypoint_loss(self, pred, gt, mask):
        keypoints = gt[:, 1:, :]
        key_mask = mask[:, 1:]
        valid_mask = (key_mask > 0).float()  # ✅ 转 float

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        dist = torch.cdist(pred, keypoints)  # [B, N_pred, N_key]
        valid_mask_exp = valid_mask.unsqueeze(1).expand(-1, pred.shape[1], -1).float()  # ✅ 转 float
        dist = dist * valid_mask_exp + (1.0 - valid_mask_exp) * 1e6

        min_dist, _ = dist.min(dim=-1)
        weights = torch.exp(-0.5 * (min_dist / self.sigma) ** 2)
        return 1.0 - weights.mean()
    # ---------------------------
    # Repulsion Loss
    # ---------------------------
    def repulsion_loss(self, pred, radius=3.0):
        dist = torch.cdist(pred, pred)
        mask = (dist > 1e-3).float()
        repulsion = torch.exp(- (dist / radius) ** 2) * mask
        return repulsion.mean()

    # ---------------------------
    # Bandwidth Loss
    # ---------------------------
    def bandwidth_loss(self, pred, gt, mask):
        dist = torch.cdist(pred, gt)
        mask_exp = mask.unsqueeze(1).expand(-1, pred.shape[1], -1).float()  # ✅ 转 float
        dist = dist * mask_exp + (1.0 - mask_exp) * 1e6
        min_dist, _ = dist.min(dim=2)
        mean_bw = min_dist.mean()
        return (mean_bw - self.target_bandwidth).pow(2)
    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, pred_points, gt_path, mask):
        L_chamfer = self.chamfer_loss(pred_points, gt_path, mask)
        L_center = self.centerline_loss(pred_points, gt_path, mask)
        L_key = self.keypoint_loss(pred_points, gt_path, mask)
        L_repulsion = self.repulsion_loss(pred_points)
        L_band = self.bandwidth_loss(pred_points, gt_path, mask)

        total = (
            self.w_chamfer * L_chamfer +
            self.w_centerline * L_center +
            self.w_keypoint * L_key +
            self.w_repulsion * L_repulsion +
            self.w_bandwidth * L_band
        )

        loss_dict = {
            "total": total.item(),
            "chamfer": L_chamfer.item(),
            "centerline": L_center.item(),
            "keypoint": L_key.item(),
            "repulsion": L_repulsion.item(),
            "bandwidth": L_band.item(),
        }

        return total, loss_dict

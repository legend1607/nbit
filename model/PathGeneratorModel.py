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
    Path Generation Loss for unordered point cloud prediction along a GT path.

    Features:
    - Centerline density loss: points near GT path centerline are dense
    - Keypoint density loss: points around GT middle points (folds) are dense
    - Repulsion loss: avoid points stacking
    - Bandwidth loss: control overall point cloud spread
    """

    def __init__(self, sigma_center=3.0, sigma_key=3.0, min_dist=0.03, target_bandwidth=5.0,
                 w_centerline=1.0, w_keypoint=1.0, w_repulsion=0.1, w_bandwidth=0.05):
        super().__init__()
        self.sigma_center = sigma_center
        self.sigma_key = sigma_key
        self.min_dist = min_dist
        self.target_bandwidth = target_bandwidth

        self.w_centerline = w_centerline
        self.w_keypoint = w_keypoint
        self.w_repulsion = w_repulsion
        self.w_bandwidth = w_bandwidth

    # ----------------------------
    # Centerline Density Loss
    # ----------------------------
    def centerline_density_loss(self, pred, gt, mask):
        B, N_pred, _ = pred.shape
        M = gt.shape[1]
        all_dists = []

        for i in range(M - 1):
            p0 = gt[:, i:i+1, :]
            p1 = gt[:, i+1:i+2, :]
            seg = p1 - p0
            seg_len2 = (seg ** 2).sum(dim=-1, keepdim=True) + 1e-8

            # projection onto segment
            t = ((pred - p0) * seg).sum(dim=-1, keepdim=True) / seg_len2
            t = torch.clamp(t, 0.0, 1.0)
            proj = p0 + t * seg

            dist = (pred - proj).norm(dim=-1)
            all_dists.append(dist.unsqueeze(-1))

        dists = torch.cat(all_dists, dim=-1)
        min_dist, _ = dists.min(dim=-1)
        weights = torch.exp(-0.5 * (min_dist / self.sigma_center)**2)
        loss = 1 - weights.mean()
        return loss

    # ----------------------------
    # Keypoint Density Loss
    # ----------------------------
    def keypoint_density_loss(self, pred, gt, mask):
        keypoints = gt[:, 1:, :]
        key_mask = mask[:, 1:]
        B, N_pred, _ = pred.shape
        K = keypoints.shape[1]

        if key_mask.sum() == 0:
            return pred.new_tensor(0.0)

        dist = torch.cdist(pred, keypoints)  # [B, N_pred, K]
        weights = torch.exp(-(dist**2)/(2*self.sigma_key**2))

        # coverage per keypoint
        gt_coverage = weights.sum(dim=1)  # [B,K]
        loss = 1 - gt_coverage / (gt_coverage.max(dim=1, keepdim=True)[0] + 1e-8)
        loss = (loss * key_mask).sum() / (key_mask.sum() + 1e-8)
        return loss

    # ----------------------------
    # Repulsion Loss
    # ----------------------------
    def repulsion_loss(self, pred):
        B, N, _ = pred.shape
        dist = torch.cdist(pred, pred) + 1e-6
        mask = torch.eye(N, device=pred.device).bool().unsqueeze(0)
        dist = dist.masked_fill(mask, 1e6)
        repel = F.relu(self.min_dist - dist)
        return repel.mean()

    # ----------------------------
    # Bandwidth Loss
    # ----------------------------
    def bandwidth_loss(self, pred, gt, mask):
        B, N_pred, _ = pred.shape
        dist = torch.cdist(pred, gt)
        mask_exp = mask.unsqueeze(1).expand(-1, N_pred, -1).float()
        dist = dist * mask_exp + (1 - mask_exp) * 1e6
        min_dist, _ = dist.min(dim=2)
        mean_bw = min_dist.mean()
        return (mean_bw - self.target_bandwidth) ** 2

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, pred_points, gt_path, mask):
        L_center = self.centerline_density_loss(pred_points, gt_path, mask)
        L_key = self.keypoint_density_loss(pred_points, gt_path, mask)
        L_rep = self.repulsion_loss(pred_points)
        L_band = self.bandwidth_loss(pred_points, gt_path, mask)

        total = (
            self.w_centerline * L_center +
            self.w_keypoint * L_key +
            self.w_repulsion * L_rep +
            self.w_bandwidth * L_band
        )

        return {
            "loss": total,
            "centerline": L_center,
            "keypoint": L_key,
            "repulsion": L_rep,
            "bandwidth": L_band
        }

"""
convert_json_to_npz_with_grids.py
---------------------------------
从 envs.json 读取环境、起点终点、专家路径，
为每条路径生成对应的环境栅格，打包为统一 npz 文件。
输出结构：每个样本包含 grid / start / goal / path。
"""

import os
import json
import time
import numpy as np
from os.path import join
from tqdm import tqdm
import open3d as o3d
# ===============================
# 路径 soft 标签生成
# ===============================
def get_path_label(pc, path, s_goal=None, sigma=None, sigma_ratio=0.05):
    pc = np.asarray(pc)
    path = np.asarray(path)
    if s_goal is None:
        s_goal = path[-1]

    if sigma is None:
        range_vec = pc.max(axis=0) - pc.min(axis=0)
        sigma = sigma_ratio * range_vec
    sigma = np.asarray(sigma)
    sigma_eps = np.maximum(sigma, 1e-8)

    path_label = np.zeros(len(pc), dtype=np.float32)

    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue
        vec = pc - p0
        t = np.clip(np.sum(vec * seg_vec, axis=1) / (seg_len**2), 0, 1)
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

    sigma = np.asarray(sigma)
    sigma_eps = np.maximum(sigma, 1e-8)

    diff = (pc[:, None, :] - keypoints[None, :, :]) / sigma_eps
    dist2 = np.sum(diff**2, axis=2)
    label = np.exp(-0.5 * np.min(dist2, axis=1))

    return sanitize_label(label)

# ===============================
# Farthest Point Sampling (FPS)
# ===============================
def sample_fps_point_cloud(env, n_points, oversample_scale=5):
    n_oversample = n_points * oversample_scale
    points = np.array([env.sample_empty_points() for _ in range(n_oversample)])
    points_3d = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd_fps = pcd.farthest_point_down_sample(num_samples=n_points)
    return np.asarray(pcd_fps.points)[:, :2]

# ===============================
# 栅格化函数
# ===============================
def env_to_grid(env_dict, resolution=1.0):
    """将 JSON 环境转为二值栅格"""
    width, height = env_dict["env_dims"]
    w_cells, h_cells = int(width / resolution), int(height / resolution)
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)

    # 矩形障碍
    for rx, ry, rw, rh in env_dict.get("rectangle_obstacles", []):
        x1, y1 = int(rx / resolution), int(ry / resolution)
        x2, y2 = int((rx + rw) / resolution), int((ry + rh) / resolution)
        grid[y1:y2, x1:x2] = 1

    return grid


# ===============================
# 安全数据清洗
# ===============================
def sanitize_label(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    if np.max(x) > 0:
        x /= np.max(x)
    x[x < 1e-3] = 0.0
    return x.astype(np.float32).copy()


# ===============================
# 主函数
# ===============================
def convert_json_to_npz_with_grids(env_type="random_2d", resolution=1.0):
    from environment.random_2d_env import Random2DEnv as Env
    dataset_dir = join("data", env_type)
    os.makedirs(dataset_dir, exist_ok=True)
    n_points = 2048

    for mode in ["train","val"]:
        env_json_path = join(dataset_dir, mode, "envs.json")
        if not os.path.exists(env_json_path):
            print(f"⚠️ 跳过 {mode} (未找到 {env_json_path})")
            continue

        with open(env_json_path, "r") as f:
            env_list = json.load(f)

        samples = {
            "token": [],
            "grid": [],
            "pc": [],
            "start": [],
            "goal": [],
            "label":[],
            "path":[]
        }

        print(f"📦 开始转换 {mode} 数据集，共 {len(env_list)} 个环境...")
        start_time = time.time()

        for env_dict in tqdm(env_list):
            env = Env(env_dict)
            env_idx = env_dict["env_idx"]
            grid = env_to_grid(env_dict, resolution=resolution).astype(np.float32)

            # 每个环境可能有多个样本 (start-goal-path)
            for sample_idx, (s_start, s_goal) in enumerate(zip(env_dict["start"], env_dict["goal"])):
                token = f"{mode}-{env_idx}_{sample_idx}"
                path = np.array(env_dict["paths"][sample_idx], dtype=np.float32)

                pc = sample_fps_point_cloud(env, n_points, oversample_scale=5)
                path_label = get_path_label(pc, path)
                keypoints = path[1:]
                keypoint_label = get_keypoint_label(pc, keypoints)
                label=path_label+keypoint_label

                samples["token"].append(token)
                samples["grid"].append(grid)             # 每条样本都附带相同环境的grid
                samples["start"].append(np.array(s_start, dtype=np.float32))
                samples["goal"].append(np.array(s_goal, dtype=np.float32))
                samples["pc"].append(pc)
                samples["label"].append(label)
                samples["path"].append(path)

        # 打包保存为 npz
        np.savez_compressed(
            join(dataset_dir, f"{mode}.npz"),
            token=np.array(samples["token"]),
            grid=np.stack(samples["grid"], axis=0),
            start=np.stack(samples["start"], axis=0),
            goal=np.stack(samples["goal"], axis=0),
            pc=np.stack(samples["pc"], axis=0),          
            label=np.stack(samples["label"], axis=0),     
            path=np.array(samples["path"], dtype=object),
        )

        elapsed = (time.time() - start_time) / 60
        print(f"✅ [{mode}] 已保存 {len(samples['token'])} 条样本 -> {mode}.npz  ({elapsed:.1f} min)")


# ===============================
# 入口
# ===============================
if __name__ == "__main__":
    convert_json_to_npz_with_grids(env_type="random_2d", resolution=1.0)

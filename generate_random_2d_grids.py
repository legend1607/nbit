"""
generate_random_2d_grids_with_env.py
------------------------------------
使用 Random2DEnv 生成随机 2D 环境，并保存为 numpy 栅格 (npy)。
支持自定义障碍数量和尺寸范围。
"""

import os
import numpy as np
from tqdm import tqdm
from environment.random_2d_env import Random2DEnv
import random

# ----------------------------
# 将 Random2DEnv 转为栅格
# ----------------------------
def env_to_grid(env, resolution=0.1):
    width, height = env.bound[1]
    w_cells = int(width / resolution)
    h_cells = int(height / resolution)
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)

    for rx, ry, rw, rh in env.rect_obstacles:
        x1 = int(rx / resolution)
        y1 = int(ry / resolution)
        x2 = int((rx + rw) / resolution)
        y2 = int((ry + rh) / resolution)
        grid[y1:y2, x1:x2] = 1

    return np.flipud(grid)

# ----------------------------
# 随机生成障碍
# ----------------------------
def add_random_obstacles(env, num_boxes_range=(2, 10), box_size_range=(2, 5)):
    num_boxes = random.randint(*num_boxes_range)
    for _ in range(num_boxes):
        w = random.uniform(*box_size_range)
        h = random.uniform(*box_size_range)
        x = random.uniform(0, env.bound[1][0] - w)
        y = random.uniform(0, env.bound[1][1] - h)
        env.rect_obstacles.append([x, y, w, h])

# ----------------------------
# 生成数据集
# ----------------------------
def generate_dataset(
    num_envs=100,
    env_dims=(20, 20),
    resolution=0.1,
    num_boxes_range=(2, 10),
    box_size_range=(10, 24),
    save_path="grids.npy"
):
    grids = []

    for _ in tqdm(range(num_envs), desc="Generating environments"):
        env = Random2DEnv({"env_dims": env_dims, "rectangle_obstacles": []})
        add_random_obstacles(env, num_boxes_range, box_size_range)
        grid = env_to_grid(env, resolution)
        grids.append(grid)

    grids_array = np.stack(grids).astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, grids_array, allow_pickle=False)
    print(f"✅ Saved {num_envs} grids to {save_path} (shape: {grids_array.shape})")

# ----------------------------
# 入口
# ----------------------------
if __name__ == "__main__":
    generate_dataset(
        num_envs=2000,
        env_dims=(224, 224),
        resolution=1,
        num_boxes_range=(2, 15),
        box_size_range=(5, 24),
        save_path="data/random_2d/grids.npy"
    )

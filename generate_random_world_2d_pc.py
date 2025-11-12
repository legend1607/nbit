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

    # 圆形障碍
    for cx, cy, r in env_dict.get("circle_obstacles", []):
        cx_i, cy_i, r_i = int(cx / resolution), int(cy / resolution), int(r / resolution)
        y, x = np.ogrid[-cy_i:h_cells - cy_i, -cx_i:w_cells - cx_i]
        mask = x*x + y*y <= r_i*r_i
        grid[mask] = 1

    return np.flipud(grid)


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
    dataset_dir = join("data", env_type)
    os.makedirs(dataset_dir, exist_ok=True)

    for mode in ["train", "val", "test"]:
        env_json_path = join(dataset_dir, mode, "envs.json")
        if not os.path.exists(env_json_path):
            print(f"⚠️ 跳过 {mode} (未找到 {env_json_path})")
            continue

        with open(env_json_path, "r") as f:
            env_list = json.load(f)

        samples = {
            "token": [],
            "grid": [],
            "start": [],
            "goal": [],
            "path": []
        }

        print(f"📦 开始转换 {mode} 数据集，共 {len(env_list)} 个环境...")
        start_time = time.time()

        for env_dict in tqdm(env_list):
            env_idx = env_dict["env_idx"]
            grid = env_to_grid(env_dict, resolution=resolution).astype(np.float32)

            # 每个环境可能有多个样本 (start-goal-path)
            for sample_idx, (s_start, s_goal) in enumerate(zip(env_dict["start"], env_dict["goal"])):
                token = f"{mode}-{env_idx}_{sample_idx}"
                path = np.array(env_dict["paths"][sample_idx], dtype=np.float32)

                samples["token"].append(token)
                samples["grid"].append(grid)             # 每条样本都附带相同环境的grid
                samples["start"].append(np.array(s_start, dtype=np.float32))
                samples["goal"].append(np.array(s_goal, dtype=np.float32))
                samples["path"].append(path)

        # 打包保存为 npz
        np.savez_compressed(
            join(dataset_dir, f"{mode}.npz"),
            token=np.array(samples["token"]),
            grid=np.stack(samples["grid"], axis=0),
            start=np.stack(samples["start"], axis=0),
            goal=np.stack(samples["goal"], axis=0),
            path=np.array(samples["path"], dtype=object),  # 路径长度不一，需保存为 object
        )

        elapsed = (time.time() - start_time) / 60
        print(f"✅ [{mode}] 已保存 {len(samples['token'])} 条样本 -> {mode}.npz  ({elapsed:.1f} min)")


# ===============================
# 入口
# ===============================
if __name__ == "__main__":
    convert_json_to_npz_with_grids(env_type="random_2d", resolution=1.0)

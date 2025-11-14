import json
import numpy as np
import matplotlib.pyplot as plt
from environment.random_2d_env import Random2DEnv

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

# -----------------------------
# 可视化函数
# -----------------------------
def visualize_env(env_dict, resolution=1.0):
    width, height = env_dict["env_dims"]

    # 原始障碍地图
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Obstacle Map (from JSON)")
    plt.xlim(0, width)
    plt.ylim(0, height)

    # 绘制矩形障碍
    for rx, ry, rw, rh in env_dict.get("rectangle_obstacles", []):
        rect = plt.Rectangle((rx, ry), rw, rh, color='black')
        plt.gca().add_patch(rect)

    # 绘制起点终点
    for s_start, s_goal in zip(env_dict["start"], env_dict["goal"]):
        plt.scatter(*s_start, color='green', marker='o', s=50, label='Start')
        plt.scatter(*s_goal, color='red', marker='x', s=50, label='Goal')

    plt.gca().set_aspect('equal')
    
    # 栅格地图
    plt.subplot(1,2,2)
    grid = env_to_grid(env_dict, resolution)
    plt.title("Rasterized Grid Map")
    plt.imshow(grid, origin='lower', cmap='gray')

    # 绘制起点终点（栅格坐标映射）
    for s_start, s_goal in zip(env_dict["start"], env_dict["goal"]):
        plt.scatter(*s_start, color='green', marker='o', s=50, label='Start')
        plt.scatter(*s_goal, color='red', marker='x', s=50, label='Goal')

    plt.gca().set_aspect('equal')
    plt.show()


# -----------------------------
# 示例使用
# -----------------------------
if __name__ == "__main__":
    env_json_path = "data/random_2d/val/envs.json"
    with open(env_json_path, "r") as f:
        env_list = json.load(f)

    # 可视化第一个环境
    visualize_env(env_list[0], resolution=1.0)

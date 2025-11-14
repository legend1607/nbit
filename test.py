# visualize_path_keypoints_rect.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from model.cae.cae_2d import Encoder_CNN_2D, Decoder_CNN_2D
from model.PathGeneratorModel import PathKeypointGenerator
from environment.random_2d_env import Random2DEnv as Env

# -----------------------
# 画矩形障碍函数
# -----------------------
def draw_obstacles(ax, env_dict):
    for rx, ry, rw, rh in env_dict.get("rectangle_obstacles", []):
        rect = plt.Rectangle((rx, ry), rw, rh, color='gray', alpha=0.7)
        ax.add_patch(rect)

# -----------------------
# 加载 CAE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder_CNN_2D(input_size=224, latent_dim=256).to(device)
decoder = Decoder_CNN_2D(feature_map_size=encoder.feature_map_size, latent_dim=256).to(device)
encoder.load_state_dict(torch.load("results/cae/encoder_best.pth", map_location=device))
decoder.load_state_dict(torch.load("results/cae/decoder_best.pth", map_location=device))
encoder.eval(); decoder.eval()

# -----------------------
# 加载 PathKeypointGenerator
# -----------------------
generator = PathKeypointGenerator(obs_latent_dim=256, n_points=128).to(device)
generator.load_state_dict(torch.load("results/pathgenerator_npz/best_pathgenerator.pth", map_location=device))
generator.eval()

# -----------------------
# 读取环境 JSON
# -----------------------
env_json_path = "data/random_2d/test/envs.json"
with open(env_json_path, "r") as f:
    env_list = json.load(f)

# -----------------------
# 遍历环境
# -----------------------
for env_dict in env_list[:5]:  # 可视化前5个环境
    env = Env(env_dict)

    for idx, (start, goal) in enumerate(zip(env_dict["start"], env_dict["goal"])):
        path = np.array(env_dict["paths"][idx], dtype=np.float32)

        # -----------------------
        # 生成环境特征 (CAE)
        # -----------------------
        # 这里直接用栅格编码（CAE要求）但不画出来
        width, height = env_dict["env_dims"]
        grid = np.zeros((int(height), int(width)), dtype=np.float32)
        for rx, ry, rw, rh in env_dict.get("rectangle_obstacles", []):
            x1, y1 = int(rx), int(ry)
            x2, y2 = int(rx+rw), int(ry+rh)
            grid[y1:y2, x1:x2] = 1.0
        grid_tensor = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            obs_latent, _ = encoder(grid_tensor)

        # -----------------------
        # PathKeypointGenerator
        # -----------------------
        start_tensor = torch.tensor(start, dtype=torch.float32).unsqueeze(0).to(device)
        goal_tensor  = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_keypoints = generator(obs_latent, start_tensor, goal_tensor)
            pred_keypoints = pred_keypoints.cpu().numpy()[0]

        # -----------------------
        # 可视化障碍 + 路径 + 预测关键点
        # -----------------------
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(0, env_dict["env_dims"][0])
        ax.set_ylim(0, env_dict["env_dims"][1])
        draw_obstacles(ax, env_dict)

        # 专家路径
        ax.plot(path[:,0], path[:,1], 'b-', label='Expert Path')
        # 预测关键点
        ax.scatter(pred_keypoints[:,0], pred_keypoints[:,1], c='r', s=40, label='Pred Keypoints')
        # 起点 / 终点
        ax.scatter(start[0], start[1], c='g', s=80, marker='o', label='Start')
        ax.scatter(goal[0], goal[1], c='k', s=80, marker='x', label='Goal')

        ax.set_aspect('equal')
        ax.set_title(f"Env {env_dict['env_idx']} Sample {idx}")
        ax.legend()
        plt.show()

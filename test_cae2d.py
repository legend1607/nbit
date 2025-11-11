"""
visualize_cae_v2.py
-------------------
加载训练好的 Encoder/Decoder 模型，查看编码器训练效果
可视化原图与重建图，支持保存图片
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model.cae.cae_2d import Encoder_CNN_2D, Decoder_CNN_2D  # 使用最新训练脚本中的模型

# ===============================
# 可视化函数
# ===============================
def visualize_reconstruction(original, reconstructed, n=5, save_path=None):
    """
    original, reconstructed: torch.Tensor, shape (N,1,H,W)
    n: 显示或保存前 n 张
    save_path: 保存路径, 若为 None 则显示
    """
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()
    n = min(n, original.shape[0])

    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.imshow(original[i,0], cmap='gray')
        plt.title("Original")
        plt.axis('off')
        plt.subplot(2, n, i+1+n)
        plt.imshow(reconstructed[i,0], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"✅ Reconstruction saved to {save_path}")
    else:
        plt.show()


# ===============================
# 主函数
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/random_2d/grids.npy',
                        help="待可视化的数据集路径")
    parser.add_argument('--model_dir', type=str, default='results',
                        help="模型所在目录")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_visualize', type=int, default=5,
                        help="显示或保存前 n 张")
    parser.add_argument('--save_dir', type=str, default=None,
                        help="保存可视化图片的目录，如果不设置则显示")
    args = parser.parse_args()

    # 模型路径
    encoder_path = os.path.join(args.model_dir,"cae", "best", "encoder_best.pth")
    decoder_path = os.path.join(args.model_dir,"cae", "best", "decoder_best.pth")

    # 加载数据
    data = np.load(args.dataset_path)
    data = torch.from_numpy(data).float().unsqueeze(1)  # [N,1,H,W]
    loader = DataLoader(TensorDataset(data), batch_size=args.batch_size, shuffle=False)

    # 获取输入大小
    input_size = data.shape[-1]
    latent_dim = 128  # 与训练时保持一致
    dropout_p = 0.0

    # 初始化模型
    encoder = Encoder_CNN_2D(input_size=input_size, latent_dim=latent_dim)
    decoder = Decoder_CNN_2D(feature_map_size=encoder.feature_map_size, latent_dim=latent_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    # 加载权重
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()

    # 可视化
    for batch in loader:
        x = batch[0].to(device)
        with torch.no_grad():
            latent = encoder(x)
            recon = decoder(latent)

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, "reconstruction_best.png")
        else:
            save_path = None

        visualize_reconstruction(x, recon, n=args.n_visualize, save_path=save_path)
        break  # 只显示/保存第一批

if __name__ == "__main__":
    main()

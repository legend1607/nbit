"""
visualize_cae_v2_from_npz_with_path.py
--------------------------------------
加载训练好的 Encoder/Decoder 模型，查看编码器训练效果
可视化原图与重建图，并在原图上显示专家路径
支持保存图片
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 导入训练时的 Encoder / Decoder
from model.cae.cae_2d import Encoder_CNN_2D, Decoder_CNN_2D


# ===============================
# 可视化函数
# ===============================
def visualize_reconstruction(original, reconstructed, paths=None, n=5, save_path=None):
    """
    original, reconstructed: torch.Tensor, shape (N,1,H,W)
    paths: list of arrays, 每个 array shape (L,2)
    n: 显示或保存前 n 张
    """
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()
    n = min(n, original.shape[0])

    plt.figure(figsize=(n * 2, 4))
    for i in range(n):
        # 原始图
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i, 0], cmap='gray')
        if paths is not None:
            path = paths[i]  # shape (L,2)
            plt.plot(path[:, 0], path[:, 1], color='red', linewidth=1.5)
        plt.title("Original")
        plt.axis('off')
        plt.gca().invert_yaxis()

        # 重建图
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i, 0], cmap='gray')
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
    parser.add_argument('--dataset_path', type=str, default='data/random_2d/train.npz',
                        help="待可视化的数据集路径 (.npz 或 .npy)")
    parser.add_argument('--model_dir', type=str, default='results',
                        help="模型所在目录（默认 ./results）")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_visualize', type=int, default=5,
                        help="显示或保存前 n 张")
    parser.add_argument('--save_dir', type=str, default=None,
                        help="保存可视化图片的目录，如果不设置则直接显示")
    parser.add_argument('--latent_dim', type=int, default=256,
                        help="必须与训练时相同")
    args = parser.parse_args()

    # ===============================
    # 模型路径与设备
    # ===============================
    encoder_path = os.path.join(args.model_dir, "cae/encoder_best.pth")
    decoder_path = os.path.join(args.model_dir, "cae/decoder_best.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        raise FileNotFoundError(f"❌ 模型文件不存在，请确认路径：\n{encoder_path}\n{decoder_path}")

    # ===============================
    # 加载数据
    # ===============================
    if args.dataset_path.endswith(".npz"):
        npz_data = np.load(args.dataset_path, allow_pickle=True)
        grids = npz_data["grid"]
        paths = npz_data["path"]
    elif args.dataset_path.endswith(".npy"):
        grids = np.load(args.dataset_path)
        paths = None
    else:
        raise ValueError("只支持 .npz 或 .npy 文件")

    if grids.ndim == 3:
        grids = grids[:, None, :, :]  # [N,1,H,W]

    data = torch.from_numpy(grids).float()
    loader = DataLoader(TensorDataset(data), batch_size=args.batch_size, shuffle=False)

    # ===============================
    # 初始化模型
    # ===============================
    input_size = data.shape[-1]
    encoder = Encoder_CNN_2D(input_size=input_size, latent_dim=args.latent_dim).to(device)
    decoder = Decoder_CNN_2D(feature_map_size=encoder.feature_map_size, latent_dim=args.latent_dim).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()

    # ===============================
    # 可视化
    # ===============================
    for batch_idx, batch in enumerate(loader):
        x = batch[0].to(device)
        with torch.no_grad():
            latent, enc_feats = encoder(x)
            recon = decoder(latent, enc_feats)

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"reconstruction_batch{batch_idx}.png")
        else:
            save_path = None

        # 对应 batch 的路径
        batch_paths = None
        if paths is not None:
            batch_paths = paths[batch_idx*args.batch_size : batch_idx*args.batch_size + x.size(0)]

        visualize_reconstruction(x, recon, paths=batch_paths, n=args.n_visualize, save_path=save_path)
        # break  # 只显示第一批

    print("✅ 可视化完成！")


if __name__ == "__main__":
    main()

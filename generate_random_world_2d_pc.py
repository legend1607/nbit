import json
import time
from os.path import join
import numpy as np
from tqdm import tqdm

# ===============================
# 保存 npz 数据
# ===============================
def save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False):
    raw_dataset_saved = {}
    for k in raw_dataset.keys():
        if k == "token":
            arr = np.array(raw_dataset[k], dtype=str)
        else:
            arr = np.stack(raw_dataset[k], axis=0).astype(np.float32)
        raw_dataset_saved[k] = arr
    filename = "_tmp.npz" if tmp else mode + ".npz"
    np.savez(join(dataset_dir, filename), **raw_dataset_saved)

# ===============================
# 单个环境处理
# ===============================
def process_env(env_dict, EnvClass, mode):
    env = EnvClass(env_dict)
    env_idx = env_dict["env_idx"]
    dataset = {
        "token": [],
        "start": [],
        "goal": [],
        "path": [],
        "rectangle_obstacles": [np.array(env.rect_obstacles, dtype=np.float32)]
    }

    for sample_idx, (s_start, s_goal) in enumerate(zip(env_dict["start"], env_dict["goal"])):
        s_start = np.array(s_start, dtype=np.float32)
        s_goal = np.array(s_goal, dtype=np.float32)
        path = np.array(env_dict["paths"][sample_idx], dtype=np.float32)
        token = f"{mode}-{env_idx}_{sample_idx}"

        dataset["token"].append(token)
        dataset["start"].append(s_start)
        dataset["goal"].append(s_goal)
        dataset["path"].append(path)

    return dataset

# ===============================
# 主函数：生成 npz 数据集
# ===============================
def generate_npz_dataset(env_type="random_2d_bitstar"):
    from environment.random_2d_env import Random2DEnv as Env

    dataset_dir = join("data", f"{env_type}")

    for mode in ["train", "val"]:
        env_json_path = join(dataset_dir, mode, "envs.json")
        with open(env_json_path, "r") as f:
            env_list = json.load(f)

        raw_dataset = {
            "token": [],
            "start": [],
            "goal": [],
            "path": [],
            "rectangle_obstacles": [],
        }

        start_time = time.time()

        for env_idx, env_dict in enumerate(tqdm(env_list, desc=f"{mode} environments")):
            env_data = process_env(env_dict, Env, mode)
            
            # 合并到总数据集
            for k in raw_dataset.keys():
                raw_dataset[k].extend(env_data[k])

            # 临时保存
            if (env_idx + 1) % 25 == 0:
                save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=True)
                elapsed = time.time() - start_time
                remaining = elapsed / (env_idx + 1) * (len(env_list) - (env_idx + 1))
                print(f"[{mode}] {env_idx + 1}/{len(env_list)} envs processed, "
                      f"estimated remaining time: {int(remaining // 60)} min", flush=True)

        save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False)
        print(f"[{mode}] saved {len(raw_dataset['token'])} samples to {mode}.npz", flush=True)


if __name__ == "__main__":
    generate_npz_dataset("random_2d")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PathPlanDataset(Dataset):
    def __init__(self, dataset_filepath):
        data = np.load(dataset_filepath, allow_pickle=True)
        self.start = data["start"].astype(np.float32)
        self.goal = data["goal"].astype(np.float32)
        self.path = [p.astype(np.float32) for p in data["path"]]
        self.rectangle_obstacles = [o.astype(np.float32) for o in data["rectangle_obstacles"]]
        self.token = data["token"]

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        return {
            "start": torch.from_numpy(self.start[idx]),
            "goal": torch.from_numpy(self.goal[idx]),
            "path": torch.from_numpy(self.path[idx]),
            "rectangle_obstacles": torch.from_numpy(self.rectangle_obstacles[idx]),
            "token": self.token[idx],
        }

def collate_fn(batch, device=None):
    # Paths
    paths = [item["path"] for item in batch]
    path_lengths = [p.shape[0] for p in paths]
    max_len = max(path_lengths)
    D = paths[0].shape[1]
    padded_paths = torch.zeros(len(paths), max_len, D)
    mask = torch.zeros(len(paths), max_len, dtype=torch.bool)
    for i, p in enumerate(paths):
        padded_paths[i, :p.shape[0]] = p
        mask[i, :p.shape[0]] = 1

    # Obstacles
    obs_list = [item["rectangle_obstacles"] for item in batch]
    n_obs_max = max(o.shape[0] for o in obs_list)
    obs_dim = obs_list[0].shape[1]
    padded_obs = torch.zeros(len(obs_list), n_obs_max, obs_dim)
    for i, o in enumerate(obs_list):
        padded_obs[i, :o.shape[0]] = o

    # Stack starts/goals
    starts = torch.stack([item["start"] for item in batch])
    goals = torch.stack([item["goal"] for item in batch])
    tokens = [item["token"] for item in batch]

    if device:
        padded_paths = padded_paths.to(device)
        mask = mask.to(device)
        padded_obs = padded_obs.to(device)
        starts = starts.to(device)
        goals = goals.to(device)

    return {
        "start": starts,
        "goal": goals,
        "path": padded_paths,
        "path_mask": mask,
        "rectangle_obstacles": padded_obs,
        "token": tokens,
        "path_lengths": path_lengths
    }

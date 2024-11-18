import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils.ray_marching import ray_marching, prepare_rays

class NeRFDataset(Dataset):
    """
    Dataset for loading NeRF data (images and camera parameters).
    """
    def __init__(self, dataset_dir, split="train"):
        """
        Initialize the dataset.
        Args:
            dataset_dir (str): Path to the dataset directory (e.g., "data/nerf_synthetic/lego").
            split (str): Dataset split to use ("train", "val", or "test").
        """
        self.split = split
        self.dataset_dir = os.path.join(dataset_dir, split)
        
        # Load transformation JSON
        transform_path = os.path.join(dataset_dir, f"transforms_{split}.json")
        with open(transform_path, 'r') as f:
            self.transforms = json.load(f)

        # Collect image file paths and camera parameters
        self.image_paths = []
        self.camera_params = []

        for frame in self.transforms["frames"]:
            self.image_paths.append(os.path.join(self.dataset_dir, frame["file_path"] + ".png"))
            self.camera_params.append(frame["transform_matrix"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        Args:
            idx (int): Index of the item.
        Returns:
            image (Tensor): Image tensor (C, H, W).
            camera_params (Tensor): Camera transformation matrix.
        """
        # Load and process image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to (C, H, W)

        # Load camera parameters
        camera_params = torch.tensor(self.camera_params[idx], dtype=torch.float32)

        return image, camera_params

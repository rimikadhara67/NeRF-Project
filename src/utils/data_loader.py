import torch
from torch.utils.data import Dataset

class NeRFDataset(Dataset):
    def __init__(self, images, camera_params):
        self.images = images
        self.camera_params = camera_params

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.camera_params[idx]

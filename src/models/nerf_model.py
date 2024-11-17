import torch
import torch.nn as nn
from models.positional_encoding import PositionalEncoding

class NeRF(nn.Module):
    def __init__(self, pos_encoding_dim=10, dir_encoding_dim=4):
        super(NeRF, self).__init__()
        self.pos_encoding = PositionalEncoding(3, pos_encoding_dim)
        self.dir_encoding = PositionalEncoding(3, dir_encoding_dim)

        self.density_net = nn.Sequential(
            nn.Linear(3 + pos_encoding_dim * 2 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.feature_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.color_net = nn.Sequential(
            nn.Linear(128 + dir_encoding_dim * 2 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, coords, view_dirs):
        encoded_coords = self.pos_encoding(coords)
        encoded_dirs = self.dir_encoding(view_dirs)

        density_feat = self.density_net(encoded_coords)
        density = density_feat[:, :1]

        intermediate_features = self.feature_net(density_feat)
        color = self.color_net(torch.cat([intermediate_features, encoded_dirs], dim=-1))

        return density, color

import torch
import torch.nn as nn
import torch.nn.functional as F

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_frequencies):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x):
        # Apply sin and cos encoding for each frequency
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)

# NeRF Model
class NeRF(nn.Module):
    def __init__(self, pos_encoding_dim=10, dir_encoding_dim=4):
        super(NeRF, self).__init__()
        
        # Input encodings
        self.pos_encoding = PositionalEncoding(3, pos_encoding_dim)
        self.dir_encoding = PositionalEncoding(3, dir_encoding_dim)

        # Layers for density and intermediate features
        self.density_net = nn.Sequential(
            nn.Linear(3 + pos_encoding_dim * 2 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs density (σ)
        )
        
        # Layers for color prediction
        self.feature_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.color_net = nn.Sequential(
            nn.Linear(128 + dir_encoding_dim * 2 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Outputs color (r, g, b)
        )

    def forward(self, coords, view_dirs):
        # Positional encoding
        encoded_coords = self.pos_encoding(coords)
        encoded_dirs = self.dir_encoding(view_dirs)

        # Density prediction
        density_feat = F.relu(self.density_net(encoded_coords))
        density = density_feat[:, :1]

        # Color prediction
        intermediate_features = self.feature_net(density_feat)
        color = self.color_net(torch.cat([intermediate_features, encoded_dirs], dim=-1))

        return density, color

# Ray Marching and Rendering
def ray_marching_and_rendering(model, rays, num_samples=64):
    """
    Perform ray marching and volume rendering.
    
    Parameters:
        model: The NeRF model.
        rays: Tensor of shape (num_rays, 6), containing ray origins and directions.
        num_samples: Number of points to sample along each ray.
    """
    ray_origins, ray_directions = rays[:, :3], rays[:, 3:6]
    near, far = 0.1, 1.0  # Near and far bounds for sampling
    t_vals = torch.linspace(near, far, num_samples).to(ray_origins.device)

    # Compute sample points along the rays
    sample_points = ray_origins[:, None, :] + t_vals[None, :, None] * ray_directions[:, None, :]
    sample_points = sample_points.reshape(-1, 3)

    # Query the model for density and color
    densities, colors = model(sample_points, ray_directions.repeat_interleave(num_samples, dim=0))

    # Accumulate densities and colors
    alpha = 1.0 - torch.exp(-densities)  # Convert densities to alpha values
    weights = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=1)
    rendered_colors = torch.sum(weights[:, :, None] * colors, dim=1)

    return rendered_colors

# Loss Function (Photometric Loss)
def photometric_loss(rendered_colors, ground_truth_colors):
    return torch.mean((rendered_colors - ground_truth_colors) ** 2)

# Example Training Loop
def train_nerf(model, dataloader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
        import torch
import torch.nn as nn
import torch.nn.functional as F

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_frequencies):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x):
        # Apply sin and cos encoding for each frequency
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)

# NeRF Model
class NeRF(nn.Module):
    def __init__(self, pos_encoding_dim=10, dir_encoding_dim=4):
        super(NeRF, self).__init__()
        
        # Input encodings
        self.pos_encoding = PositionalEncoding(3, pos_encoding_dim)
        self.dir_encoding = PositionalEncoding(3, dir_encoding_dim)

        # Layers for density and intermediate features
        self.density_net = nn.Sequential(
            nn.Linear(3 + pos_encoding_dim * 2 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs density (σ)
        )
        
        # Layers for color prediction
        self.feature_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.color_net = nn.Sequential(
            nn.Linear(128 + dir_encoding_dim * 2 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Outputs color (r, g, b)
        )

    def forward(self, coords, view_dirs):
        # Positional encoding
        encoded_coords = self.pos_encoding(coords)
        encoded_dirs = self.dir_encoding(view_dirs)

        # Density prediction
        density_feat = F.relu(self.density_net(encoded_coords))
        density = density_feat[:, :1]

        # Color prediction
        intermediate_features = self.feature_net(density_feat)
        color = self.color_net(torch.cat([intermediate_features, encoded_dirs], dim=-1))

        return density, color

# Ray Marching and Rendering
def ray_marching_and_rendering(model, rays, num_samples=64):
    """
    Perform ray marching and volume rendering.
    
    Parameters:
        model: The NeRF model.
        rays: Tensor of shape (num_rays, 6), containing ray origins and directions.
        num_samples: Number of points to sample along each ray.
    """
    ray_origins, ray_directions = rays[:, :3], rays[:, 3:6]
    near, far = 0.1, 1.0  # Near and far bounds for sampling
    t_vals = torch.linspace(near, far, num_samples).to(ray_origins.device)

    # Compute sample points along the rays
    sample_points = ray_origins[:, None, :] + t_vals[None, :, None] * ray_directions[:, None, :]
    sample_points = sample_points.reshape(-1, 3)

    # Query the model for density and color
    densities, colors = model(sample_points, ray_directions.repeat_interleave(num_samples, dim=0))

    # Accumulate densities and colors
    alpha = 1.0 - torch.exp(-densities)  # Convert densities to alpha values
    weights = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=1)
    rendered_colors = torch.sum(weights[:, :, None] * colors, dim=1)

    return rendered_colors

# Loss Function (Photometric Loss)
def photometric_loss(rendered_colors, ground_truth_colors):
    return torch.mean((rendered_colors - ground_truth_colors) ** 2)

# Example Training Loop
def train_nerf(model, dataloader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            rays, gt_colors = batch  # Ray origins and directions, Ground truth colors
            optimizer.zero_grad()

            # Ray marching and rendering
            rendered_colors = ray_marching_and_rendering(model, rays)

            # Compute loss
            loss = photometric_loss(rendered_colors, gt_colors)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    rays, gt_colors = batch  # Ray origins and directions, Ground truth colors
            optimizer.zero_grad()

            # Ray marching and rendering
            rendered_colors = ray_marching_and_rendering(model, rays)

            # Compute loss
            loss = photometric_loss(rendered_colors, gt_colors)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


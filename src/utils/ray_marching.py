import torch

def ray_marching(model, rays, num_samples=64, near=0.1, far=1.0):
    ray_origins, ray_directions = rays[:, :3], rays[:, 3:]
    t_vals = torch.linspace(near, far, num_samples).to(ray_origins.device)
    sample_points = ray_origins[:, None, :] + t_vals[None, :, None] * ray_directions[:, None, :]
    sample_points = sample_points.reshape(-1, 3)

    densities, colors = model(sample_points, ray_directions.repeat_interleave(num_samples, dim=0))
    alpha = 1.0 - torch.exp(-densities)
    weights = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=1)
    rendered_colors = torch.sum(weights[:, :, None] * colors, dim=1)

    return rendered_colors

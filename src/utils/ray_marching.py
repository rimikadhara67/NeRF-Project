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

def prepare_rays(camera_params, H=800, W=800, focal_length=1111.11):
    """
    Prepare rays for ray marching based on camera parameters.
    
    Args:
        camera_params (Tensor): Camera transformation matrices (B, 4, 4).
        H (int): Height of the image.
        W (int): Width of the image.
        focal_length (float): Focal length of the camera.
    
    Returns:
        rays (Tensor): Tensor of shape (B, H * W, 6) containing ray origins and directions.
    """
    # Generate pixel coordinates
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="ij")
    i, j = i.to(camera_params.device), j.to(camera_params.device)
    directions = torch.stack([(i - W * 0.5) / focal_length,
                              -(j - H * 0.5) / focal_length,
                              -torch.ones_like(i)], dim=-1)

    # Apply camera rotation and translation
    rays_d = torch.sum(directions[..., None, :] * camera_params[:, None, None, :3, :3], dim=-1)
    rays_o = camera_params[:, None, None, :3, 3].expand(rays_d.shape)

    # Reshape rays to (B, H * W, 6)
    rays = torch.cat([rays_o, rays_d], dim=-1).reshape(camera_params.shape[0], -1, 6)

    return rays

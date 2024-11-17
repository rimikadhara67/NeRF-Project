import numpy as np
import torch
from skimage.measure import marching_cubes
import trimesh

def extract_3d_mesh(model, grid_size=128, density_threshold=0.5):
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        densities, _ = model(grid_tensor, torch.zeros_like(grid_tensor))
        densities = densities.view(grid_size, grid_size, grid_size).cpu().numpy()

    vertices, faces, normals, _ = marching_cubes(densities, level=density_threshold)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    return mesh

def save_mesh(mesh, file_path):
    mesh.export(file_path)

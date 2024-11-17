import torch
from models.nerf_model import NeRF
from utils.mesh_extraction import extract_3d_mesh, save_mesh

if __name__ == "__main__":
    model = NeRF()
    model.load_state_dict(torch.load("nerf_model.pth"))
    model.eval()

    mesh = extract_3d_mesh(model)
    save_mesh(mesh, "data/meshes/scene_mesh.obj")
    print("Mesh saved to data/meshes/scene_mesh.obj")

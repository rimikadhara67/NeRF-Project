import torch
from torch.utils.data import DataLoader
from models.nerf_model import NeRF
from utils.data_loader import NeRFDataset
from utils.ray_marching import ray_marching

def train_nerf(model, dataloader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, camera_params in dataloader:
            optimizer.zero_grad()
            rays = camera_params  # Prepare rays from camera params
            rendered_colors = ray_marching(model, rays)
            loss = torch.mean((rendered_colors - images) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    model = NeRF()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load dataset
    images = torch.randn(100, 3, 64, 64)  # Example data
    camera_params = torch.randn(100, 6)
    dataset = NeRFDataset(images, camera_params)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    train_nerf(model, dataloader, optimizer)
    torch.save(model.state_dict(), "nerf_model.pth")

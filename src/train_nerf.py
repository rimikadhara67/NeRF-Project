import torch
from torch.utils.data import DataLoader
from models.nerf_model import NeRF
from utils.data_loader import NeRFDataset
from utils.ray_marching import ray_marching, prepare_rays

def train_nerf(model, dataloader, optimizer, epochs=10):
    """
    Train the NeRF model.
    Args:
        model (nn.Module): The NeRF model.
        dataloader (DataLoader): DataLoader for the dataset.
        optimizer (Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, camera_params in dataloader:
            optimizer.zero_grad()
            
            # Prepare rays from camera parameters
            rays = prepare_rays(camera_params)  # Implement this function based on your ray marching logic
            
            # Render colors
            rendered_colors = ray_marching(model, rays)
            
            # Compute photometric loss
            loss = torch.mean((rendered_colors - images) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    dataset_dir = "data/nerf_synthetic/lego"
    model = NeRF()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Load dataset
    train_dataset = NeRFDataset(dataset_dir, split="train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Train the NeRF model
    train_nerf(model, train_loader, optimizer)
    torch.save(model.state_dict(), "nerf_model.pth")

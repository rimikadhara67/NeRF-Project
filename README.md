# NeRF Project

This repository contains the implementation of a **Neural Radiance Field (NeRF)** model for 3D scene reconstruction and 3D mesh extraction. The project includes functionalities for training a NeRF model, rendering novel views, and exporting 3D meshes.

## File Structure

## Description of Key Files
- **`train_nerf.py`**: Script for training the NeRF model on synthetic or real-world datasets.
- **`extract_mesh.py`**: Script for extracting and saving 3D meshes from a trained NeRF.
- **`ray_marching.py`**: Implementation of ray marching for rendering novel views.
- **`mesh_extraction.py`**: Contains functions for extracting 3D meshes using Marching Cubes.
- **`data_loader.py`**: Data utilities for loading datasets and camera parameters.
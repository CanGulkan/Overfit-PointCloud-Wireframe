"""
Simplified Training Script for Point Cloud to Wireframe Model

This script provides a clean and easy-to-understand implementation for training
a neural network that converts point clouds to wireframe representations.
"""

import os
import glob
import random
import time
import torch
import numpy as np
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging

from models.PointCloudToWireframe import PointCloudToWireframe
from losses.WireframeLoss import WireframeLoss
from demo_dataset.MultiPairPCWF import MultiPairPCWF

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_matching_files(point_cloud_dir: str, wireframe_dir: str) -> List[Tuple[str, str]]:
    """
    Find pairs of .xyz and .obj files with matching names.
    
    Args:
        point_cloud_dir: Directory containing .xyz point cloud files
        wireframe_dir: Directory containing .obj wireframe files
    
    Returns:
        List of tuples containing (point_cloud_path, wireframe_path)
    """
    # Get all .xyz and .obj files
    xyz_files = glob.glob(os.path.join(point_cloud_dir, "*.xyz"))
    obj_files = glob.glob(os.path.join(wireframe_dir, "*.obj"))
    
    # Create maps from filename (without extension) to full path
    xyz_map = {os.path.splitext(os.path.basename(p))[0]: p for p in xyz_files}
    obj_map = {os.path.splitext(os.path.basename(p))[0]: p for p in obj_files}
    
    # Find common filenames
    common_names = sorted(set(xyz_map.keys()) & set(obj_map.keys()))
    
    # Return paired paths
    return [(xyz_map[name], obj_map[name]) for name in common_names]


def split_data_into_train_test(
    file_pairs: List[Tuple[str, str]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split file pairs into training and testing sets.
    
    Args:
        file_pairs: List of (point_cloud_path, wireframe_path) tuples
        train_ratio: Fraction of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (training_pairs, testing_pairs)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the pairs
    shuffled_pairs = file_pairs.copy()
    random.shuffle(shuffled_pairs)
    
    # Calculate split point
    split_index = int(len(shuffled_pairs) * train_ratio)
    
    # Split the data
    training_pairs = shuffled_pairs[:split_index]
    testing_pairs = shuffled_pairs[split_index:]
    
    logger.info(f"Split {len(file_pairs)} pairs: {len(training_pairs)} training, {len(testing_pairs)} testing")
    
    return training_pairs, testing_pairs


def create_data_collator(max_vertices: int = 32):
    """
    Create a function that batches data together with consistent dimensions.
    
    Args:
        max_vertices: Maximum number of vertices to process
    
    Returns:
        Collate function for DataLoader
    """
    # Create all possible edge combinations
    edge_pairs = [(i, j) for i in range(max_vertices) for j in range(i+1, max_vertices)]
    
    def collate_function(batch):
        """
        Combine multiple samples into a single batch.
        
        Args:
            batch: List of individual samples
            
        Returns:
            Dictionary containing batched data
        """
        batch_size = len(batch)
        
        # Get maximum number of points across all samples in batch
        max_points = max(sample["pc"].shape[0] for sample in batch)
        
        # Initialize tensors for batched data
        batched_point_clouds = torch.zeros(batch_size, max_points, 8, dtype=torch.float32)
        batched_vertices = torch.zeros(batch_size, max_vertices, 3, dtype=torch.float32)
        batched_adjacency = torch.zeros(batch_size, max_vertices, max_vertices, dtype=torch.float32)
        
        # Lists to store metadata
        sample_names = []
        scalers = []
        
        # Process each sample in the batch
        for i, sample in enumerate(batch):
            # Handle point cloud data
            num_points = sample["pc"].shape[0]
            batched_point_clouds[i, :num_points, :] = torch.from_numpy(sample["pc"]).float()
            
            # Handle vertex data
            num_vertices = sample["V"].shape[0]
            vertices_to_take = min(num_vertices, max_vertices)
            batched_vertices[i, :vertices_to_take, :] = torch.from_numpy(sample["V"][:vertices_to_take]).float()
            
            # Handle adjacency data
            batched_adjacency[i, :vertices_to_take, :vertices_to_take] = torch.from_numpy(
                sample["A"][:vertices_to_take, :vertices_to_take]
            ).float()
            
            # Store metadata
            sample_names.append(sample["name"])
            scalers.append(sample["scaler"])
        
        # Create edge labels from adjacency matrix
        num_edges = len(edge_pairs)
        edge_labels = torch.zeros(batch_size, num_edges, dtype=torch.float32)
        
        for i in range(batch_size):
            for edge_idx, (u, v) in enumerate(edge_pairs):
                edge_labels[i, edge_idx] = batched_adjacency[i, u, v]
        
        return {
            "point_cloud": batched_point_clouds,
            "vertices": batched_vertices,
            "edge_labels": edge_labels,
            "edge_indices": edge_pairs,
            "names": sample_names,
            "scalers": scalers,
        }
    
    return collate_function


def evaluate_model(model: PointCloudToWireframe, dataset, device: torch.device) -> Dict[str, float]:
    """
    Evaluate the trained model on a dataset.
    
    Args:
        model: Trained PointCloudToWireframe model
        dataset: PointCloudWireframeDataset instance
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Check if dataset has all required data
        if not hasattr(dataset, 'normalized_point_cloud') or dataset.normalized_point_cloud is None:
            raise ValueError(
                "Dataset missing normalized point cloud data. "
                "Make sure to call: dataset.load_point_cloud() and dataset.normalize_data()"
            )
        
        if not hasattr(dataset, 'normalized_vertices') or dataset.normalized_vertices is None:
            raise ValueError(
                "Dataset missing normalized vertices. "
                "Make sure to call: dataset.load_wireframe() and dataset.normalize_data()"
            )
        
        if not hasattr(dataset, 'edge_adjacency_matrix') or dataset.edge_adjacency_matrix is None:
            raise ValueError(
                "Dataset missing adjacency matrix. "
                "Make sure to call: dataset.create_adjacency_matrix() after loading wireframe data"
            )
        
        # Prepare input data - add batch dimension and ensure float32
        point_cloud = torch.from_numpy(dataset.normalized_point_cloud).float().unsqueeze(0).to(device)
        
        # Get predictions
        predictions = model(point_cloud)
        
        # Get ground truth - ensure float32
        target_vertices = torch.from_numpy(dataset.normalized_vertices).float().to(device)
        target_adjacency = torch.from_numpy(dataset.edge_adjacency_matrix).float().to(device)
        
        # Calculate vertex RMSE
        pred_vertices = predictions["vertices"][0]  # Remove batch dimension
        num_vertices = min(pred_vertices.shape[0], target_vertices.shape[0])
        
        if num_vertices > 0:
            vertex_rmse = torch.sqrt(torch.mean((pred_vertices[:num_vertices] - target_vertices[:num_vertices])**2)).item()
        else:
            vertex_rmse = 0.0
        
        # Calculate edge accuracy (simplified)
        pred_edge_probs = predictions["edge_probs"][0]  # Remove batch dimension
        edge_accuracy = 0.0
        
        if len(pred_edge_probs) > 0:
            # Convert probabilities to binary predictions
            pred_edges = (pred_edge_probs > 0.5).float()
            # This is a simplified accuracy calculation
            edge_accuracy = pred_edges.mean().item()
        
        return {
            "vertex_rmse": vertex_rmse,
            "edge_accuracy": edge_accuracy,
            "edge_f1_score": edge_accuracy  # Simplified F1 score
        }


def train_model(
    training_pairs: List[Tuple[str, str]],
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    max_vertices: int = 32,
    batch_size: int = 4
) -> PointCloudToWireframe:
    """
    Train the point cloud to wireframe model.
    
    Args:
        training_pairs: List of (point_cloud_path, wireframe_path) tuples for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        max_vertices: Maximum number of vertices to process
        batch_size: Number of samples per batch
    
    Returns:
        Trained model
    """
    # Set up device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Create dataset and data loader
    dataset = MultiPairPCWF(training_pairs)
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=create_data_collator(max_vertices)
    )
    
    # Initialize model, loss function, and optimizer
    model = PointCloudToWireframe(input_dim=8, num_vertices=max_vertices).to(device)
    loss_function = WireframeLoss(vertex_weight=50.0, edge_weight=0.1)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0, eps=1e-8)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[40, 80, 150], 
        gamma=0.3
    )
    
    # Training state
    model.train()
    best_loss = float('inf')
    best_model_state = None
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        samples_processed = 0
        
        # Process each batch
        for batch_idx, batch in enumerate(data_loader):
            # Move data to device
            point_clouds = batch["point_cloud"].to(device)
            target_vertices = batch["vertices"].to(device)
            target_edge_labels = batch["edge_labels"].to(device)
            sample_names = batch["names"]
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(point_clouds)
            
            # Calculate loss
            targets = {
                "vertices": target_vertices, 
                "edge_labels": target_edge_labels
            }
            loss_info = loss_function(predictions, targets)
            total_loss = loss_info["total_loss"]
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Log progress with loss information
            batch_size = point_clouds.size(0)
            names_to_show = ', '.join(sample_names[:min(4, len(sample_names))])
            
            # Show detailed loss breakdown
            vertex_loss = loss_info.get("vertex_loss", 0.0)
            edge_loss = loss_info.get("edge_loss", 0.0)
            
            print(f"[Epoch {epoch:03d}, Batch {batch_idx+1}/{len(data_loader)}] Processing: {names_to_show}")
            print(f"  └─ Loss: {total_loss.item():.6f} (Vertex: {vertex_loss:.6f}, Edge: {edge_loss:.6f})")
            
            # Accumulate loss
            epoch_loss += total_loss.item() * batch_size
            samples_processed += batch_size
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss for this epoch
        average_loss = epoch_loss / max(1, samples_processed)
        
        # Save best model
        if average_loss < best_loss:
            best_loss = average_loss
            best_model_state = model.state_dict()
        
        # Log epoch summary
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            elapsed_time = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            
            logger.info(
                f"Epoch {epoch:4d}/{num_epochs} | "
                f"Avg Loss: {average_loss:.6f} | "
                f"LR: {current_lr:.6f} | "
                f"Best Loss: {best_loss:.6f} | "
                f"Time: {elapsed_time:.1f}s"
            )
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with loss: {best_loss:.6f}")
    
    return model


def main():
    """Main function to run the training process."""
    # Configuration
    POINT_CLOUD_DIR = "demo_dataset/pointcloud"
    WIREFRAME_DIR = "demo_dataset/wireframe"
    
    # Training parameters
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    MAX_VERTICES = 32
    BATCH_SIZE = 4
    
    # Find matching files
    logger.info("Finding matching point cloud and wireframe files...")
    file_pairs = find_matching_files(POINT_CLOUD_DIR, WIREFRAME_DIR)
    
    if not file_pairs:
        logger.error("No matching files found!")
        return
    
    logger.info(f"Found {len(file_pairs)} matching file pairs")
    
    # Split into training and testing sets
    training_pairs, testing_pairs = split_data_into_train_test(file_pairs)
    
    # Train the model
    logger.info("Starting training...")
    trained_model = train_model(
        training_pairs=training_pairs,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        max_vertices=MAX_VERTICES,
        batch_size=BATCH_SIZE
    )
    
    # Save the trained model
    torch.save(trained_model.state_dict(), "trained_model.pth")
    logger.info("Training completed! Model saved as 'trained_model.pth'")


if __name__ == "__main__":
    main()

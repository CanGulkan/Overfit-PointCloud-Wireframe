import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import logging
from models.PointCloudToWireframe import PointCloudToWireframe
from losses.WireframeLoss import WireframeLoss
from models.EdgePredictor import EdgePredictor
from models.PointNetEncoder import PointNetEncoder
from demo_dataset.PointCloudWireframeDataset import PointCloudWireframeDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_adjacency_matrix_from_predictions(edge_probs, edge_indices, num_vertices, threshold=0.5):
    """Convert edge predictions to adjacency matrix"""
    batch_size = edge_probs.shape[0]
    adj_matrices = torch.zeros(batch_size, num_vertices, num_vertices)
    
    for batch_idx in range(batch_size):
        for edge_idx, (i, j) in enumerate(edge_indices):
            if edge_probs[batch_idx, edge_idx] > threshold:
                adj_matrices[batch_idx, i, j] = 1
                adj_matrices[batch_idx, j, i] = 1  # Symmetric
                
    return adj_matrices


def create_edge_labels_from_adjacency(adj_matrix, edge_indices):
    """Create edge labels tensor from adjacency matrix"""
    batch_size = 1  # Single example
    num_edges = len(edge_indices)
    edge_labels = torch.zeros(batch_size, num_edges)
    
    for edge_idx, (i, j) in enumerate(edge_indices):
        if adj_matrix[i, j] == 1:
            edge_labels[0, edge_idx] = 1
            
    return edge_labels

def train_model(dataset, num_epochs=5000, learning_rate=0.001):
    """Train model to overfit on single example"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Create model
    num_vertices = len(dataset.vertices)
    model = PointCloudToWireframe(input_dim=8, num_vertices=num_vertices).to(device)
    
    # Prepare data
    point_cloud_tensor = torch.FloatTensor(dataset.normalized_point_cloud).unsqueeze(0).to(device)
    target_vertices = torch.FloatTensor(dataset.normalized_vertices).unsqueeze(0).to(device)
    
    # Create edge labels
    edge_labels = create_edge_labels_from_adjacency(
        dataset.edge_adjacency_matrix, 
        [(i, j) for i in range(num_vertices) for j in range(i+1, num_vertices)]
    ).to(device)
    
    criterion = WireframeLoss(vertex_weight=50.0, edge_weight=0.1)  # Extreme vertex focus
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0, eps=1e-8)  # No weight decay
    # More aggressive learning rate schedule
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 700, 1700, 4000], gamma=0.3)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    best_vertex_rmse = float('inf')  # Initialize best vertex RMSE
    best_model_state = None  # Initialize best model state
    loss_history = []
    vertex_rmse_history = []
    patience = 500  # Increased patience for better convergence
    patience_counter = 0
    current_vertex_rmse = 999999

    logger.info(f"Starting vertex-optimized for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(point_cloud_tensor)
        
        # Calculate loss
        targets = {
            'vertices': target_vertices,
            'edge_labels': edge_labels
        }
        loss_dict = criterion(predictions, targets)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Track progress
        loss_history.append(total_loss.item())
        
        # Calculate current vertex RMSE for monitoring
        with torch.no_grad():
            pred_vertices_np = predictions['vertices'].cpu().numpy()[0]
            target_vertices_np = target_vertices.cpu().numpy()[0]
            # Convert back to original scale for RMSE calculation
            pred_vertices_orig = dataset.spatial_scaler.inverse_transform(pred_vertices_np)
            target_vertices_orig = dataset.spatial_scaler.inverse_transform(target_vertices_np)
            current_vertex_rmse = np.sqrt(np.mean((pred_vertices_orig - target_vertices_orig) ** 2))
            vertex_rmse_history.append(current_vertex_rmse)
        
        # Early stopping based on vertex RMSE and save best model
        if current_vertex_rmse < best_vertex_rmse:
            best_vertex_rmse = current_vertex_rmse
            best_model_state = model.state_dict().copy()  # Save the best model state
            patience_counter = 0
        else:
            patience_counter += 1
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
        
        # Early stopping check (more aggressive)
        if patience_counter >= patience and epoch > 400:
            logger.info(f"Early stopping at epoch {epoch}! Vertex RMSE hasn't improved for {patience} epochs")
            break

        # Log progress 
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            elapsed_time = time.time() - start_time
            logger.info(f"Epoch {epoch:4d}/{num_epochs} | "
                       f"Total Loss: {total_loss.item():.6f} | "
                       f"Vertex Loss: {loss_dict['vertex_loss'].item():.6f} | "
                       f"Edge Loss: {loss_dict['edge_loss'].item():.6f} | "
                       f"Vertex RMSE: {current_vertex_rmse:.6f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                       f"Time: {elapsed_time:.1f}s")
            
    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model state with Vertex RMSE: {best_vertex_rmse:.6f}")
    
    logger.info(f"Training completed! Best loss: {best_loss:.6f}")
    
    return model, loss_history


def evaluate_model(model, dataset, device):
    """Evaluate the trained model"""
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        point_cloud_tensor = torch.FloatTensor(dataset.normalized_point_cloud).unsqueeze(0).to(device)
        
        # Forward pass
        predictions = model(point_cloud_tensor)
        
        # Get predictions
        pred_vertices = predictions['vertices'].cpu().numpy()[0]
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices']
        
        # Convert back to original scale
        pred_vertices_original = dataset.spatial_scaler.inverse_transform(pred_vertices)
        true_vertices_original = dataset.vertices
        
        # Calculate metrics
        vertex_mse = np.mean((pred_vertices_original - true_vertices_original) ** 2)
        vertex_rmse = np.sqrt(vertex_mse)
        
        # Edge accuracy (threshold at 0.5)
        pred_adj_matrix = create_adjacency_matrix_from_predictions(
            torch.FloatTensor(pred_edge_probs).unsqueeze(0),
            edge_indices,
            len(dataset.vertices),
            threshold=0.5
        )[0].numpy()
        
        true_adj_matrix = dataset.edge_adjacency_matrix
        edge_accuracy = np.mean((pred_adj_matrix == true_adj_matrix).astype(float))
        
        # Edge precision and recall
        true_edges = (true_adj_matrix == 1)
        pred_edges = (pred_adj_matrix == 1)
        
        tp = np.sum(true_edges & pred_edges)
        fp = np.sum(~true_edges & pred_edges)
        fn = np.sum(true_edges & ~pred_edges)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'vertex_rmse': vertex_rmse,
            'edge_accuracy': edge_accuracy,
            'edge_precision': precision,
            'edge_recall': recall,
            'edge_f1_score': f1_score,
            'predicted_vertices': pred_vertices_original,
            'predicted_adjacency': pred_adj_matrix,
            'edge_probabilities': pred_edge_probs
        }
        
        return results


import torch
from demo_dataset.PointCloudWireframeDataset import PointCloudWireframeDataset
from train import evaluate_model, train_overfit_model


def load_and_preprocess_data():
    """Load and preprocess the main dataset"""
    dataset = PointCloudWireframeDataset('demo_dataset/pointcloud/1003.xyz', 'demo_dataset/wireframe/1003.obj')
    
    # Load data
    point_cloud = dataset.load_point_cloud()
    vertices, edges = dataset.load_wireframe()
    
    # Create adjacency matrix
    adj_matrix = dataset.create_adjacency_matrix()
    
    # Normalize data
    normalized_pc = dataset.normalize_data()
    
    # Find nearest points to vertices
    distances, indices = dataset.find_nearest_points_to_vertices(k=10)
    
    return dataset

if __name__ == "__main__":
    # Load data
    dataset = load_and_preprocess_data()
    
    print(f"Point cloud shape: {dataset.point_cloud.shape}")
    print(f"Vertices shape: {dataset.vertices.shape}") 
    print(f"Edges shape: {dataset.edges.shape}")
    print(f"Adjacency matrix shape: {dataset.edge_adjacency_matrix.shape}")
    print(f"Normalized point cloud shape: {dataset.normalized_point_cloud.shape}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*50)
    print("STARTING OVERTRAINING ON SINGLE EXAMPLE")
    print("="*50)
    
    model, loss_history = train_overfit_model(dataset, num_epochs=600, learning_rate=0.003)
    
    print("\n" + "="*50)
    print("EVALUATING TRAINED MODEL")
    print("="*50)
    
    # Evaluate model
    results = evaluate_model(model, dataset, device)
    
    print(f"Vertex RMSE: {results['vertex_rmse']:.6f}")
    print(f"Edge Accuracy: {results['edge_accuracy']:.4f}")
    print(f"Edge Precision: {results['edge_precision']:.4f}")
    print(f"Edge Recall: {results['edge_recall']:.4f}")
    print(f"Edge F1-Score: {results['edge_f1_score']:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'trained_model.pth')
    print("\nModel saved as 'trained_model.pth'")
    
    print("\nOvertraining completed successfully!")

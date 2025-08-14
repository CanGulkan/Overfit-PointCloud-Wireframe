import torch
from demo_dataset.PointCloudWireframeDataset import PointCloudWireframeDataset
from train import evaluate_model, train_model 
from train_multiple import train_models, make_pcwf_collate

import os
import glob
import random
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader

# ───────────────────────── pairing utils ─────────────────────────

def find_paired_files(points_dir: str, wire_dir: str) -> List[Tuple[str, str]]:
    """Match .xyz and .obj files by basename (stem)."""
    xyz_paths = glob.glob(os.path.join(points_dir, "*.xyz"))
    obj_paths = glob.glob(os.path.join(wire_dir, "*.obj"))

    xyz_map = {os.path.splitext(os.path.basename(p))[0]: p for p in xyz_paths}
    obj_map = {os.path.splitext(os.path.basename(p))[0]: p for p in obj_paths}

    common_stems = sorted(set(xyz_map.keys()) & set(obj_map.keys()))
    return [(xyz_map[s], obj_map[s]) for s in common_stems] 

def select_train_test_pairs(
    pairs: List[Tuple[str, str]],
    train_n: Optional[int] = 20,
    test_n: Optional[int] = 5,
    seed: int = 42,
    shuffle: bool = True
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Select N train and M test pairs deterministically."""
    if shuffle:
        random.Random(seed).shuffle(pairs)

    if train_n is None:
        train_n = int(len(pairs) * 0.8)
    if test_n is None:
        test_n = len(pairs) - train_n

    total_needed = train_n + test_n
    if len(pairs) < total_needed:
        raise ValueError(f"Not enough paired samples. Needed {total_needed}, found {len(pairs)}.")

    return pairs[:train_n], pairs[train_n:train_n+test_n]

# ───────────────────────── dataset wrapper ─────────────────────────

class PairListDataset(Dataset):
    """Wraps your PointCloudWireframeDataset for multiple (xyz,obj) pairs."""
    def __init__(self, pair_paths: List[Tuple[str, str]], k_nearest: int = 10):
        self.pair_paths = pair_paths
        self.k_nearest = k_nearest
        self.cached_data = {}  # Cache loaded data to avoid reloading
        
        # Pre-load all data during initialization
        print(f"Pre-loading {len(pair_paths)} datasets...")
        for i, (xyz_path, obj_path) in enumerate(pair_paths):
            print(f"  Loading {i+1}/{len(pair_paths)}: {os.path.basename(xyz_path)}")
            ds = PointCloudWireframeDataset(xyz_path, obj_path)
            
            # Load all data once
            point_cloud = ds.load_point_cloud()
            vertices, edges = ds.load_wireframe()
            _adj = ds.create_adjacency_matrix()
            _normalized_pc = ds.normalize_data()
            distances, indices = ds.find_nearest_points_to_vertices(k=self.k_nearest)
            
            # Cache the data
            self.cached_data[i] = {
                "point_cloud": torch.from_numpy(ds.normalized_point_cloud).float(),  # Use normalized data
                "normalized_point_cloud": torch.from_numpy(ds.normalized_point_cloud).float(),
                "vertices": torch.from_numpy(ds.normalized_vertices).float(),  # Use normalized vertices
                "edges": torch.tensor(edges, dtype=torch.long) if len(edges) else torch.empty((0,2), dtype=torch.long),
                "adjacency": torch.from_numpy(_adj).float(),
                "nearest_dists": torch.from_numpy(distances).float(),
                "nearest_idx": torch.from_numpy(indices).long(),
                "xyz_path": xyz_path,
                "obj_path": obj_path,
            }
        print("Dataset pre-loading completed!")

    def __len__(self):
        return len(self.pair_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return cached data instead of reloading
        return self.cached_data[idx]

# ───────────────────────── high-level loader ─────────────────────────

def load_datasets(
    points_dir: str = "demo_dataset/pointcloud",
    wire_dir: str = "demo_dataset/wireframe",
    train_n: Optional[int] = 20,
    test_n: Optional[int] = 5,
    seed: int = 42,
    k_nearest: int = 10
) -> Tuple[PairListDataset, PairListDataset, List[Tuple[str, str]], List[Tuple[str, str]]]:
    pairs = find_paired_files(points_dir, wire_dir)
    train_pairs, test_pairs = select_train_test_pairs(pairs, train_n=train_n, test_n=test_n, seed=seed)
    return PairListDataset(train_pairs, k_nearest), PairListDataset(test_pairs, k_nearest), train_pairs, test_pairs

def create_dataloaders(train_ds, test_ds, batch_size=4, num_workers=0, shuffle=True, cap_num_vertices=60):
    """Create dataloaders with proper collate function for variable vertex counts."""
    collate_fn = make_pcwf_collate(cap_num_vertices)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    )

# ───────────────────────── single-file loader ─────────────────────────

def load_and_preprocess_data_single():
    dataset = PointCloudWireframeDataset('demo_dataset/pointcloud/1003.xyz',
                                         'demo_dataset/wireframe/1003.obj')
    dataset.load_point_cloud()
    dataset.load_wireframe()
    dataset.create_adjacency_matrix()
    dataset.normalize_data()
    dataset.find_nearest_points_to_vertices(k=10)
    return dataset

# ───────────────────────── main ─────────────────────────

if __name__ == "__main__":
    # Configuration
    CAP_NUM_VERTICES = 60
    BATCH_SIZE = 2  # Reduced batch size for small dataset
    NUM_EPOCHS = 100  # Reduced epochs for small dataset
    LEARNING_RATE = 1e-5  # Conservative learning rate for stability
    
    print(f"Configuration:")
    print(f"  Max vertices per sample: {CAP_NUM_VERTICES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Number of epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    print(f"\nLoading datasets...")
    train_ds, test_ds, train_pairs, test_pairs = load_datasets(
        points_dir="demo_dataset/pointcloud",
        wire_dir="demo_dataset/wireframe",
        train_n=3,
        test_n=2,
        seed=42,
        k_nearest=10
    )

    print(f"\nDataset loaded:")
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")

    # Check data normalization
    first = train_ds[0]
    print(f"\nSample data shapes and ranges:")
    print(f"  Point cloud: {first['point_cloud'].shape}")
    print(f"    Range: [{first['point_cloud'].min():.3f}, {first['point_cloud'].max():.3f}]")
    print(f"  Vertices: {first['vertices'].shape}")
    print(f"    Range: [{first['vertices'].min():.3f}, {first['vertices'].max():.3f}]")
    print(f"  Adjacency matrix: {first['adjacency'].shape}")
    print(f"    Non-zero elements: {(first['adjacency'] > 0).sum().item()}")
    if 'normalized_point_cloud' in first:
        print(f"  Normalized point cloud: {first['normalized_point_cloud'].shape}")
        print(f"    Range: [{first['normalized_point_cloud'].min():.3f}, {first['normalized_point_cloud'].max():.3f}]")
    
    # Validate data ranges
    pc_range = first['point_cloud'].abs().max().item()
    v_range = first['vertices'].abs().max().item()
    
    if pc_range > 10.0:
        print(f"⚠️  WARNING: Point cloud range is very large ({pc_range:.3f}). This may cause training issues.")
    if v_range > 10.0:
        print(f"⚠️  WARNING: Vertex range is very large ({v_range:.3f}). This may cause training issues.")
    
    if pc_range <= 2.0 and v_range <= 2.0:
        print(f"✅ Data ranges look good for training.")
    else:
        print(f"❌ Data ranges may cause training instability. Consider checking normalization.")

    print(f"\nCreating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        train_ds, test_ds, 
        batch_size=BATCH_SIZE, 
        cap_num_vertices=CAP_NUM_VERTICES
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Test the model with a single batch before training
    print(f"\nTesting model with single batch...")
    from models.PointCloudToWireframe import PointCloudToWireframe
    
    test_model = PointCloudToWireframe(input_dim=8, num_vertices=CAP_NUM_VERTICES).to(device)
    test_model.eval()
    
    with torch.no_grad():
        for test_batch in train_loader:
            pc = test_batch['point_cloud'].to(device)
            preds = test_model(pc)
            
            print(f"  Model test successful:")
            print(f"    Input shape: {pc.shape}")
            print(f"    Predicted vertices: {preds['vertices'].shape}")
            print(f"    Edge probabilities: {preds['edge_probs'].shape}")
            print(f"    Edge indices: {len(preds['edge_indices'])}")
            
            # Check prediction ranges
            v_pred_range = preds['vertices'].abs().max().item()
            e_pred_range = preds['edge_probs'].abs().max().item()
            print(f"    Vertex prediction range: [{preds['vertices'].min().item():.3f}, {preds['vertices'].max().item():.3f}]")
            print(f"    Edge prediction range: [{preds['edge_probs'].min().item():.3f}, {preds['edge_probs'].max().item():.3f}]")
            
            if v_pred_range > 10.0:
                print(f"⚠️  WARNING: Vertex predictions have large range ({v_pred_range:.3f})")
            if e_pred_range > 1.0:
                print(f"⚠️  WARNING: Edge predictions exceed [0,1] range ({e_pred_range:.3f})")
            
            break
    
    print(f"Model test completed!")

    print("\n" + "="*50)
    
    print("STARTING TRAINING")
    print("="*50)

    model, loss_hist, rmse_hist = train_models(
        train_loader=train_loader,
        cap_num_vertices=CAP_NUM_VERTICES,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )

    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved as 'trained_model.pth'")
    
    print(f"\nTraining completed!")
    print(f"  Final loss: {loss_hist[-1]:.6f}")
    print(f"  Final RMSE: {rmse_hist[-1]:.6f}")
    print(f"  Best RMSE: {min(rmse_hist):.6f}")
    
    # Plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_hist)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')  # Log scale for better visualization
        
        plt.subplot(1, 2, 2)
        plt.plot(rmse_hist)
        plt.title('Vertex RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Training curves saved as 'training_curves.png'")
        
    except ImportError:
        print("Matplotlib not available - skipping training curve plots")

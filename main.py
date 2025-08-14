import torch
from demo_dataset.PointCloudWireframeDataset import PointCloudWireframeDataset
from train_multiple import train_model, find_matching_files, split_data_into_train_test, create_data_collator, evaluate_model
from demo_dataset.MultiPairPCWF import MultiPairPCWF
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
    train_n: Optional[int] = 4,
    test_n: Optional[int] = 2,
    seed: int = 42,
    k_nearest: int = 10
) -> Tuple[PairListDataset, PairListDataset, List[Tuple[str, str]], List[Tuple[str, str]]]:
    pairs = find_paired_files(points_dir, wire_dir)
    train_pairs, test_pairs = select_train_test_pairs(pairs, train_n=train_n, test_n=test_n, seed=seed)
    return PairListDataset(train_pairs, k_nearest), PairListDataset(test_pairs, k_nearest), train_pairs, test_pairs

def create_dataloaders(train_ds, test_ds, batch_size=4, num_workers=0, shuffle=True, cap_num_vertices=60):
    """Create dataloaders with proper collate function for variable vertex counts."""
    collate_fn = create_data_collator(cap_num_vertices)
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
    import os
    import torch
    import logging
    from torch.utils.data import DataLoader

    # ───────────────────────── Config ─────────────────────────
    CAP_NUM_VERTICES = 60
    BATCH_SIZE = 2           # Small batch for small data
    NUM_EPOCHS = 100         # Few epochs for small data
    LEARNING_RATE = 1e-5     # Conservative LR for stability

    print("Configuration:")
    print(f"  Max vertices per sample: {CAP_NUM_VERTICES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Number of epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")

    # ───────────────────────── Data finding/splitting ─────────────────────────
    print("\nLoading datasets (pairing .xyz ↔ .obj)...")
    pairs = find_paired_files(
        points_dir="demo_dataset/pointcloud",
        wire_dir="demo_dataset/wireframe"
    )
    train_pairs, test_pairs = select_train_test_pairs(
        pairs, train_n=3, test_n=2, seed=42, shuffle=True
    )

    print("\nDataset loaded:")
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")

    # ───────────────────────── Quick data health check ─────────────────────────
    # Range/shape checks from old main, done by loading and normalizing a single example directly from file
    if len(train_pairs) == 0:
        raise RuntimeError("No training pairs found.")

    from demo_dataset.PointCloudWireframeDataset import PointCloudWireframeDataset

    first_xyz, first_obj = train_pairs[0]
    ds_chk = PointCloudWireframeDataset(first_xyz, first_obj)
    ds_chk.load_point_cloud()
    ds_chk.load_wireframe()
    ds_chk.create_adjacency_matrix()  # Create adjacency matrix
    ds_chk.normalize_data()  # Should set normalized_point_cloud / normalized_vertices / edge_adjacency_matrix

    import numpy as np
    print("\nSample data shapes and ranges (from first training sample):")
    print(f"  Point cloud: {ds_chk.normalized_point_cloud.shape}")
    print(f"    Range: [{np.min(ds_chk.normalized_point_cloud):.3f}, {np.max(ds_chk.normalized_point_cloud):.3f}]")
    print(f"  Vertices: {ds_chk.normalized_vertices.shape}")
    print(f"    Range: [{np.min(ds_chk.normalized_vertices):.3f}, {np.max(ds_chk.normalized_vertices):.3f}]")

    # Range warnings
    pc_range = float(np.abs(ds_chk.normalized_point_cloud).max())
    v_range = float(np.abs(ds_chk.normalized_vertices).max())
    if pc_range > 10.0:
        print(f"⚠️  WARNING: Point cloud range is very large ({pc_range:.3f}). This may cause training issues.")
    if v_range > 10.0:
        print(f"⚠️  WARNING: Vertex range is very large ({v_range:.3f}). This may cause training issues.")
    if pc_range <= 2.0 and v_range <= 2.0:
        print("✅ Data ranges look good for training.")
    else:
        print("❌ Data ranges may cause training instability. Consider checking normalization.")

    # ───────────────────────── Sanity check: single batch forward ─────────────────────────
    # Single batch model test from old main, using our collate function
    print("\nCreating a tiny loader for a single-batch model test...")
    tmp_count = min(max(1, BATCH_SIZE), len(train_pairs))
    tmp_ds = MultiPairPCWF(train_pairs[:tmp_count])
    tmp_loader = DataLoader(
        tmp_ds,
        batch_size=tmp_count,
        shuffle=False,
        collate_fn=create_data_collator(CAP_NUM_VERTICES)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print("\nTesting model with single batch...")
    from models.PointCloudToWireframe import PointCloudToWireframe
    test_model = PointCloudToWireframe(input_dim=8, num_vertices=CAP_NUM_VERTICES).to(device)
    test_model.eval()
    with torch.no_grad():
        for test_batch in tmp_loader:
            pc = test_batch['point_cloud'].to(device)
            preds = test_model(pc)
            print("  Model test successful:")
            print(f"    Input shape: {pc.shape}")
            print(f"    Predicted vertices: {preds['vertices'].shape}")
            print(f"    Edge probabilities: {preds['edge_probs'].shape}")
            print(f"    Edge indices: {len(preds['edge_indices'])}")

            v_pred_min = preds['vertices'].min().item()
            v_pred_max = preds['vertices'].max().item()
            e_pred_min = preds['edge_probs'].min().item()
            e_pred_max = preds['edge_probs'].max().item()
            print(f"    Vertex prediction range: [{v_pred_min:.3f}, {v_pred_max:.3f}]")
            print(f"    Edge prediction range: [{e_pred_min:.3f}, {e_pred_max:.3f}]")
            if abs(v_pred_max) > 10.0 or abs(v_pred_min) > 10.0:
                print(f"⚠️  WARNING: Vertex predictions have large range.")
            if e_pred_max > 1.0 or e_pred_min < 0.0:
                print(f"⚠️  WARNING: Edge predictions exceed [0,1] range.")
            break
    print("Model test completed!")

    # ───────────────────────── Training ─────────────────────────
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)

    # Use the simplified training function
    model = train_model(
        training_pairs=train_pairs,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        max_vertices=CAP_NUM_VERTICES,
        batch_size=BATCH_SIZE
    )

    # ───────────────────────── Saving ─────────────────────────
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved as 'trained_model.pth'")

    # ───────────────────────── Quick test evaluation (optional) ─────────────────────────
    # Evaluate model on test samples
    if len(test_pairs) > 0:
        print("\nQuick evaluation on a couple of test samples:")
        for i, (xyz_path, obj_path) in enumerate(test_pairs[:2]):
            ds_single = PointCloudWireframeDataset(xyz_path, obj_path)
            ds_single.load_point_cloud()
            ds_single.load_wireframe()
            ds_single.create_adjacency_matrix()  # Create adjacency matrix
            ds_single.normalize_data()
            results = evaluate_model(model, ds_single, device)
            name = os.path.splitext(os.path.basename(obj_path))[0]
            print(f"  [TEST:{name}] "
                  f"RMSE={results['vertex_rmse']:.4f} | "
                  f"Edge Acc={results['edge_accuracy']:.3f} | "
                  f"F1={results['edge_f1_score']:.3f}")

    print("\nTraining completed!")

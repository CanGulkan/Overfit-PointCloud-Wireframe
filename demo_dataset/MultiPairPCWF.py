# demo_dataset/MultiPairPCWF.py  (yalnızca ilgili kısım)
import os
import numpy as np
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
from .PointCloudWireframeDataset import PointCloudWireframeDataset

# ───────────────────────── SpatialScaler ─────────────────────────
class SpatialScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
    def fit(self, xyz: np.ndarray):
        xyz = np.asarray(xyz, dtype=np.float64)
        self.mean_ = xyz.mean(axis=0)
        self.std_  = xyz.std(axis=0) + 1e-8
        return self
    def transform(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        return (xyz - self.mean_) / self.std_
    def inverse_transform(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        return xyz * self.std_ + self.mean_

# ───────────── wireframe yardımcıları: adjacency/mapping ─────────────
def _get_vertices(ds: PointCloudWireframeDataset) -> np.ndarray:
    # vertices alan adı farklı olabilir: vertices / V / points vs.
    for name in ["normalized_vertices", "vertices", "V"]:
        if hasattr(ds, name) and getattr(ds, name) is not None:
            return np.asarray(getattr(ds, name))
    raise AttributeError("Wireframe vertices bulunamadı (vertices / normalized_vertices yok).")

def _get_adjacency_or_edges(ds: PointCloudWireframeDataset):
    """
    Aşağıdaki sırayla adjacency'i arar; yoksa edges'i döndürür:
      adjacency adları: edge_adjacency_matrix, adjacency, edge_adjacency, A
      edges adları: edges, E (E×2)
    Dönüş:
      ('adj', A)  veya  ('edges', edges)
    """
    for name in ["edge_adjacency_matrix", "adjacency", "edge_adjacency", "A"]:
        if hasattr(ds, name) and getattr(ds, name) is not None:
            A = np.asarray(getattr(ds, name))
            return "adj", A
    for name in ["edge_adjacency_matrix_", "adjacency_matrix"]:  # olası varyantlar
        if hasattr(ds, name) and getattr(ds, name) is not None:
            A = np.asarray(getattr(ds, name))
            return "adj", A
    for name in ["edges", "E"]:
        if hasattr(ds, name) and getattr(ds, name) is not None:
            e = np.asarray(getattr(ds, name))
            return "edges", e
    return None, None

def _edges_to_adjacency(edges: np.ndarray, num_vertices: int) -> np.ndarray:
    A = np.zeros((num_vertices, num_vertices), dtype=np.float32)
    if edges.size == 0:
        return A
    edges = np.asarray(edges, dtype=int)
    for i, j in edges:
        if 0 <= i < num_vertices and 0 <= j < num_vertices:
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A

# ───────────────────── normalize helper ─────────────────────
def normalize_inplace(ds: PointCloudWireframeDataset):
    """
    Garanti eder:
      - ds.normalized_point_cloud : (N, >=3) — XYZ normalize edilir (ilk 3 sütun)
      - ds.normalized_vertices    : (V, 3)    — XYZ normalize edilir
      - ds.edge_adjacency_matrix  : (V, V)    — yoksa edges'ten kurulur
      - ds.spatial_scaler         : SpatialScaler
    """
    # hazırsa çık
    if (hasattr(ds, "normalized_point_cloud") and ds.normalized_point_cloud is not None and
        hasattr(ds, "normalized_vertices") and ds.normalized_vertices is not None and
        hasattr(ds, "spatial_scaler") and ds.spatial_scaler is not None and
        hasattr(ds, "edge_adjacency_matrix") and ds.edge_adjacency_matrix is not None):
        return

    # ham alan kontrolleri
    if not hasattr(ds, "point_cloud") or ds.point_cloud is None:
        raise AttributeError("normalize_inplace: ds.point_cloud yok. Önce load_point_cloud() çağırın.")
    pc = np.asarray(ds.point_cloud, dtype=np.float64)
    if pc.shape[1] < 3:
        raise ValueError("Point cloud en az 3 sütun (XYZ) içermeli.")

    # vertices çek
    V = _get_vertices(ds)  # normalized_vertices varsa onu da döndürebilir
    if V.shape[1] != 3:
        raise ValueError("Vertices (V) 3 sütunlu (XYZ) olmalı.")

    # adjacency veya edges'i bul
    kind, A_or_E = _get_adjacency_or_edges(ds)
    if kind is None:
        raise AttributeError(
            "normalize_inplace: adjacency ya da edges bulunamadı. "
            "Dataset'in load_wireframe() metodu adjacency (VxV) veya edges (Ex2) üretmeli."
        )

    # scaler'ı point cloud XYZ'e fit et (vertex'i de aynı scaler ile normalize edeceğiz)
    scaler = SpatialScaler().fit(pc[:, :3])

    # point cloud normalize (XYZ)
    pc_norm = pc.copy()
    pc_norm[:, :3] = scaler.transform(pc[:, :3])

    # vertices normalize (XYZ)
    V_norm = scaler.transform(V)

    # adjacency sağla
    if kind == "adj":
        A = np.asarray(A_or_E, dtype=np.float32)
        # boyut eşitle: V_norm sayısı ile uyumsuzsa truncate/pad
        Vn = V_norm.shape[0]
        if A.shape != (Vn, Vn):
            A_fixed = np.zeros((Vn, Vn), dtype=np.float32)
            m = min(Vn, A.shape[0])
            A_fixed[:m, :m] = A[:m, :m]
            A = A_fixed
    else:  # 'edges'
        edges = np.asarray(A_or_E, dtype=int)
        A = _edges_to_adjacency(edges, num_vertices=V_norm.shape[0]).astype(np.float32)

    # alanları yaz
    ds.normalized_point_cloud = pc_norm.astype(np.float32)
    ds.normalized_vertices = V_norm.astype(np.float32)
    ds.edge_adjacency_matrix = A.astype(np.float32)
    ds.spatial_scaler = scaler

# ───────────────────── Çoklu örnek dataset ─────────────────────
class MultiPairPCWF(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.items = []
        for xyz_path, obj_path in pairs:
            ds = PointCloudWireframeDataset(xyz_path, obj_path)
            ds.load_point_cloud()
            ds.load_wireframe()
            normalize_inplace(ds)  # <-- Artık adjacency adı ne olursa olsun çözülür.

            name = os.path.splitext(os.path.basename(obj_path))[0]
            self.items.append({
                "name": name,
                "pc": ds.normalized_point_cloud,     # [N, >=3]
                "V":  ds.normalized_vertices,        # [V, 3]
                "A":  ds.edge_adjacency_matrix,      # [V, V]
                "scaler": ds.spatial_scaler,         # SpatialScaler
            })

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

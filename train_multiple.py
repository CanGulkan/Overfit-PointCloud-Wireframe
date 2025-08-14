import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import logging

from models.PointCloudToWireframe import PointCloudToWireframe
from losses.WireframeLoss import WireframeLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ───────────────────────── collate (multi-file training support) ─────────────────────────

def make_pcwf_collate(cap_num_vertices: int):
    """
    Returns a collate_fn that:
      - pads/truncates vertices & adjacency to cap_num_vertices
      - pads point clouds in the batch to the batch's max N
      - emits:
          vertex_counts        -> used (truncated) vertex counts per sample
          point_counts         -> raw point counts per sample (for logging)
          vertex_raw_counts    -> raw vertex counts per sample (for logging)
      - preserves 'xyz_path' and 'obj_path' lists (if present) so we can print filenames
    """
    cap = cap_num_vertices

    def collate(batch):
        B = len(batch)

        # Per-sample sizes
        Ns = [item["point_cloud"].shape[0] for item in batch]     # raw N_i
        Vs_raw = [item["vertices"].shape[0] for item in batch]    # raw V_i
        Vs = [min(v, cap) for v in Vs_raw]                        # truncated to cap
        N_max = max(Ns)

        # Allocate padded tensors
        pc_pad   = torch.zeros(B, N_max, batch[0]["point_cloud"].shape[1], dtype=torch.float32)
        v_pad    = torch.zeros(B, cap, 3, dtype=torch.float32)
        adj_pad  = torch.zeros(B, cap, cap, dtype=torch.float32)
        vcounts  = torch.tensor(Vs, dtype=torch.long)         # [B] used vertices
        ncounts  = torch.tensor(Ns, dtype=torch.long)         # [B] raw points (for logs)
        vraw     = torch.tensor(Vs_raw, dtype=torch.long)     # [B] raw vertices (for logs)

        # Fill pads
        for b, item in enumerate(batch):
            N_i, V_i = Ns[b], Vs[b]
            pc_pad[b, :N_i, :] = item["point_cloud"]
            if V_i > 0:
                v_pad[b, :V_i, :] = item["vertices"][:V_i, :]
                adj_pad[b, :V_i, :V_i] = item["adjacency"][:V_i, :V_i]

        # Keep anything else as lists
        keep_list_keys = ["edges", "nearest_dists", "nearest_idx", "xyz_path", "obj_path"]
        out = {
            "point_cloud": pc_pad,
            "vertices": v_pad,
            "adjacency": adj_pad,
            "vertex_counts": vcounts,
            "point_counts": ncounts,           # NEW
            "vertex_raw_counts": vraw,         # NEW
        }
        for k in keep_list_keys:
            if k in batch[0]:
                out[k] = [it[k] for it in batch]
        return out

    return collate


# ───────────────────────── helpers ─────────────────────────

def create_adjacency_matrix_from_predictions(edge_probs, edge_indices, num_vertices, threshold=0.5):
    """Convert edge predictions to adjacency matrix"""
    batch_size = edge_probs.shape[0]
    adj_matrices = torch.zeros(batch_size, num_vertices, num_vertices, device=edge_probs.device)
    for b in range(batch_size):
        for e_idx, (i, j) in enumerate(edge_indices):
            if edge_probs[b, e_idx] > threshold:
                adj_matrices[b, i, j] = 1
                adj_matrices[b, j, i] = 1
    return adj_matrices


def create_edge_labels_from_adjacency(adj_matrix_b, edge_indices):
    """Create edge labels tensor from adjacency matrix for a batch"""
    B = adj_matrix_b.shape[0]
    num_edges = len(edge_indices)
    edge_labels = adj_matrix_b.new_zeros((B, num_edges))
    for e_idx, (i, j) in enumerate(edge_indices):
        edge_labels[:, e_idx] = adj_matrix_b[:, i, j]
    return edge_labels


def _edge_indices_upto(v: int):
    """Generate edge indices for vertices 0..v-1 where i<j"""
    return [(i, j) for i in range(v) for j in range(i + 1, v)]


def _edge_labels_from_adj(adj_bxvxv: torch.Tensor, edge_indices):
    """Convert adjacency matrix to edge labels for a single sample"""
    device = adj_bxvxv.device
    E = len(edge_indices)
    y = torch.zeros((E,), device=device, dtype=adj_bxvxv.dtype)
    for e_idx, (i, j) in enumerate(edge_indices):
        y[e_idx] = adj_bxvxv[i, j]
    return y


# ───────────────────────── training ─────────────────────────

def train_models(
    train_loader,
    cap_num_vertices,
    num_epochs: int = 2000,
    learning_rate: float = 1e-3,
    device=None,
    show_paths: bool = True,   # print which files are in each batch
    paths_every: int = 1       # print every k batches
):
    """
    Multiple training with variable vertex counts. Model is built with fixed 'cap_num_vertices'.
    Dataloader must provide padded tensors and 'vertex_counts'.
    If dataset provides 'obj_path'/'xyz_path', they will be printed per batch when show_paths=True.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    # Model (fixed architecture based on cap)
    model = PointCloudToWireframe(input_dim=8, num_vertices=cap_num_vertices).to(device)

    # Loss, optimizer, scheduler
    criterion = WireframeLoss(vertex_weight=1.0, edge_weight=1.0)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[400, 700, 1700, 4000], gamma=0.3
    )

    best_loss = float('inf')
    best_rmse = float('inf')
    best_model_state = None
    loss_history, rmse_history = [], []
    patience = 50
    no_improve = 0

    # Early stopping for exploding loss
    max_loss_threshold = 1e6

    model.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_rmse = 0.0
        batch_count = 0

        for bi, batch in enumerate(train_loader):
            # ── NEW: show which files are in this batch (names + sizes) ──
            if show_paths and isinstance(batch, dict):
                if ('obj_path' in batch) and ('point_counts' in batch) and ('vertex_raw_counts' in batch):
                    names = [os.path.basename(p) for p in batch['obj_path']]
                    Ns    = batch['point_counts'].tolist() if isinstance(batch['point_counts'], torch.Tensor) else batch['point_counts']
                    Vraw  = batch['vertex_raw_counts'].tolist() if isinstance(batch['vertex_raw_counts'], torch.Tensor) else batch['vertex_raw_counts']
                    Vused = batch['vertex_counts'].tolist() if isinstance(batch['vertex_counts'], torch.Tensor) else batch['vertex_counts']
                    if bi % paths_every == 0:
                        print(f"\n[Epoch {epoch} | Batch {bi+1}] Processing {len(names)} sample(s):")
                        for nm, n_i, v_r, v_u in zip(names, Ns, Vraw, Vused):
                            print(f"  - {nm:<32}  N={n_i:<6}  V_raw={v_r:<4}  V_used={v_u:<4}")
            # ──────────────────────────────────────────────────────────────

            if isinstance(batch, dict):
                pc = batch['point_cloud']      # [B, N, 8]
                tgt_v = batch['vertices']      # [B, cap, 3]
                tgt_adj = batch['adjacency']   # [B, cap, cap]
                vcounts = batch['vertex_counts']  # [B]
            else:
                pc, tgt_v, tgt_adj, vcounts = batch

            pc     = pc.to(device, dtype=torch.float32)
            tgt_v  = tgt_v.to(device, dtype=torch.float32)
            tgt_adj= tgt_adj.to(device, dtype=torch.float32)
            vcounts= vcounts.to(device, dtype=torch.long)

            optimizer.zero_grad()
            preds = model(pc)  # {'vertices':[B,cap,3], 'edge_probs':[B,E_cap], 'edge_indices':...}

            B = pc.shape[0]
            total_loss = 0.0
            total_rmse = 0.0

            # Per-sample slicing to each sample's V_i and E_i
            for b in range(B):
                V_i = int(vcounts[b].item())
                if V_i < 2:  # can't define edges; still learn vertices
                    V_i = max(V_i, 1)

                pred_v_i = preds['vertices'][b, :V_i, :]              # [V_i, 3]
                tgt_v_i  = tgt_v[b, :V_i, :]                           # [V_i, 3]

                edge_idx_i = _edge_indices_upto(V_i)
                E_i = len(edge_idx_i)

                pred_edge_probs_i = preds['edge_probs'][b, :E_i]       # [E_i]
                tgt_edge_labels_i = _edge_labels_from_adj(tgt_adj[b, :V_i, :V_i], edge_idx_i)  # [E_i]

                preds_i = {
                    'vertices': pred_v_i.unsqueeze(0),                 # [1, V_i, 3]
                    'edge_probs': pred_edge_probs_i.unsqueeze(0),      # [1, E_i]
                    'edge_indices': edge_idx_i
                }
                targets_i = {
                    'vertices': tgt_v_i.unsqueeze(0),                  # [1, V_i, 3]
                    'edge_labels': tgt_edge_labels_i.unsqueeze(0)      # [1, E_i]
                }

                loss_dict_i = criterion(preds_i, targets_i)
                loss_i = loss_dict_i['total_loss']

                # We backward per-sample; graph is shared across samples, so retain
                loss_i.backward(retain_graph=True)
                total_loss += loss_i.item()

                rmse_i = torch.sqrt(torch.mean((pred_v_i - tgt_v_i) ** 2)).item()
                total_rmse += rmse_i

            # Combined gradient step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            batch_count += 1
            epoch_loss += (total_loss / B)
            epoch_rmse += (total_rmse / B)

        scheduler.step()

        # Epoch averages
        if batch_count > 0:
            epoch_loss /= batch_count
            epoch_rmse /= batch_count

        loss_history.append(epoch_loss)
        rmse_history.append(epoch_rmse)

        # Early stop for exploding loss
        if epoch_loss > max_loss_threshold:
            logger.warning(f"Loss exploded to {epoch_loss:.2e} at epoch {epoch}. Stopping training.")
            break

        # Make sure elapsed is always defined before printing
        elapsed = time.time() - start_time

        # Console line every epoch
        print(
            f"Epoch {epoch:4d}/{num_epochs} | "
            f"Avg Loss: {epoch_loss:.6f} | "
            f"Avg Vertex RMSE (norm): {epoch_rmse:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Best RMSE: {best_rmse:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Track best
        improved = False
        if epoch_rmse < best_rmse:
            best_rmse = epoch_rmse
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            improved = True
        else:
            no_improve += 1

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            logger.info(
                f"Epoch {epoch:4d}/{num_epochs} | "
                f"Avg Loss: {epoch_loss:.6f} | "
                f"Avg Vertex RMSE (norm): {epoch_rmse:.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Best RMSE: {best_rmse:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        if no_improve >= patience and epoch > 20:
            logger.info(f"Early stopping at epoch {epoch}! RMSE hasn't improved for {patience} epochs.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model state with Vertex RMSE (norm): {best_rmse:.6f}")

    logger.info(f"Training completed! Best avg loss: {best_loss:.6f}")
    return model, loss_history, rmse_history

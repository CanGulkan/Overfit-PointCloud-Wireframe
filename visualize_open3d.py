import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import open3d as o3d

from main import load_and_preprocess_data
from models.PointCloudToWireframe import PointCloudToWireframe

# ───────────────────────── model/data ─────────────────────────

def load_trained_model():
    """Load the trained model and dataset."""
    print("Loading trained model...")
    dataset = load_and_preprocess_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointCloudToWireframe(input_dim=8, num_vertices=len(dataset.vertices)).to(device)
    model.load_state_dict(torch.load('trained_model.pth', map_location=device))
    model.eval()
    print(f"✓ Model loaded successfully on {device}")
    return model, dataset, device

# ──────────────────────── viz helpers ─────────────────────────

def scene_extent_from_geoms(geoms):
    pts = []
    for g in geoms:
        if isinstance(g, o3d.geometry.PointCloud):
            pts.append(np.asarray(g.points))
        elif isinstance(g, o3d.geometry.LineSet):
            pts.append(np.asarray(g.points))
        elif isinstance(g, o3d.geometry.TriangleMesh):
            pts.append(np.asarray(g.vertices))
    if not pts:
        return 1.0
    pts = np.vstack([p for p in pts if len(p) > 0])
    ext = np.linalg.norm(pts.max(0) - pts.min(0))
    return float(max(ext, 1e-6))
1
def setup_view(geometries, title="Wireframe View", width=1600, height=900,
               bg=(1,1,1), point_size=3000.0, line_width=1.4, screenshot_path=None):
    """Create an Open3D window with global line/point sizes."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=width, height=height, visible=True)

    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg, dtype=np.float32)
    opt.point_size = float(point_size)
    opt.line_width = float(line_width)

    # Auto-frame camera to fit scene
    vc = vis.get_view_control()
    bbox = o3d.geometry.AxisAlignedBoundingBox()
    first = True
    for g in geometries:
        try:
            b = g.get_axis_aligned_bounding_box()
            bbox = b if first else (bbox + b)
            first = False
        except Exception:
            pass
    if not first:
        vc.set_lookat(bbox.get_center())
        vc.set_front([0.3, -0.5, 0.8])
        vc.set_up([0.0, 0.0, 1.0])
        vc.set_zoom(0.7)

    vis.poll_events()
    vis.update_renderer()

    if screenshot_path:
        vis.capture_screen_image(screenshot_path, do_render=True)

    vis.run()
    vis.destroy_window()

def create_red_wireframe(vertices, edges):
    """Thin red lines."""
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(vertices))
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(edges), 1)))
    return ls

def create_blue_vertex_dots(vertices, color=(0.0, 0.1, 0.9)):
    """Blue vertex markers as a point cloud (size set globally)."""
    V = np.asarray(vertices, dtype=np.float32)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(V)
    pc.paint_uniform_color(np.asarray(color, dtype=np.float32))
    return pc


def create_yellow_edges(vertices, edges_subset):
    """Optional: highlight edges in yellow."""
    if edges_subset is None or len(edges_subset) == 0:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(vertices))
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(edges_subset, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.85, 0.0], (len(edges_subset), 1)))
    return ls

# ─────────────────────── visualizations ───────────────────────

def visualize_ground_truth(point_px=2.0, line_px=1.4):
    ds = load_and_preprocess_data()
    red = create_red_wireframe(ds.vertices, ds.edges)
    blue = create_blue_vertex_dots(ds.vertices)
    setup_view([red, blue], title="GT (red edges + blue vertices)",
               point_size=point_px, line_width=line_px)

def visualize_predicted(threshold=0.5, highlight_mid_conf=False, point_px=2.0, line_px=1.4):
    model, ds, device = load_trained_model()
    with torch.no_grad():
        pc_t = torch.as_tensor(ds.normalized_point_cloud, dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(pc_t)
        V = pred['vertices'].detach().cpu().numpy()[0]
        V = ds.spatial_scaler.inverse_transform(V)
        probs = pred['edge_probs'].detach().cpu().numpy()[0]
        E = np.asarray(pred['edge_indices'], dtype=np.int32)

    keep = np.where(probs > float(threshold))[0]
    E_keep = E[keep]

    geoms = [create_red_wireframe(V, E_keep), create_blue_vertex_dots(V, color=(0.0, 0.5, 1.0))]

    if highlight_mid_conf:
        band = (probs >= 0.35) & (probs <= 0.55)
        yellow = create_yellow_edges(V, E[np.where(band)[0]])
        if yellow is not None:
            geoms.append(yellow)

    setup_view(geoms, title=f"Predicted (thr={threshold:.2f})",
               point_size=point_px, line_width=line_px)

def visualize_overlay(threshold=0.5, point_px=2.0, line_px=1.3):
    model, ds, device = load_trained_model()
    with torch.no_grad():
        pc_t = torch.as_tensor(ds.normalized_point_cloud, dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(pc_t)
        Vp = pred['vertices'].detach().cpu().numpy()[0]
        Vp = ds.spatial_scaler.inverse_transform(Vp)
        probs = pred['edge_probs'].detach().cpu().numpy()[0]
        Ep = np.asarray(pred['edge_indices'], dtype=np.int32)

    keep = np.where(probs > float(threshold))[0]
    Ep_keep = Ep[keep]

    gt_red   = create_red_wireframe(ds.vertices, ds.edges)
    gt_blue  = create_blue_vertex_dots(ds.vertices, color=(0.0, 0.1, 0.9))
    pred_red = create_red_wireframe(Vp, Ep_keep)
    pred_blue= create_blue_vertex_dots(Vp, color=(0.0, 0.5, 1.0))

    setup_view([gt_red, gt_blue, pred_red, pred_blue],
               title=f"Overlay (GT + Pred, thr={threshold:.2f})",
               point_size=point_px, line_width=line_px)

def save_overlay_image(threshold=0.5, out_path="wireframe.png", point_px=2.0, line_px=1.6):
    model, ds, device = load_trained_model()
    with torch.no_grad():
        pc_t = torch.as_tensor(ds.normalized_point_cloud, dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(pc_t)
        Vp = pred['vertices'].detach().cpu().numpy()[0]
        Vp = ds.spatial_scaler.inverse_transform(Vp)
        probs = pred['edge_probs'].detach().cpu().numpy()[0]
        Ep = np.asarray(pred['edge_indices'], dtype=np.int32)

    keep = np.where(probs > float(threshold))[0]
    Ep_keep = Ep[keep]

    geoms = [
        create_red_wireframe(ds.vertices, ds.edges),
        create_blue_vertex_dots(ds.vertices, color=(0.0, 0.1, 0.9)),
        create_red_wireframe(Vp, Ep_keep),
        create_blue_vertex_dots(Vp, color=(0.0, 0.5, 1.0)),
    ]
    setup_view(geoms, title="Saving…", point_size=point_px, line_width=line_px,
               screenshot_path=out_path)
    print(f"✓ Saved: {out_path}")

# ─────────────────────────── CLI ────────────────────────────

def print_menu():
    print("\nChoose visualization:")
    print("1. Ground Truth (red edges + blue vertices)")
    print("2. Predicted (red edges + blue vertices)")
    print("3. Overlay (GT + Pred)")
    print("4. Save overlay as PNG")
    print("0. Exit")

if __name__ == "__main__":
    print("="*72)
    print("OPEN3D WIREFRAME VISUALIZATION")
    print("Style: thin red edges on white, tiny blue vertex dots")
    print("="*72)

    while True:
        try:
            print_menu()
            choice = input("\nEnter your choice (0-4): ").strip()
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                visualize_ground_truth(point_px=2.0, line_px=1.4)
            elif choice == "2":
                thr = 0.7
                visualize_predicted(threshold=float(thr), highlight_mid_conf=False,
                                     point_px=2.0, line_px=1.4)
            elif choice == "3":
                thr = 0.7
                visualize_overlay(threshold=float(thr), point_px=2.0, line_px=1.3)
            elif choice == "4":
                thr = 0.7
                path = input("Output path [wireframe.png]: ").strip() or "wireframe.png"
                save_overlay_image(threshold=float(thr), out_path=path, point_px=2.0, line_px=1.6)
            else:
                print("Invalid choice. Please enter 0-4.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Make sure 'trained_model.pth' exists and run again.")
        except Exception as e:
            print(f"Error: {e}")
            print("If this is a model issue, train first via 'python main.py'.")

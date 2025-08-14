import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import logging


# Reduce logging verbosity - only show warnings and errors
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class PointCloudWireframeDataset:
    """Dataset class for loading and preprocessing point cloud and wireframe data"""
    
    def __init__(self, xyz_file, obj_file):
        self.xyz_file = xyz_file
        self.obj_file = obj_file
        self.point_cloud = None
        self.vertices = None
        self.edges = None
        self.edge_adjacency_matrix = None
        
    def load_point_cloud(self):
        """Load point cloud data from XYZ file"""
        logger.info(f"Loading point cloud from {self.xyz_file}")
        data = []
        
        with open(self.xyz_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:  # X Y Z R G B A Intensity
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b, a = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                    intensity = float(parts[7])
                    data.append([x, y, z, r, g, b, a, intensity])
        
        self.point_cloud = np.array(data)
        logger.info(f"Loaded {len(self.point_cloud)} points")
        return self.point_cloud
    
    def load_wireframe(self):
        """Load wireframe data from OBJ file"""
        logger.info(f"Loading wireframe from {self.obj_file}")
        vertices = []
        edges = []
        
        with open(self.obj_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                    
                if parts[0] == 'v':  # Vertex
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
                elif parts[0] == 'l':  # Line (edge)
                    # OBJ format uses 1-based indexing
                    v1, v2 = int(parts[1]) - 1, int(parts[2]) - 1
                    edges.append([v1, v2])
        
        self.vertices = np.array(vertices)
        self.edges = np.array(edges)
        
        logger.info(f"Loaded {len(self.vertices)} vertices and {len(self.edges)} edges")
        return self.vertices, self.edges
    
    def create_adjacency_matrix(self):
        """Create adjacency matrix from edges"""
        if self.vertices is None or self.edges is None:
            raise ValueError("Must load wireframe data first")
            
        n_vertices = len(self.vertices)
        self.edge_adjacency_matrix = np.zeros((n_vertices, n_vertices))
        
        for edge in self.edges:
            v1, v2 = edge[0], edge[1]
            self.edge_adjacency_matrix[v1, v2] = 1
            self.edge_adjacency_matrix[v2, v1] = 1  # Undirected graph
            
        return self.edge_adjacency_matrix
    
    def normalize_data(self):
        """Normalize point cloud and vertex coordinates to prevent huge loss values"""
        if self.point_cloud is None:
            raise ValueError("Must load point cloud first")
            
        # Normalize spatial coordinates (X, Y, Z) to [-1, 1] range
        spatial_coords = self.point_cloud[:, :3]
        spatial_min = spatial_coords.min(axis=0)
        spatial_max = spatial_coords.max(axis=0)
        spatial_range = spatial_max - spatial_min
        
        # Avoid division by zero
        spatial_range = np.where(spatial_range == 0, 1.0, spatial_range)
        
        # Normalize to [-1, 1] range
        normalized_spatial = 2.0 * (spatial_coords - spatial_min) / spatial_range - 1.0
        
        # Normalize color values (R, G, B, A) to [0, 1]
        color_vals = self.point_cloud[:, 3:7] / 255.0
        
        # Normalize intensity to [0, 1] range
        intensity = self.point_cloud[:, 7:8]
        intensity_min = intensity.min()
        intensity_max = intensity.max()
        intensity_range = intensity_max - intensity_min
        if intensity_range > 0:
            normalized_intensity = (intensity - intensity_min) / intensity_range
        else:
            normalized_intensity = intensity * 0.0  # Set to 0 if no variation
        
        # Combine normalized features
        self.normalized_point_cloud = np.hstack([
            normalized_spatial, color_vals, normalized_intensity
        ])
        
        # Normalize vertex coordinates using same spatial normalization
        if self.vertices is not None:
            # Use the same spatial normalization as point cloud
            normalized_vertices = 2.0 * (self.vertices - spatial_min) / spatial_range - 1.0
            self.normalized_vertices = normalized_vertices
        
        # Store normalization parameters for later use
        self.normalization_params = {
            'spatial_min': spatial_min,
            'spatial_max': spatial_max,
            'spatial_range': spatial_range,
            'intensity_min': intensity_min,
            'intensity_max': intensity_max
        }
        
        logger.info("Data normalization completed")
        logger.info(f"Spatial range: [{spatial_min.min():.3f}, {spatial_max.max():.3f}] -> [-1, 1]")
        logger.info(f"Intensity range: [{intensity_min:.3f}, {intensity_max:.3f}] -> [0, 1]")
        return self.normalized_point_cloud
    
    def find_nearest_points_to_vertices(self, k=5):
        """Find k nearest points for each vertex in the wireframe"""
        if self.normalized_point_cloud is None or self.normalized_vertices is None:
            raise ValueError("Must normalize data first")
            
        # Use only spatial coordinates for nearest neighbor search
        point_spatial = self.normalized_point_cloud[:, :3]
        vertex_spatial = self.normalized_vertices[:, :3]
        
        # Find k nearest points for each vertex
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(point_spatial)
        distances, indices = nbrs.kneighbors(vertex_spatial)
        
        self.vertex_to_points_mapping = {
            'distances': distances,
            'indices': indices
        }
        
        logger.info(f"Found {k} nearest points for each of {len(self.normalized_vertices)} vertices")
        return distances, indices

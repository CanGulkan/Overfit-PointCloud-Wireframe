import torch
import torch.nn as nn

class VertexPredictorExplicit(nn.Module):
    """Same over-capacity vertex predictor, written without nn.Sequential."""
    
    def __init__(self, global_feature_dim=512, num_vertices=32, vertex_dim=3):
        super(VertexPredictorExplicit, self).__init__()
        
        self.num_vertices = num_vertices
        self.vertex_dim = vertex_dim
        
        # ─── Block 1 ─────────────────────────────────────────────────────────
        self.fc1          = nn.Linear(global_feature_dim, 2048)
        self.ln1          = nn.LayerNorm(2048)
        self.relu1        = nn.ReLU(inplace=True)
        self.dropout1     = nn.Dropout(0.0)
        
        # ─── Block 2 ─────────────────────────────────────────────────────────
        self.fc2          = nn.Linear(2048, 1024)
        self.ln2          = nn.LayerNorm(1024)
        self.relu2        = nn.ReLU(inplace=True)
        self.dropout2     = nn.Dropout(0.0)
        
        # ─── Block 3 ─────────────────────────────────────────────────────────
        self.fc3          = nn.Linear(1024, 1024)
        self.ln3          = nn.LayerNorm(1024)
        self.relu3        = nn.ReLU(inplace=True)
        self.dropout3     = nn.Dropout(0.0)
        
        # ─── Block 4 ─────────────────────────────────────────────────────────
        self.fc4          = nn.Linear(1024, 512)
        self.ln4          = nn.LayerNorm(512)
        self.relu4        = nn.ReLU(inplace=True)
        self.dropout4     = nn.Dropout(0.0)
        
        # ─── Final regression ────────────────────────────────────────────────
        self.final_layer  = nn.Linear(512, num_vertices * vertex_dim)
        
        # ─── Residual projections ────────────────────────────────────────────
        self.residual_proj1 = nn.Linear(global_feature_dim, 1024)
        self.residual_proj2 = nn.Linear(global_feature_dim, 512)
        
    def forward(self, global_features):
        # Block 1
        x = self.fc1(global_features)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # First residual + Block 3
        res1 = self.residual_proj1(global_features)  # (B, 1024)
        x    = self.fc3(x)
        x    = self.ln3(x)
        x    = self.relu3(x)
        x    = self.dropout3(x)
        x    = x + res1
        
        # Second residual + Block 4
        res2 = self.residual_proj2(global_features)  # (B, 512)
        x    = self.fc4(x)
        x    = self.ln4(x)
        x    = self.relu4(x)
        x    = self.dropout4(x)
        x    = x + res2
        
        # Final prediction & reshape
        vertex_coords = self.final_layer(x)  
        vertex_coords = vertex_coords.view(-1, self.num_vertices, self.vertex_dim)
        return vertex_coords

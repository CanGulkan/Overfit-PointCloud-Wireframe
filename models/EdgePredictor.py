import torch
import torch.nn as nn

class EdgePredictorExplicit(nn.Module):
    """Predict edge connectivity between vertices, without using nn.Sequential."""
    
    def __init__(self, vertex_dim=3, hidden_dim=128):
        super(EdgePredictorExplicit, self).__init__()
        
        # First linear layer + activation + dropout
        self.fc1      = nn.Linear(vertex_dim * 2, hidden_dim)
        self.relu1    = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second linear layer + activation + dropout
        self.fc2      = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2    = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Final linear layer + sigmoid to output a probability
        self.fc3      = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, vertices):
        # vertices: (batch_size, num_vertices, vertex_dim)
        batch_size, num_vertices, vertex_dim = vertices.shape
        
        # Build list of edge‐feature tensors and record their index pairs
        edges = []
        edge_indices = []
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                v1 = vertices[:, i, :]  # (batch_size, vertex_dim)
                v2 = vertices[:, j, :]  # (batch_size, vertex_dim)
                edge_feature = torch.cat([v1, v2], dim=1)  # (batch_size, 2*vertex_dim)
                edges.append(edge_feature)
                edge_indices.append((i, j))
        
        # Stack into (batch_size, num_edges, 2*vertex_dim)
        edge_features = torch.stack(edges, dim=1)
        _, num_edges, _ = edge_features.shape
        
        # Flatten for MLP: (batch_size * num_edges, 2*vertex_dim)
        edge_features = edge_features.view(-1, vertex_dim * 2)
        
        # Pass through the unpacked MLP
        x = self.fc1(edge_features)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        edge_probs = self.sigmoid(x)  # (batch_size * num_edges, 1)
        
        # Reshape back to (batch_size, num_edges)
        edge_probs = edge_probs.view(batch_size, num_edges)
        
        return edge_probs, edge_indices

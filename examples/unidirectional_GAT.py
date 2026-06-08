import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class MLP(nn.Module):
    def __init__(self, d_embd, dropout):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(d_embd, d_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_embd, d_embd)
        )

    def forward(self, x):
        h = self.ffwd(x)
        return h + x


class GATLayer(nn.Module):
    def __init__(self, d_embd, n_heads, dropout):
        super().__init__()
        self.gat = GATv2Conv(d_embd, d_embd, heads=n_heads, dropout=dropout, concat=True, edge_dim=1)
        self.proj = nn.Linear(n_heads * d_embd, d_embd)
        self.ln = nn.LayerNorm(d_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_idx, edge_attr):
        h = self.ln(x)
        h = self.gat(h, edge_idx, edge_attr)  # (N, n_heads*d_embd)
        h = self.proj(h)                       # (N, d_embd)
        return h + x



class GAT_model_single(nn.Module):
    def __init__(self, n_cells, n_features, n_targets, clks, d_embd, n_heads, n_layers, dropout):
        super().__init__()
        self.clks = clks

        self.cell_embedding = nn.Embedding(n_cells, d_embd)
        self.input_proj = nn.Linear(n_features-1, d_embd)   # cell_id has separate embedding

        if clks:
            self.clk_embd = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, d_embd)
            )

        # GAT layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'gat': GATLayer(d_embd, n_heads, dropout),
                'mlp': MLP(d_embd, dropout),
            })
            for _ in range(n_layers)
        ])


        self.output_proj = nn.Linear(d_embd, n_targets)

    def forward(self, x, edge_idx, edge_attr, clk_period, batch_idx):

        # embed
        cell_id = x[:, 0].long()
        features = x[:, 1:]
        h = self.input_proj(features) + self.cell_embedding(cell_id)

        if self.clks:

            clk_embd = self.clk_embd(clk_period[:, None])   # (G) -> (G C); G=number of graphs in batch
            clk_node = clk_embd[batch_idx] # (G C) -> (N C); batch_idx maps node to graph
            h = h + clk_node


        # GAT
        for layer in self.layers:
            h = layer['gat'](h, edge_idx, edge_attr)  # GAT + residual
            h = layer['mlp'](h)  # MLP + residual

        # out projection
        h = self.output_proj(h)

        return h
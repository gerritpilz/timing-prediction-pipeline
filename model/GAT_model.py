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

class embed_mlp(nn.Module):
    def __init__(self, d_embd, d_hidden):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_embd)
        )
    def forward(self, x):
        h = self.ffwd(x)
        return h


class BiGATLayer(nn.Module):
    def __init__(self, d_embd, n_heads, dropout):
        super().__init__()
        self.fwd_gat = GATv2Conv(d_embd, d_embd, heads=n_heads, dropout=dropout, concat=True, edge_dim=1)  # concat=true: (N n_heads*d_embd)
        self.bwd_gat = GATv2Conv(d_embd, d_embd, heads=n_heads, dropout=dropout, concat=True, edge_dim=1)
        self.proj = nn.Linear(2 * n_heads * d_embd, d_embd)
        self.ln = nn.LayerNorm(d_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_idx, edge_attr):

        # reversed edges for backward pass
        rev_edge_idx = edge_idx[[1, 0]]   # edge_idx: (2 E)

        h = self.ln(x)

        h_fwd = self.fwd_gat(h, edge_idx, edge_attr)  # (N n_heads*d_embd)
        h_bwd = self.bwd_gat(h, rev_edge_idx, edge_attr)

        h = torch.cat([h_fwd, h_bwd], dim=-1)  # (N 2*heads*d_embd)
        h = self.proj(h)  # (N d_embd)

        return h + x

class GAT_model(nn.Module):
    def __init__(self, n_cells, n_features, n_targets, clks, d_embd, n_heads, n_layers, dropout):
        super().__init__()
        self.clks = clks

        self.cell_embed = nn.Embedding(n_cells, d_embd)
        self.input_proj = nn.Linear(n_features-2, d_embd)   # cell_id, cell_strength have separate embedding

        self.strength_embed = embed_mlp(d_embd, 32)

        if clks:
            self.clk_embed = embed_mlp(d_embd, 8)

        # GAT layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'gat': BiGATLayer(d_embd, n_heads, dropout),
                'mlp': MLP(d_embd, dropout),
            })
            for _ in range(n_layers)
        ])


        self.output_proj = nn.Linear(d_embd, n_targets)

    def forward(self, x, edge_idx, edge_attr, clk_period, batch_idx):

        # embed
        cell_id = x[:, 0].long()
        cell_drive_strength = x[:, 1]
        features = x[:, 2:]

        h = (self.input_proj(features)
             + self.cell_embed(cell_id)
             + self.strength_embed(cell_drive_strength)
             )

        if self.clks:

            clk_embed = self.clk_embed(clk_period[:, None])   # (G) -> (G C); G=number of graphs in batch
            clk_node = clk_embed[batch_idx] # (G C) -> (N C); batch_idx maps node to graph
            h = h + clk_node


        # GAT
        for layer in self.layers:
            h = layer['gat'](h, edge_idx, edge_attr)  # GAT + residual
            h = layer['mlp'](h)  # MLP + residual

        # out projection
        h = self.output_proj(h)

        return h


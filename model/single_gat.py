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




'gat': GATLayer(d_embd, n_heads, dropout)
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from GAT_model import GAT_model
import argparse
import os
import json

#-----------hyperparameters------------

batch_size = 2    # chose small batch size as graphs when using large designs as data
d_embd     = 64
n_heads    = 6
n_layers   = 6
dropout    = 0.1

lr         = 1e-3
n_epochs   = 100
eval_iter  = 4
n_eval_batches = 4
val_split  = 0.8

#--------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--pyg_datasets_dir", required=True, help="Path to dataset directory")
parser.add_argument('--cell_to_idx', required=True)
parser.add_argument('--different_clk_periods', action="store_true")
args = parser.parse_args()

dataset = []
for f in os.listdir(args.pyg_datasets_dir):
    if f.endswith(".pt"):
        dataset.append(torch.load(os.path.join(args.pyg_datasets_dir, f),  weights_only=False))

with open(args.cell_to_idx) as f:
    cell_to_idx = json.load(f)


n_cells = len(cell_to_idx)
n_features = dataset[0].x.shape[-1]
n_targets = dataset[0].y.shape[-1]


train_size = int(val_split * len(dataset))
val_size   = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GAT_model(
    n_cells=n_cells,
    n_features=n_features,
    n_targets=n_targets,
    clks=args.different_clk_periods,
    d_embd=d_embd,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}

    for split, loader in [('train', train_loader), ('val', val_loader)]:

        losses = []

        for it, batch in enumerate(loader):
            if it == n_eval_batches:
                break

            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.clk_period, batch.batch)

            mask = ~torch.isinf(batch.y).any(dim=-1)
            losses.append(criterion(pred[mask], batch.y[mask]).item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out


for epoch in range(n_epochs):
    model.train()

    for it, batch in enumerate(train_loader):
        batch = batch.to(device)

        pred  = model(batch.x, batch.edge_index, batch.edge_attr, batch.clk_period, batch.batch)
        mask = ~torch.isinf(batch.y).any(dim=-1)

        loss = criterion(pred[mask], batch.y[mask])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % eval_iter == 0:
            losses = estimate_loss()
            print(f"epoch {epoch:03d} step {it:03d}: train {losses['train']:.4f} | val {losses['val']:.4f}")

    torch.cuda.empty_cache()

torch.save({
    'model_state': model.state_dict(),
    'model_config': {
        'n_cells':    n_cells,
        'n_features': n_features,
        'n_targets':  n_targets,
        'd_embd':     d_embd,
        'n_heads':    n_heads,
        'n_layers':   n_layers,
        'dropout':    dropout,
        'clks':       args.different_clk_periods
    }
}, 'checkpoints/model.pt')

# N ... total number nodes
# E ... total number edges
# batch.x (N, C)
# batch.edge_index (2, E)
# batch.batch (N) maps node to graph


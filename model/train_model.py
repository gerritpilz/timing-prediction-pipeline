import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from GAT_model import GAT_model
import argparse
import os
import json

#-----------hyperparameters------------

batch_size = 2    # chose small batch size as graphs when using large designs as data
d_embd = 64
n_heads = 4
n_layers = 4
dropout = 0.1
crit_weight=5.0

lr = 1e-3
n_epochs = 300
eval_iter = 4
n_eval_batches = 4
val_split = 0.8

#--------------------------------------

def calc_train_loss(pred, y, mask, crit_weight=5.0):
    pred_valid = pred[mask]
    y_valid = y[mask]

    slack_loss = F.mse_loss(pred_valid[:, 0], y_valid[:, 0])
    slew_loss  = F.mse_loss(pred_valid[:, 1:3], y_valid[:, 1:3])
    crit_loss  = F.mse_loss(pred_valid[:, 3], y_valid[:, 3])

    loss = slack_loss + slew_loss + crit_weight * crit_loss

    return loss

def calc_crit_mae(pred, y, mask):

    pred_valid = pred[mask]
    y_valid = y[mask]

    # MAE losses
    slack_mae = F.l1_loss(pred_valid[:, 0], y_valid[:, 0])
    slew_mae = F.l1_loss(pred_valid[:, 1:3], y_valid[:, 1:3])
    crit_mae = F.l1_loss(pred_valid[:, 3], y_valid[:, 3])

    return crit_mae, slack_mae, slew_mae


@torch.no_grad()
def estimate_loss():

    model.eval()
    out = {}

    for split, loader in [('train', train_loader), ('val', val_loader)]:

        losses = []
        crit_slack_mae_list = []
        slack_mae_list = []
        slew_mae_list = []

        for it, batch in enumerate(loader):
            if it == n_eval_batches:
                break

            batch = batch.to(device)

            pred = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.clk_period,
                batch.batch
            )

            mask = ~torch.isinf(batch.y).any(dim=-1)

            # loss
            loss = calc_train_loss(pred, batch.y, mask)
            losses.append(loss.item())

            # MAE losses
            crit_slack_mae, slack_mae, slew_mae = calc_crit_mae(pred, batch.y, mask)

            crit_slack_mae_list.append(crit_slack_mae.item())
            slack_mae_list.append(slack_mae.item())
            slew_mae_list.append(slew_mae.item())

        out[split] = {
            'loss': sum(losses) / len(losses),
            'crit_mae': sum(crit_slack_mae_list) / len(crit_slack_mae_list),
            'slack_mae': sum(slack_mae_list) / len(slack_mae_list),
            'slew_mae': sum(slew_mae_list) / len(slew_mae_list),
        }

    model.train()
    return out
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyg_datasets_dir", required=True, help="Path to dataset directory")
    parser.add_argument('--cell_to_idx', required=True)
    parser.add_argument('--different_clk_periods', action="store_true")
    args = parser.parse_args()

    dataset = []
    for f in os.listdir(args.pyg_datasets_dir):
        if f.endswith(".pt"):
            data = torch.load(os.path.join(args.pyg_datasets_dir, f), weights_only=False)
            dataset.append(data)

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



    for epoch in range(n_epochs):
        model.train()

        for it, batch in enumerate(train_loader):
            batch = batch.to(device)

            pred  = model(batch.x, batch.edge_index, batch.edge_attr, batch.clk_period, batch.batch)
            mask = ~torch.isinf(batch.y).any(dim=-1)

            loss = calc_train_loss(pred, batch.y, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if it % eval_iter == 0:
                losses = estimate_loss()

                print(f"train loss: {losses['train']['loss']:.4f} | val loss: {losses['val']['loss']:.4f}")
                print(f"val crit slack MAE: {losses['val']['crit_mae']}")
                print(f"val slack MAE: {losses['val']['slack_mae']}")
                print(f"val slew MAE: {losses['val']['slew_mae']}")
                print("")

        torch.cuda.empty_cache()

    os.makedirs('/content/drive/MyDrive/checkpoints', exist_ok=True)

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
    }, '/content/drive/MyDrive/checkpoints/model.pt')

    # N ... total number nodes
    # E ... total number edges
    # batch.x (N, C)
    # batch.edge_index (2, E)
    # batch.batch (N) maps node to graph


import torch
import argparse
import pandas as pd
from torch_geometric.loader import DataLoader
from model import GAT_model

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',  required=True, help="model.pt")
parser.add_argument('--dataset',     required=True, help=".pt PyG graph")
parser.add_argument('--output',      required=True, help="output CSV")
args = parser.parse_args()

# load model
checkpoint = torch.load(args.checkpoint, map_location='cpu')
config = checkpoint['model_config']

model = GAT_model(**config)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# load dataset
data = torch.load(args.dataset)
loader = DataLoader([data], batch_size=1, shuffle=False)

all_preds = []
with torch.no_grad():
    pred = model(
        data.x,
        data.edge_index,
        data.edge_attr,
        data.clk_period.unsqueeze(0),  # (1 1)
        torch.zeros(data.x.shape[0], dtype=torch.long)  # batch_idx: all nodes are from graph 0
    )
    pred[:, 0] = pred[:, 0] * data.clk_period

preds = pred.cpu().numpy()

df = pd.DataFrame(preds, columns=['slack_min', 'slew_r', 'slew_f'])

df.index = data.pin_ids
df.index.name = 'pin_id'

df.to_csv(args.output)
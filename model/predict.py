import os
import argparse
import torch
import pandas as pd
from GAT_model import GAT_model

parser = argparse.ArgumentParser()

parser.add_argument('--design_name', required=True)
parser.add_argument('--checkpoint', required=True,
                    help='Path to model checkpoint (.pt)')
parser.add_argument('--pyg_graph', required=True,
                    help='Path to PyTorch Geometric graph (.pt)')

args = parser.parse_args()

# load model
checkpoint = torch.load(args.checkpoint, map_location='cpu')
config = checkpoint['model_config']

model = GAT_model(**config)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# load graph
data = torch.load(args.pyg_graph)

# inference
with torch.no_grad():
    pred = model(
        data.x,
        data.edge_index,
        data.edge_attr,
        data.clk_period.unsqueeze(0),
        torch.zeros(data.x.shape[0], dtype=torch.long)
    )

    # convert normalized slack back to absolute slack
    pred[:, 0] = pred[:, 0] * data.clk_period

preds = pred.cpu().numpy()

df = pd.DataFrame(
    preds,
    columns=[
        'slack_min',
        'criticality',
        'slew_r',
        'slew_f'
    ]
)

df.index = data.pin_ids
df.index.name = 'pin_id'

# save results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

output_csv = os.path.join(results_dir, f"{args.design_name}_predictions.csv")

df.to_csv(output_csv)

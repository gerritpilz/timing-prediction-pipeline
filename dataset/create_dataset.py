import re
import json
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
import torch
import argparse
import os
import logging
import math
import numpy as np

def get_clk_period(path):
    with open(path) as f:
        for line in f:
            match = re.search(r"-period\s+([\d.]+)", line)
            if match:
                return float(match.group(1))

def normalize(s):
    return s.replace("\\", "").strip()   # remove spaces in front/end of line, remove '\'


if __name__ == "__main__":

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_dir", required=True, help="Path to build directory")
    parser.add_argument('--pin_features_dir', required=True, help="Path to pin features directory")
    parser.add_argument('--cell_pin_direction', required=True)
    parser.add_argument('--cell_pin_cap', required=True)
    parser.add_argument('--cell_drive_strength', required=True)
    parser.add_argument('--cell_to_idx', required=True)
    args = parser.parse_args()

    with open(args.cell_pin_direction) as f:
        cell_pin_direction = json.load(f)

    with open(args.cell_pin_cap) as f:
        cell_pin_cap = json.load(f)

    with open(args.cell_to_idx) as f:
        cell_to_idx = json.load(f)

    with open(args.cell_drive_strength) as f:
        cell_drive_strength = json.load(f)

    os.makedirs("pyg_datasets", exist_ok=True)

    for design_clk in os.listdir(args.build_dir):

        design = design_clk.split("_")[0]   # e.g. gcd_10 -> gcd

        pin_features_path = os.path.join(args.pin_features_dir, f"pin_features_{design_clk}.csv")
        pin_df = pd.read_csv(pin_features_path)
        pin_df = pin_df.set_index('pin_id')


        # init for graph parsing
        G = nx.DiGraph()

        net_driver = {}   # net -> driver pin
        net_loads = {}    # net -> [load pins]

        cell_inputs = {}  # cell -> [input pins]
        cell_outputs = {} # cell -> [output pins]

        # parse verilog
        verilog_path = os.path.join(args.build_dir, f"{design_clk}/job0/write.views/0/outputs/{design}.vg")

        if not os.path.exists(verilog_path):
            print(f"Skipping {design}, missing verilog")
            continue

        with open(verilog_path) as f:

            current_block = ""

            for line in f:
                line = line.strip()

                # ignore attributes
                if line.startswith("(*"):
                    continue

                # append line to current block
                current_block += " " + line

                # block ends
                if line.endswith(");"):

                    # parse gate and instance
                    match = re.search(r'\b(\w+)\s+(\\?\S+)\s*\(', current_block)   # 'cell_type instance ('
                    if not match:
                        current_block = ""
                        continue

                    cell_type = match.group(1)
                    cell_instance = normalize(match.group(2))

                    # parse pins
                    pins = re.findall(r'\.(\w+)\s*\(\s*([^)]+)\s*\)', current_block)  # '.pin( net )'

                    # append relevant pins to driver/loads dict
                    for pin, net in pins:
                        if cell_type not in cell_pin_direction:
                            continue

                        pin_id = f"{cell_instance}/{pin}"

                        if pin_id not in pin_df.index:
                            continue

                        net = normalize(net)

                        # add node to graph
                        G.add_node(pin_id)

                        # add props
                        G.nodes[pin_id]['pin'] = pin
                        G.nodes[pin_id]['cell_id'] = cell_to_idx[cell_type]
                        G.nodes[pin_id]['cell_strength'] = float(cell_drive_strength[cell_type])
                        G.nodes[pin_id]['pin_cap'] = cell_pin_cap[cell_type][pin]
                        direction = cell_pin_direction[cell_type][pin]

                        # Output
                        if direction == 'output':

                            # add prop
                            G.nodes[pin_id]['direction'] = 1

                            # driver pin of net
                            net_driver[net] = pin_id

                            # output pin cell
                            if cell_instance not in cell_outputs:
                                cell_outputs[cell_instance] = []
                            cell_outputs[cell_instance].append(pin_id)


                        # Input
                        else:
                            # add prop
                            G.nodes[pin_id]['direction'] = 0

                            # load pin of net
                            if net not in net_loads:
                                net_loads[net] = []
                            net_loads[net].append(pin_id)

                            # input pin of cell
                            if cell_instance not in cell_inputs:
                                cell_inputs[cell_instance] = []
                            cell_inputs[cell_instance].append(pin_id)


                    # reset for next block
                    current_block = ""


        # ------make graph----------

        # input_pin → output_pin (connected by cell)
        for cell_instance in cell_inputs:
            inputs  = cell_inputs[cell_instance]
            outputs = cell_outputs.get(cell_instance, [])

            for inp in inputs:
                for out in outputs:
                    G.add_edge(inp, out, edge_type=1)

        # driver → load (connected by net)
        for net in net_driver:
            driver = net_driver[net]
            loads  = net_loads.get(net, [])

            for load in loads:
                G.add_edge(driver, load, edge_type=0)



        # ---------add features-----------

        sdc_path = os.path.join(args.build_dir, f"{design_clk}/job0/write.views/0/outputs/{design}.sdc")

        if not os.path.exists(sdc_path):
            print(f"Skipping {design}, missing sdc")
            continue

        clk_period = get_clk_period(sdc_path)


        G_dag = G.copy()

        # treat register outputs as endpoints to break cycles
        for node, data in G_dag.nodes(data=True):
            pin = data.get('pin')
            if pin in {'Q', 'QN', 'QB', 'QN0'}:
                for succ in list(G_dag.successors(node)):
                    G_dag.remove_edge(node, succ)

        input_depth  = {}
        output_depth = {}

        for node in nx.topological_sort(G_dag):
            preds = list(G_dag.predecessors(node))
            input_depth[node] = 0 if not preds else 1 + max(input_depth[p] for p in preds)

        for node in reversed(list(nx.topological_sort(G_dag))):
            succs = list(G_dag.successors(node))
            output_depth[node] = 0 if not succs else 1 + max(output_depth[s] for s in succs)

        for node in G.nodes():
            G.nodes[node]['input_depth']  = input_depth[node]
            G.nodes[node]['output_depth'] = output_depth[node]

        # pin features
        numeric_props_pin = [
            'is_clk',
            'fanout',
            'slack_min',
            'slew_r',
            'slew_f',
        ]

        # to numeric
        for prop in numeric_props_pin:
            pin_df[prop] = pd.to_numeric(pin_df[prop], errors='coerce')

        # add features to nodes
        for node, row in pin_df.iterrows():
            if node in G:

                for prop in ['is_clk', 'fanout', 'slew_r', 'slew_f']:
                    G.nodes[node][prop] = row[prop]

                G.nodes[node]['slack_min_norm'] = row['slack_min'] / clk_period


        # criticality label
        alpha = 8.0
        for n, attr in G.nodes(data=True):
            slack = attr['slack_min_norm']
            criticality = np.exp(-alpha * slack)
            G.nodes[n]['criticality'] = criticality


        #-------Networkx to PyG--------------
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}   # node_name -> idx

        x = []
        y = []
        edge_index = []
        edge_attr  = []

        for n in nodes:
            data = G.nodes[n]

            # features
            x.append([
                # get extra embedding later, first two attributes
                data.get('cell_id'),
                data.get('cell_strength'),

                # pin features
                data.get('direction'),
                math.log1p(data.get('fanout')),
                data.get('input_depth'),
                data.get('output_depth'),
                data.get('is_clk'),
                data.get('pin_cap')
            ])

            # target
            y.append([
                data.get('slack_min_norm'),
                data.get('slew_r'),
                data.get('slew_f'),
                data.get('criticality')
            ])

        # edges
        for u, v, d in G.edges(data=True):
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_attr.append([d.get('edge_type', 0)])


        # PyG
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        edge_index = torch.tensor(edge_index).t().contiguous() # edge list: (2 n_edges)
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

        dataset = Data(
            x=x,
            clk_period=torch.tensor([clk_period], dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
        dataset.pin_ids = nodes

        torch.save(dataset, f"pyg_datasets/dataset_{design_clk}.pt")


        #------------logging-----------------------------

        logging.info(f"=== Dataset Check {design_clk} ===")
        logging.info(f"{dataset}")

        # shapes
        logging.info(f"Nodes:      {dataset.x.shape[0]}")
        logging.info(f"Features:   {dataset.x.shape[1]}")
        logging.info(f"Targets:    {dataset.y.shape[1]}")
        logging.info(f"Edges:      {dataset.edge_index.shape[1]}")

        # NaN check
        logging.info(f"x NaN:      {torch.isnan(dataset.x).any().item()}")
        logging.info(f"y NaN:      {torch.isnan(dataset.y).any().item()}")

        # edge index
        logging.info(f"edge_index max: {dataset.edge_index.max().item()} (should be < {dataset.x.shape[0]})")
        logging.info(f"edge_index min: {dataset.edge_index.min().item()} (should be >= 0)")

        # features
        logging.info(f"cell_id              min/max: {dataset.x[:, 0].min():.0f} / {dataset.x[:, 0].max():.0f}")
        logging.info(f"cell_drive_strength  min/max: {dataset.x[:, 1].min():.0f} / {dataset.x[:, 0].max():.0f}")
        logging.info(f"direction            unique:  {dataset.x[:, 2].unique().tolist()}")
        logging.info(f"log1p(fanout)        min/max: {dataset.x[:, 3].min():.0f} / {dataset.x[:, 2].max():.0f}")
        logging.info(f"input_depth          max:     {dataset.x[:, 4].max():.0f}")
        logging.info(f"output_depth         max:     {dataset.x[:, 5].max():.0f}")
        logging.info(f"is_clk               unique:  {dataset.x[:, 6].unique().tolist()}")
        logging.info(f"pin_cap              min/max: {dataset.x[:, 7].min():.4f} / {dataset.x[:, 6].max():.4f}")

        # targets
        logging.info(f"slack_min    min/max: {dataset.y[:, 0].min():.4f} / {dataset.y[:, 0].max():.4f}")
        logging.info(f"criticality  min/max: {dataset.y[:, 1].min():.4f} / {dataset.y[:, 1].max():.4f}")
        logging.info(f"slew_r       min/max: {dataset.y[:, 2].min():.4f} / {dataset.y[:, 2].max():.4f}")
        logging.info(f"slew_f       min/max: {dataset.y[:, 3].min():.4f} / {dataset.y[:, 3].max():.4f}")

        # edge types
        inter = (dataset.edge_attr[:, 0] == 0).sum().item()
        intra = (dataset.edge_attr[:, 0] == 1).sum().item()
        logging.info(f"inter-cell edges: {inter}")
        logging.info(f"intra-cell edges: {intra}")

        logging.info("=" * 50)


import re
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lib', required=True)
args = parser.parse_args()

cell_pin_direction = {}
cell_pin_cap       = {}
current_cell       = None
current_pin        = None

with open(args.lib) as f:
    for line in f:
        line = line.strip()

        # cell
        cell_match = re.match(r'cell\s*\(\s*"([^"]+)"\s*\)', line)
        if cell_match:
            current_cell = cell_match.group(1)
            current_pin  = None
            cell_pin_direction[current_cell] = {}
            cell_pin_cap[current_cell]       = {}
            continue

        # pin
        pin_match = re.match(r'pin\s*\(\s*"([^"]+)"\s*\)', line)
        if pin_match:
            current_pin = pin_match.group(1)
            continue

        # properties
        if current_cell and current_pin:
            dir_match = re.search(
                r'direction\s*:\s*"?(input|output|inout)"?',
                line
            )
            if dir_match:
                cell_pin_direction[current_cell][current_pin] = dir_match.group(1)

            cap_match = re.search(
                r'capacitance\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                line
            )
            if cap_match:
                cell_pin_cap[current_cell][current_pin] = float(cap_match.group(1))


# cell_pin_direction
with open("cell_pin_direction.json", "w") as f:
    json.dump(cell_pin_direction, f, indent=2)

# cell_pin_cap
with open("cell_pin_cap.json", "w") as f:
    json.dump(cell_pin_cap, f, indent=2)

# cell_to_idx
cell_types  = sorted(cell_pin_direction.keys())
cell_to_idx = {cell: i for i, cell in enumerate(cell_types)}

with open("cell_to_idx.json", "w") as f:
    json.dump(cell_to_idx, f, indent=2)



'''
#readme
## Usage
python build_graph.py --lib mylib.lib --vg gcd.vg

## Required Files
- `mylib.lib` - Liberty library file
- `gcd.vg`    - Verilog netlist
- `pin_features.csv` - Pin timing features
'''
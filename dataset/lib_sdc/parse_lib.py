import re
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--lib', required=True)
args = parser.parse_args()

cell_pin_direction = {}
cell_pin_cap       = {}
cell_drive_strength = {}
current_cell       = None
current_pin        = None

with open(args.lib) as f:
    for line in f:
        line = line.strip()

        # cell
        cell_match = re.match(r'cell\s*\(\s*"([^"]+)"\s*\)', line)
        if cell_match:

            # update
            current_cell = cell_match.group(1)
            current_pin  = None
            cell_pin_direction[current_cell] = {}
            cell_pin_cap[current_cell]       = {}

            # drive strength
            m = re.search(r'_(\d+)$', current_cell)
            cell_drive_strength[current_cell] = float(m.group(1)) if m else 1.0

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


# output directory
output_dir = "Dataset/lib_sdc/cell_dicts"
os.makedirs(output_dir, exist_ok=True)

# cell_pin_direction
with open(os.path.join(output_dir, "cell_pin_direction.json"), "w") as f:
    json.dump(cell_pin_direction, f, indent=2)

# cell_pin_cap
with open(os.path.join(output_dir, "cell_pin_cap.json"), "w") as f:
    json.dump(cell_pin_cap, f, indent=2)

# cell_drive_strength
with open(os.path.join(output_dir, "cell_drive_strength.json"), "w") as f:
    json.dump(cell_drive_strength, f, indent=2)

# cell_to_idx
cell_types = sorted(cell_pin_direction.keys())
cell_to_idx = {cell: i for i, cell in enumerate(cell_types)}

with open(os.path.join(output_dir, "cell_to_idx.json"), "w") as f:
    json.dump(cell_to_idx, f, indent=2)



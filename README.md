## Overview

ChipGAT is an end-to-end machine learning pipeline for pre-routing timing analysis of digital circuits using Graph Neural Networks.

The project transforms RTL designs into graph-based representations and learns to predict per-pin timing behavior (slack, slew, and criticality) directly from netlist structure, enabling early-stage timing risk estimation before placement and routing.

## Pipeline Summary

### 1. Design Compilation (SiliconCompiler)

Verilog designs with arbitrary SDC constraints are parsed and synthesized using SiliconCompiler, producing an OpenDB (ODB) representation of the circuit for downstream processing. 
Multiple clock speeds and operating points can be sampled to improve robustness across different timing scenarios.

### 2. Liberty Parsing

The Liberty file is parsed to extract pin capacitance and cell drive strength. In addition, a cell to index mapping is created.

### 3. Feature Extraction (OpenROAD + Tcl)

Using OpenROAD scripts, per-pin features are extracted from the ODB database:

- pin direction
- min slack 
- max rise/fall slew 
- fanout 
- clock pin indicators  
- cell type 

### 4. Dataset Construction (Graph Generation)

A dataset builder merges all extracted information into a graph representation of the design, where:

- nodes: pins  
- edges: intra-cell and inter-cell connections from the netlist  

- node features:
  - cell ID
  - log2(cell drive strength)
  - pin direction  
  - log1p(fanout)  
  - input/output depth  
  - clock indicator  
  - pin capacitance  

- targets:
  - min slack / clock period 
  - slack-derived criticality; exp(-8 * (min slack / clk period)) 
  - max rise/fall slew
    
- edge attribute:
  - intra-cell / inter-cell 
  
- clock period

The resulting graphs are converted into PyTorch Geometric (.pt) format for training.

### 5. Model Training (BiGAT)

A bi-directional graph attention network (BiGAT) is trained on the constructed graphs to learn pre-routing timing relationships in digital circuits. 

Per-pin numerical features are embedded using a shared linear projection, while categorical cell-level attributes (cell type and drive strength) and the clock period are encoded separately and added as learned embeddings.

The model performs node-level prediction of the minimal slack, the maximum rise/fall slew and a slack-derived criticality score. 
The criticality is computed as exp(-8 * (min slack / clk period)), which maps timing slack to a risk metric that increases as the available timing margin decreases. 
It serves as the primary target for identifying timing-critical regions of the design and prioritizing pins with the highest likelihood of timing violations.

### 6. Pedeicion 

## How to Use
### 1. Setup

### 2. Design Compilation

For each design that should be part of the dataset, put all verilog files in a directory. A generic sdc file is already included at Dataset/lib_sdc . Note that in the first line, 

python3 Dataset/chip_run.py \
  --rtl Dataset/Designs/slowfil/rtl/*.v \
  --sdc Dataset/lib_sdc/clk10.sdc \
  --clk_period 10 \
  --design slowfil \
  --top_module slowfil




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
  - slack-derived criticality; exp(-8 * min slack) 
  - rise/fall max slew
  - 
- edge attribute:
  - intra-cell / inter-cell 
  
- clock period

The resulting graphs are converted into PyTorch Geometric (.pt) format for training.

### 5. Model Training (BiGAT / GAT)

A bi-directional graph attention network (BiGAT) is trained on the constructed graphs to learn pre-routing timing relationships in digital circuits. 

The model performs node-level prediction of:
- timing slack  
- criticality scores  
- signal slew characteristics  

## How to Use


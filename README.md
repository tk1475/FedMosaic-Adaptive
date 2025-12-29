# CS6304/EE5102 Advanced Topics in Machine Learning (Dr. Muhammad Tahir, LUMS)

# FedMosaic-Adaptive 

This repository accompanies the ATML project report and currently implements the Layerwise Routing and Data-Free Geometric Alignment components for federated alignment of heterogeneous LLM adapters. RepSim remains future work; existing experiments revolve around routing controllers on top of block-wise PQ-LoRA adapters plus geometry-aware alignment.

---

## Conceptual Overview

- **Block-wise PQ-LoRA adapters:** Each client trains lightweight adapters inserted into the last Transformer blocks, mixing local and globally aggregated low-rank projections via a learnable gate $\\beta$:
  $$
  h_O = W_p h_I + (1 - \\beta) h_L + \\beta h_G
  $$
- **Layerwise routing controller (implemented):** [`core_blockwise.FederatedClientBlockWise`](core_blockwise.py) activates/deactivates adapter blocks per round using gradient-based salience heuristics, emulating the “Layerwise Routing” routine from the paper.
- **Data-free geometric alignment (implemented):** [`alignment.py`](alignment.py) synthesizes anchor features and performs orthogonal alignment of adapter subspaces without sharing raw data.
- **SimGrad baseline:** Clients share sanitized gradient signatures derived from [`core.FederatedClient`](core.py); aggregation uses cosine relevance on EMA-smoothed gradients.
- **Server personalization:** [`core_blockwise.ServerBlockWise`](core_blockwise.py) stores per-client adapter dictionaries on disk and rebuilds personalized PQ weights each round.
- **Data diversity:** [`data_manager.DataManager`](data_manager.py) partitions MNLI, CommonsenseQA, SQuAD, and GSM8K into 10 cognitively distinct clients to mimic the paper’s “Scaled-Up DRAKE” benchmark.

---

## Running Experiments

1. **Data preparation:** `DataManager().setup_real_world_benchmark()` builds the 10-client benchmark with cached slices.
2. **Baseline (SimGrad):** Execute the “STARTING BASELINE: GRADIENT MODE (SimGrad)” cell in [simulation.ipynb](simulation.ipynb). This instantiates [`core_blockwise.FederatedClientBlockWise`](core_blockwise.py) with `mode="grad"` to collect gradient signatures.
3. **Layerwise routing + geometric alignment:** Run the “LAYERWISE ROUTING + DATA-FREE ALIGNMENT” cell. Clients invoke `_phase_2_layerwise_route` and the alignment utilities to reweight adapters without exchanging raw representations.
4. **50-round study:** The “PAPER” section of the notebook defines `FederatedClientPaper` with learning-rate decay (see [`core_blockwise.FederatedClientPaper.train_and_rela`](simulation.ipynb)) to reproduce the report’s 50-round curves for the implemented components.
5. **Visualization:** The `visualize_impact` helper plots loss trajectories and alignment metrics comparing routing-enabled vs. baseline runs.

---

## Key Implementation Notes

- **Adapter injection:** The final four Transformer blocks are wrapped with PQ-LoRA hooks (`self_attn.q_proj` / `v_proj`) as shown in [`core_blockwise.FederatedClientBlockWise._phase_1_train`](core_blockwise.py).
- **Layerwise routing masks:** Routing decisions are cached per block and logged each round, ensuring only salient adapters receive bandwidth.
- **Geometric alignment:** `AlignmentManager.align_client()` generates synthetic anchors, projects adapters into a shared basis, and enforces orthogonality to keep data-free guarantees.
- **Memory discipline:** All models load in 4-bit NF4 via `BitsAndBytesConfig`, adapters remain in `torch.float32`, and hooks are removed after each round to prevent leaks.
- **Server offloading:** Personalized adapter dictionaries are checkpointed under `server_paper_tmp` to emulate disk-backed aggregation.

---

## Results Snapshot

- Layerwise routing consistently lowers communication and improves convergence on relation-heavy clients compared with the vanilla SimGrad baseline.
- Data-free geometric alignment stabilizes personalization under heterogeneous adapters, yielding tighter loss bands across MNLI and CommonsenseQA groups.

---

## Citing

If you extend this work, please cite the accompanying ATML project report (`ATML_Project_Report.pdf`) and reference this repository as **FedMosaic-Adaptive: Layerwise Routing + Data-Free Geometric Alignment**.
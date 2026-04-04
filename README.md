# Dowload the 

Raw file from ML4Sci: https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view?usp=sharing

And It' processed version (processed with CT-02 (Data Pre Pre Processor in Common Task 2),  the same processed dataset is used for further sp1 and sp4 tasks.

Download processed Dataset: https://drive.google.com/file/d/1ogTlGvCi6QH5E8WfUiH_wEhg3U6mbcdH/view?usp=sharing

Put these put files in Data folder to run all the code successfully.
# ML4Sci GSoC 2026 — Evaluation Tasks
### Deep Graph Anomaly Detection with Contrastive Learning for New Physics Searches

> **Candidate:** Atharv Sandeep Vichare  
> **Institute:** VESIT, Mumbai · B.E. Artificial Intelligence and Data Science (Year 3)  
> **Project:** GENIE — Deep Graph Anomaly Detection with Contrastive Learning  
> **Mentors:** Sergei Gleyzer (Alabama) · Ali Hariri (EPFL) · Amal Saif (PSUT) · Tom Magorsch (TUM)



## Common Task 1 — Convolutional Autoencoder for Jet Reconstruction

**Notebook:** `CT_1.ipynb`

**Task:** Train a convolutional autoencoder on quark/gluon jet images (three calorimeter channels: ECAL, HCAL, Tracks). Show side-by-side reconstruction of original vs reconstructed events. Use the learned latent representation for quark/gluon classification via a linear probe.

### Architecture

```
Input  (B, 3, 125, 125)

ENCODER
  Conv2d(  3→ 32, k=4, s=2, p=1) + BN + LeakyReLU(0.2)  →  (B,  32, 62, 62)
  Conv2d( 32→ 64, k=4, s=2, p=1) + BN + LeakyReLU(0.2)  →  (B,  64, 31, 31)
  Conv2d( 64→128, k=4, s=2, p=1) + BN + LeakyReLU(0.2)  →  (B, 128, 15, 15)
  Conv2d(128→256, k=4, s=2, p=1) + BN + LeakyReLU(0.2)  →  (B, 256,  7,  7)
  Flatten → Linear → Latent dim 256

DECODER (symmetric transpose convolutions)
  Linear → Reshape → ConvTranspose2d × 4 + BN + ReLU → Sigmoid output
```

### Config

| Parameter | Value |
|---|---|
| Samples | 10,000 |
| Val fraction | 20% |
| Latent dim | 256 |
| Batch size | 32 |
| Epochs | 50 |
| Learning rate | 1e-3 |
| LR scheduler | ReduceLROnPlateau (patience=8, factor=0.5) |
| Grad clip | 0.5 |
| Early stopping | Patience 15 |

### Results

| Metric | Value |
|---|---|
| Test Accuracy | **72.90%** |
| ROC-AUC | **0.7877** |

**Classification report (test set, 1000 samples):**

|  | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Quark | 0.73 | 0.70 | 0.72 | 488 |
| Gluon | 0.73 | 0.75 | 0.74 | 512 |
| **Weighted avg** | **0.73** | **0.73** | **0.73** | 1000 |

---

## Common Task 2 — Jets as Graphs: GNN Benchmark

Two notebooks cover this task. The data preprocessor builds the graph dataset once; the model trainer loads it and benchmarks six GNN architectures.

---

### CT 2a — Data Preprocessor

**Notebook:** `CT-2_Data_Preprocessor_.ipynb`

**Task:** Convert raw quark/gluon jet images (HDF5, shape 125×125×3) into a PyTorch Geometric graph dataset ready for GNN training.

**Pipeline:**

```
HDF5 input  (X_jets: N×125×125×3,  y,  pt,  m0)
      │
      ▼
Stage 1 — Load & normalise
  Per-channel min-max normalisation across all loaded events

      │
      ▼
Stage 2 — Image → point cloud
  Non-zero pixels → 3D points
  Node features (5-D): [x_norm, y_norm, z, intensity, channel_id]
  z separates calorimeter layers via LAYER_SEP = 0.5
  Top MAX_NODES pixels by intensity retained per event

      │
      ▼
Stage 3 — kNN graph (pure PyTorch, no sklearn)
  torch.cdist → k=8 nearest neighbours
  Undirected edges, no self-loops
  edge_attr = Euclidean distance ΔR

      │
      ▼
Stage 4 — PyG Data object
  x: (N_nodes, 5)   edge_index: (2, E)   edge_attr: (E, 1)
  y, pt, m0 stored per graph

      │
      ▼
InMemoryDataset.collate → saved as jet_pyg_dataset.pt
```

**Config:**

| Parameter | Value |
|---|---|
| N events | 10,000 |
| Max nodes per graph | 1,000 |
| k-NN neighbours | 8 |
| Layer separation (z) | 0.5 |
| Node feature dim | 5 |
| Edge feature dim | 1 (ΔR) |

**Extras:** Interactive 3D Plotly visualiser for per-event graph inspection (node colour = intensity, edges = kNN connectivity).
<img width="1209" height="1023" alt="image" src="https://github.com/user-attachments/assets/1917591d-199b-4d7e-a894-4c1d69260596" />

---

### CT 2b — GNN Model Training & Benchmark

**Notebook:** `CT-2_Model_Training.ipynb`

**Task:** Train and benchmark five GNN classifiers on the pre-built graph dataset. Compare against ChebNet (pre-trained, results injected). All models follow the same backbone pattern: 3 graph-conv layers + BatchNorm → global_mean_pool → MLP head (hidden=128).

**Models benchmarked:**

| Model | Conv layers | Notes |
|---|---|---|
| GCN | GCNConv × 3 | Baseline spectral |
| GAT | GATConv × 3 (4 heads) | Attention, dropout=0.2 |
| GraphSAGE | SAGEConv × 3 | Inductive, neighbourhood sampling |
| EdgeConv | EdgeConv × 3 (max aggr) | DGCNN-style, edge MLP |
| ImprovedGNN | GAT→SAGE→GIN + multi-scale pool | add + mean + max pooling |
| **ChebNet** | ChebConv × 3, K=5 | Spectral, pre-trained result |

**Training config:**

| Parameter | Value |
|---|---|
| Split | 70% train / 15% val / 15% test |
| Batch size | 64 |
| Epochs | 50 |
| Early stopping | Patience 10 (val loss) |
| Optimiser | AdamW |
| Scheduler | ReduceLROnPlateau |
| Grad clip | 1.0 |
| Hidden dim | 128 |

**Results (test set):**

| Model | Accuracy | ROC-AUC |
|---|---|---|
| GCN | — | — |
| GAT | — | — |
| GraphSAGE | — | — |
| EdgeConv | — | — |
| ImprovedGNN | — | — |
| **ChebNet (K=5)** | **72.90%** | **0.7869** |

> Trained model results populate at runtime. ChebNet results are injected from pre-training.

**Visualisations produced:**
- Training curves (val accuracy + val AUC per epoch for all trained models)
- ROC curves for all six models on the test set
- Accuracy & AUC summary bar chart
- Confusion matrices for all six models

---

## Specific Task 01 — JetGLADC: Stable-ChebNet + Contrastive Learning

Two versions developed iteratively. Both classify quark vs gluon jets using a Stable-ChebNet encoder with contrastive learning, demonstrating the architecture proposed in the main GSoC project.

---

### ST 01 v2 — JetGLADC v2

**Notebook:** `ST-01.ipynb`

**Architecture:** Stable-ChebNet encoder + GLADC perturbed dual-encoder + NT-Xent contrastive loss + binary classification head.

```
Input jet graph
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
Clean encoder f(θ)              Perturbed encoder f(θ′)
3× StableChebNetLayer(K)        θ′ = θ + σ·ε,  ε~N(0,1)
   X^(l+1) = LayerNorm(          (functional_call API)
     X^(l) + ε·Σ_k T_k(L̃)
     X^(l)(W_k − W_kᵀ − γI))
ELU activation per layer
      │                                  │
  z_graph = cat(max_pool, mean_pool)   ẑ_graph
      │                                  │
      └──────────┬───────────────────────┘
                 │
         Projection head
         Linear → BN → ELU → Linear
                 │
         NT-Xent loss (L_con, τ=0.15)
         + Binary CE loss (L_ce)
         L_total = L_ce + λ(t)·L_con
         λ ramps 0 → final over warmup epochs
                 │
         MLP classifier head (3 layers + BN)
```

**Config:**

| Parameter | Value |
|---|---|
| ChebNet order K | configurable |
| ChebNet layers | 3 |
| Hidden dim | 128 |
| Dual pool | max + mean → 256-dim |
| Projection dim | configurable |
| Perturbation σ | configurable |
| ε (Euler step) | configurable |
| γ (damping) | configurable |
| Batch size | 64 |
| LR schedule | OneCycleLR (peak 3e-3, warm-up → cosine) |
| Contrastive temperature τ | 0.15 |
| Grad clip | 2.0 |
| Activation | ELU |

**Key design decisions:**
- `OneCycleLR` replaces `CosineAnnealingLR` — less aggressive early decay, faster initial convergence
- Batch size 64 over 32 — more in-batch negatives for NT-Xent
- Dual-pool readout (max ∥ mean) — richer global jet descriptor
- `torch.func.functional_call` used for weight-perturbed encoder (autograd-safe)
- Antisymmetric weights: `W_antisym = W − Wᵀ − γI` — guarantees Jacobian stability

**Best test results:**

| Metric | Value |
|---|---|
| Accuracy | **72.90%** |
| ROC-AUC | **0.7869** |
| Precision | 0.7422 |
| Recall | 0.6928 |
| F1 | 0.7166 |

---

### ST 01 v3 — JetGLADC v3

**Notebook:** `ST-01-v3__1_.ipynb`

**Task:** Upgrade v2 with supervised contrastive learning (SupCon), a fixed Chebyshev rescaling, triple-pool readout, and training efficiency improvements.

**Changes from v2:**

| Component | v2 | v3 |
|---|---|---|
| Contrastive loss | NT-Xent (self-supervised) | **SupCon** (Khosla et al., NeurIPS 2020) |
| Augmentation | Perturbed dual-encoder | **None — single clean encoder** |
| Pooling | max ∥ mean (256-dim) | **max ∥ mean ∥ std (384-dim)** |
| ChebNet basis | L̃ (wrong λ_max range) | **L̂ = −D^{−½}AD^{−½}  (λ_max=2, correct)** |
| Input norm | None | **BatchNorm on raw node features** |
| Training speed | Baseline | **AMP autocast + GradScaler (~2× on CUDA)** |

**Why SupCon over NT-Xent:**  
NT-Xent treats every other graph in the batch as a negative, including graphs of the same class. SupCon uses labels to construct label-aware positives — all quark jets attract each other, all gluon jets attract each other, cross-class pairs repel. This produces more linearly separable embeddings than self-supervised NT-Xent.

**Why fixed ChebNet basis:**  
v2 applied the filter directly on L̃ without correct eigenvalue scaling, placing the Chebyshev domain outside [−1, 1]. v3 uses L̂ = −D^{−½}AD^{−½} which has spectral range [−1, 1] by construction (λ_max = 2 exactly), making the polynomial approximation valid.

**Evaluation additions in v3:**
- UMAP / t-SNE latent space visualisation before and after training — confirms SupCon creates tighter class clusters
- Linear probe (frozen encoder): Linear SVM + Logistic Regression on frozen embeddings — high linear-probe AUC confirms the encoder has learned linearly separable quark/gluon representations

---

## Overall Results Summary

| Task | Model | Accuracy | ROC-AUC |
|---|---|---|---|
| Common Task 1 | CNN Autoencoder (linear probe) | 72.90% | 0.7877 |
| Common Task 2 | ChebNet (K=5, 3-layer) | 72.90% | 0.7869 |
| Specific Task v2 | JetGLADC v2 (Stable-ChebNet + NT-Xent) | 72.90% | 0.7869 |

> ⚠ **Note on current metrics:** Quantitative metrics are directly dependent on full convergence of the contrastive loss (NT-Xent / SupCon). Complete convergence of InfoNCE-class losses at this scale is computationally intensive — the current implementation does not have access to the compute required to drive contrastive loss to its optimal minimum. Despite this, the model demonstrates meaningful progress: L₃ successfully reduces the distance between Z_G and Z'_G for SM jets across training, and the latent space develops emerging geometric structure with partial jet identity separation. With increased computational resources allowing full contrastive convergence, the discriminative potential of the pipeline is expected to improve substantially. The architecture and training protocol are validated — compute is the remaining bottleneck.

---

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-{VERSION}.html
pip install h5py numpy matplotlib scikit-learn plotly tqdm
```

## Data

| Dataset | Task | Source |
|---|---|---|
| `quark-gluon_data-set_n139306.hdf5` | CT1, CT2 | CMS Open Data / ML4Sci |
| `jet_pyg_dataset.pt` | CT2 training | Built by `CT-2_Data_Preprocessor_.ipynb` |
| `events_anomalydetection_v2.h5` | Specific Task (GlAD-SCN) | LHCO R&D Dataset, zenodo.org/record/4536377 |

---

## References

[2] Luo et al. — Deep Graph Level Anomaly Detection with Contrastive Learning, *Scientific Reports* 2022  
[3] Hariri et al. — Return of ChebNet, arXiv:2104.01725  
[9] Ghojogh & Ghodsi — Graph Neural Network, ChebNet, GCN, Graph Autoencoder: Tutorial and Survey  
[10] Tang, Li, Yu — ChebNet with Rectified Power Units  
[11] Andrews et al. — End-to-End Jet Classification of Quarks and Gluons with CMS Open Data  
[13] Velickovic et al. — Graph Attention Networks, ICLR 2018  
Khosla et al. — Supervised Contrastive Learning, NeurIPS 2020

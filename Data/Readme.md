Jet Graph Dataset Builder
Converts raw quark/gluon jet images from HDF5 into a PyTorch Geometric (PyG) graph dataset ready for GNN training.

What it does

Stage 1 — Load & Normalize
Reads up to N_EVENTS events from the HDF5 file (X_jets, y, pt, m0). Each jet image is shape (125, 125, 3) — three calorimeter channels. Per-channel min-max normalization is applied across the full loaded dataset.

Stage 2 — Image → Point Cloud
Each channel's non-zero pixels become 3D points. Node features are 5-dimensional: [x_norm, y_norm, z, intensity, channel_id], where z separates channels in 3D space via LAYER_SEP. Only the top-MAX_NODES pixels by intensity are kept per graph to control size.

Stage 3 — kNN Graph Construction
A k-nearest-neighbor graph (k=8) is built per event using torch.cdist — no sklearn dependency. The graph is undirected, and edge attributes store Euclidean distances between connected nodes.

Stage 4 — Save
All Data objects are collated via InMemoryDataset.collate() and saved as a single .pt file containing a (data_obj, slices) tuple.

Config
ParameterDefaultDescriptionDATA_PATHData/quark-gluon_data-set_n139306.hdf5Input 

HDF5SAVE_PATHData/jet_pyg_dataset.ptOutput graph datasetN_EVENTS10,000Events to process MAX_NODES1, 000Max nodes per graphK_NEIGHBORS8kNN connectivityLAYER_SEP0.5Z-spacing between channelsVISUALIZETrueShow Plotly 3D for first event

Output — each Data object contains
AttributeShapeDescriptionx(N, 5)Node featurespos(N, 3)3D spatial positionsedge_index(2, E)COO edge connectivityedge_attr(E, 1)Edge distancesy(1,)Label — 0=gluon, 1=quarkpt(1,)Jet transverse momentumm0(1,)Jet invariant mass

Loading in the next task
pythonimport torch
from torch_geometric.data import InMemoryDataset

data_obj, slices = torch.load("Data/jet_pyg_dataset.pt")
# Reconstruct individual graphs using slices as needed,
# or wrap in a dataset class for DataLoader use.
The .pt file is a drop-in input for any PyG-based GNN training pipeline (GCN, GAT, ChebNet, EdgeConv, etc.).

Dependencies
h5py, numpy, torch, torch_geometric, plotly (optional, for visualization)


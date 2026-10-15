# apholo8-EGNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)

Binding-site predictor based on VN-EGNN (Virtual Node E(n)-equivariant Graph Neural Network), inference only.

Model architecture and training code are from [ml-jku/vnegnn](https://github.com/ml-jku/vnegnn).
The weights are automatically downloaded from [Hugging face](https://huggingface.co/owkin/apholo8-egnn).

Inference of Cryptobench 1bk2, chain A, 57 residues :
- on CPU : ~77s, peak RAM 17Gb
- on GPU (Tesla T4, 15Gb of VRAM) : 74s

## Credits
Developed at [Owkin](https://www.owkin.com) by Christian Esposito (Owkin), Laura Dumont (Owkin), Rose Hoste (Owkin), Floriane Montanari (Owkin)

## Usage

### Install

```bash
# With GPU
uv sync --extra gpu

# With CPU
uv sync --extra cpu
```

### CLI

**Single structure**

```bash
uv run apholo8-egnn --structure protein.pdb
```

Results are written to `<structure stem>_predictions.csv` (override with `--output`).
Each row is one predicted binding site:

```
confidence_rank,nms_rank,confidence,x,y,z
1,1,0.823451,12.345,6.789,-3.210
2,4,0.761234,13.100,6.500,-3.000
3,2,0.654321,45.678,23.456,12.345
```

**Folder (batch mode)**

```bash
uv run apholo8-egnn --folder /path/to/structures/
```

Results are written to `<folder name>_predictions.csv`. Each row is one structure file, with
predictions stored in wide format (`vn1_conf`, `vn1_x`, `vn1_y`, `vn1_z`, `vn1_nms_rank`, `vn2_...`).
Virtual nodes are indexed in descending confidence order (vn1 = highest confidence).
Invalid files (empty, no atom records, no protein residues) are rejected before ESM-2 runs.
Failed and invalid structures produce an empty row and processing continues.

```
structure,vn1_conf,vn1_x,vn1_y,vn1_z,vn1_nms_rank,...
1abc.pdb,0.823451,12.345,6.789,-3.210,1,...
2xyz.cif,0.761234,45.678,23.456,12.345,1,...
```

**All flags**

| Flag | Default | Description |
|---|---|---|
| `--structure` / `-s` | required | Path to a single `.pdb`, `.cif`, or `.mmcif` file |
| `--folder` / `-f` | required | Path to a folder of structure files (batch mode) |
| `--device` / `-d` | `cuda` | Device for VNEGNN and ESM-2 3B (`cuda`, `cuda:1`, …) |
| `--output` / `-o` | auto | Output CSV path |
| `--top` / `-n` | all | Limit to top-N predictions per structure ranked by confidence |
| `--cycles` | `1` | Number of iterative prediction cycles |
| `--virtual-nodes` | `8` | Candidate binding sites per cycle |
| `--nms-radius` | `2.0` | NMS minimum separation radius in Å |

`--structure` and `--folder` are mutually exclusive; one is required.

NMS is a greedy reordering, not a filter: the highest-confidence site is always picked first,
then each next pick must be ≥ `--nms-radius` Å from all already-chosen sites.
If no candidate satisfies the constraint, the algorithm falls back to the next best by confidence,
so every prediction appears in both orderings and the global maximum is preserved.

### Python API

```python
from apholo8_egnn.predictor import VNEGNNPredictor
from apholo8_egnn.utils.nms import nms_order
import numpy as np

predictor = VNEGNNPredictor(device="cuda")  # omit device to auto-detect

coords, confidence = predictor.predict(
    structure_path="protein.pdb",  # or .cif / .mmcif
    num_cycles=1,
    num_virtual_nodes=8,
)
# coords:     np.ndarray [K, 3]  binding-site positions (Å), sorted by descending confidence
# confidence: np.ndarray [K]     confidence scores, descending

# NMS reordering ("with NMS" order): returns a permutation of [0..K-1]
nms_perm = nms_order(coords, confidence, radius=2.0)

print("Without NMS (by confidence):")
for rank, (coord, conf) in enumerate(zip(coords, confidence), 1):
    print(f"  {rank}  conf={conf:.4f}  xyz=({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")

print("With NMS:")
for nms_rank, idx in enumerate(nms_perm, 1):
    coord, conf = coords[idx], confidence[idx]
    print(f"  {nms_rank}  conf={conf:.4f}  xyz=({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
```

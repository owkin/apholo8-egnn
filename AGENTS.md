# AGENTS.md

## Project overview

apholo-egnn is an inference-only package that predicts protein binding-site
locations using a VN-EGNN model with ESM-2 3B embeddings. It reproduces the
apholo8 paper results on the CryptoBench benchmark.

## Install

```bash
# GPU
uv sync --extra gpu

# CPU only
uv sync --extra cpu
```

## Run inference

```bash
# Single structure
uv run apholo-egnn --structure protein.pdb --device cuda

# Folder of structures
uv run apholo-egnn --folder /path/to/structures/ --device cuda

# Reproduce paper results on CryptoBench
uv run python scripts/eval_cryptobench.py \
    --structures-dir /path/to/cif_files \
    --dataset-json   /path/to/dataset.json \
    --folds-json     /path/to/folds.json \
    --seed-mode paper
```

## Tests

```bash
uv run pytest tests/ -x
```

## Lint

```bash
uv run ruff check src/ scripts/
uv run mypy src/
```

## Code structure

- `src/apholo_egnn/predictor.py` — `VNEGNNPredictor`: loads checkpoint, runs
  ESM-2 3B + VNEGNN inference, returns binding-site coordinates and confidence scores
- `src/apholo_egnn/model/vnegnn.py` — VNEGNN architecture (from ml-jku/vnegnn)
- `src/apholo_egnn/data/graph.py` — builds the heterogeneous protein graph
- `src/apholo_egnn/utils/structure.py` — PDB/mmCIF parsing via BioPython
- `src/apholo_egnn/utils/nms.py` — non-maximum suppression for binding sites
- `src/apholo_egnn/cli/app.py` — CLI entry point
- `scripts/eval_cryptobench.py` — CryptoBench evaluation script

## Key conventions

- Parser dispatches by file extension: `.cif`/`.mmcif` → MMCIFParser, else PDBParser
- Chain filtering matches the training pipeline: pass `chains=['A', 'B']` to
  `predict()` to select specific chains; chains are embedded independently with ESM-2
- Confidence scores are sigmoid-transformed logits, values in (0, 1)
- Virtual-node seeding: `vn_seed=None` disables random rotations (deterministic);
  passing an integer enables them and makes results reproducible

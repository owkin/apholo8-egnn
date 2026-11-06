# AGENTS.md

## Project overview

apholo8-egnn is an inference-only package that predicts protein binding-site
locations using a VN-EGNN model with ESM-2 3B embeddings. It reproduces the
apholo8 paper results on the CryptoBench benchmark.

## Install

```bash
# Default (torch and pyg-lib from PyPI defaults / the pyg CUDA 13.0 index)
uv sync

# GPU (CUDA 13.0 wheels)
uv sync --group gpu

# CPU only
uv sync --group cpu
```

## Run inference

```bash
# Single structure
uv run apholo8-egnn --structure protein.pdb

# Folder of structures
uv run apholo8-egnn --folder /path/to/structures/ --device cuda  # default: auto-detect

# Reproduce paper results on CryptoBench
uv run python scripts/eval_cryptobench.py \
    --structures-dir /path/to/cif_files \
    --dataset-json   /path/to/dataset.json \
    --folds-json     /path/to/folds.json \
    --seed-mode paper
```

## Tests

```bash
uv sync --group gpu --group dev
uv run pytest tests/
```

## Lint

```bash
uv run ruff check src/ scripts/
uv run mypy src/
```

## Code structure

- `src/apholo8_egnn/predictor.py` — `VNEGNNPredictor`: loads checkpoint, runs
  ESM-2 3B + VNEGNN inference, returns binding-site coordinates and confidence scores
- `src/apholo8_egnn/model/vnegnn.py` — VNEGNN architecture (from ml-jku/vnegnn)
- `src/apholo8_egnn/data/graph.py` — builds the heterogeneous protein graph
- `src/apholo8_egnn/utils/structure.py` — PDB/mmCIF parsing via BioPython
- `src/apholo8_egnn/utils/nms.py` — non-maximum suppression for binding sites
- `src/apholo8_egnn/cli/app.py` — CLI entry point
- `scripts/eval_cryptobench.py` — CryptoBench evaluation script

## Key conventions

- Format detection (`detect_format`): `.cif`/`.mmcif` → mmCIF, `.pdb`/`.ent` → PDB, other extensions are detected from the content; `predict()` reads each file once (`read_structure`) and validates it before ESM-2 runs
- Chain filtering matches the training pipeline: pass `chains=['A', 'B']` to
  `predict()` to select specific chains; chains are embedded independently with ESM-2
- CLI `--device` defaults to `None`, so `VNEGNNPredictor` picks `cuda` if available, else `cpu`;
  `scripts/eval_cryptobench.py` keeps `cuda` as its default
- Dependencies: `torch` and `pyg-lib` are required `[project]` dependencies; `cpu`/`gpu`/`dev`
  are uv dependency groups (not extras), so use `--group`, not `--extra`
- Confidence scores are sigmoid-transformed logits, values in (0, 1)
- Virtual-node seeding: `vn_seed=None` disables random rotations (deterministic);
  passing an integer enables them and makes results reproducible

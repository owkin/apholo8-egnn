#!/usr/bin/env python3
"""Run apholo8-egnn inference on the CryptoBench test set.

CryptoBench dataset
-------------------
Download the dataset JSON files (dataset.json and folds.json) from the
CryptoBench repository (see the apholo8 paper for the reference).

Download apo structure CIF files from the PDB (https://www.rcsb.org) for
each PDB ID in the test fold, and place them in a single directory as
``<pdb_id>.cif``.

Output
------
One CSV row per test entry:

    entry, vn1_conf, vn1_x, vn1_y, vn1_z, vn1_nms_rank, vn2_conf, ...

Predictions are sorted by descending confidence (vn1 = highest confidence).
``vn{n}_nms_rank`` is the 1-based rank assigned by NMS (radius 2 Å).

Seeding
-------
Virtual nodes are placed on a Fibonacci sphere around the protein. Three
seeding strategies are available via ``--seed-mode``:

  none   (default) Random rotations disabled; fully deterministic output.
  fixed  Same integer seed for every entry (set with ``--vn-seed``).
  paper  Per-entry CRC32-based seed matching the apholo8 paper's evaluation
         pipeline, enabling exact reproduction of published results.

Usage
-----
    python scripts/eval_cryptobench.py \\
        --structures-dir /path/to/cif_files \\
        --dataset-json   /path/to/dataset.json \\
        --folds-json     /path/to/folds.json

    # Reproduce paper results exactly:
    python scripts/eval_cryptobench.py \\
        --structures-dir /path/to/cif_files \\
        --dataset-json   /path/to/dataset.json \\
        --folds-json     /path/to/folds.json \\
        --seed-mode paper --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import zlib
from pathlib import Path

import numpy as np

from apholo8_egnn.predictor import VNEGNNPredictor
from apholo8_egnn.utils.nms import nms_order
from apholo8_egnn.utils.structure import parse_chains

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_NUM_VN = 8


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def load_test_entries(
    folds_json: Path,
    dataset_json: Path,
) -> list[tuple[str, str, list[str]]]:
    """Return list of (entry_id, pdb_id, chains) for the test fold.

    entry_id follows the paper format, e.g. '1fd4_G-H'.
    chains is the parsed apo_chain list, e.g. ['G', 'H'].
    """
    with open(folds_json) as f:
        folds = json.load(f)
    if "test" not in folds:
        raise KeyError(f"'test' key not found in {folds_json}; available keys: {list(folds.keys())}")
    test_pdb_ids: set[str] = set(folds["test"])

    with open(dataset_json) as f:
        dataset = json.load(f)

    entries = []
    seen = set()
    for pdb_id, holo_entries in dataset.items():
        if pdb_id not in test_pdb_ids:
            continue
        chain_str = holo_entries[0]["apo_chain"]
        chains = parse_chains(chain_str)
        entry_id = f"{pdb_id}_{chain_str}"
        if entry_id in seen:
            continue
        seen.add(entry_id)
        entries.append((entry_id, pdb_id, chains))

    logger.info("Test entries resolved: %d", len(entries))
    return sorted(entries)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        required=True,
        help="Directory containing apo CIF files named <pdb_id>.cif.",
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="Path to CryptoBench dataset.json.",
    )
    parser.add_argument(
        "--folds-json",
        type=Path,
        required=True,
        help="Path to CryptoBench folds.json.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("apholo8_egnn_cryptobench_test.csv"),
        help="Output CSV path (default: apholo8_egnn_cryptobench_test.csv).",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        help="PyTorch device for VNEGNN and ESM-2 3B (default: cuda).",
    )
    parser.add_argument(
        "--seed-mode",
        choices=["none", "fixed", "paper"],
        default="none",
        help=(
            "Virtual-node Fibonacci initialisation strategy. "
            "'none': random rotations disabled, fully deterministic (default). "
            "'fixed': same --vn-seed integer for every entry. "
            "'paper': per-entry CRC32 seed reproducing the paper's evaluation pipeline."
        ),
    )
    parser.add_argument(
        "--vn-seed",
        type=int,
        default=0,
        help="Seed integer used with --seed-mode fixed (default: 0).",
    )
    args = parser.parse_args()

    entries = load_test_entries(args.folds_json, args.dataset_json)

    logger.info("Loading model ...")
    predictor = VNEGNNPredictor(device=args.device)

    header = ["entry"] + [
        f"vn{n}_{field}"
        for n in range(1, _NUM_VN + 1)
        for field in ("conf", "x", "y", "z", "nms_rank")
    ]

    n_ok = n_fail = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for entry_id, pdb_id, chains in entries:
            cif_path = args.structures_dir / f"{pdb_id}.cif"
            if not cif_path.exists():
                logger.warning("%s: CIF not found at %s, skipping", entry_id, cif_path)
                writer.writerow([entry_id] + [""] * (len(header) - 1))
                n_fail += 1
                continue

            if args.seed_mode == "paper":
                digest = zlib.crc32(entry_id.encode()) & 0xFFFFFFFF
                # 0x9E3779B9 is the Fibonacci/golden-ratio hash constant, matching the paper pipeline
                vn_seed = int((digest + 0x9E3779B9) & 0xFFFFFFFF)
            elif args.seed_mode == "fixed":
                vn_seed = args.vn_seed
            else:
                vn_seed = None

            logger.info("Processing %s (chains=%s) ...", entry_id, chains)
            try:
                coords, confidence = predictor.predict(
                    structure_path=cif_path,
                    chains=chains,
                    num_virtual_nodes=_NUM_VN,
                    vn_seed=vn_seed,
                )

                perm = nms_order(coords, confidence, radius=2.0)
                nms_rank = np.empty(len(coords), dtype=int)
                for nms_pos, conf_idx in enumerate(perm, 1):
                    nms_rank[conf_idx] = nms_pos

                row = [entry_id]
                for coord, conf, nr in zip(coords[:_NUM_VN], confidence[:_NUM_VN], nms_rank[:_NUM_VN]):
                    row += [_fmt(conf), _fmt(coord[0]), _fmt(coord[1]), _fmt(coord[2]), nr]
                writer.writerow(row)
                n_ok += 1

            except Exception as exc:
                logger.warning("%s: failed — %s", entry_id, exc)
                writer.writerow([entry_id] + [""] * (len(header) - 1))
                n_fail += 1

    logger.info("Done — %d ok, %d failed. Results: %s", n_ok, n_fail, args.output)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

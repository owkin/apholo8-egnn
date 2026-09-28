"""Command-line interface for apholo-EGNN inference.

Usage:
    uv run apholo-egnn --structure protein.pdb
    uv run apholo-egnn --folder /path/to/structures/ --device cuda
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from apholo_egnn.predictor import VNEGNNPredictor
from apholo_egnn.utils.nms import nms_order
from apholo_egnn.utils.structure import parse_chains

logger = logging.getLogger(__name__)

_STRUCTURE_EXTENSIONS = {".pdb", ".cif", ".mmcif"}


def _fmt_site(conf: float, coord: np.ndarray) -> list:
    return [f"{conf:.6f}", f"{coord[0]:.3f}", f"{coord[1]:.3f}", f"{coord[2]:.3f}"]


def _rank_predictions(
    coords: np.ndarray,
    confidence: np.ndarray,
    nms_radius: float,
    top: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute NMS ranks and optionally slice to top-N.

    Expects coords/confidence already sorted by descending confidence (as returned
    by VNEGNNPredictor.predict). Returns (coords, confidence, nms_rank) all of
    length min(K, top). nms_rank[i] is the 1-based NMS rank of prediction i.
    """
    perm = nms_order(coords, confidence, radius=nms_radius)
    nms_rank = np.empty(len(coords), dtype=int)
    for nms_pos, conf_idx in enumerate(perm, 1):
        nms_rank[conf_idx] = nms_pos

    if top is not None:
        coords = coords[:top]
        confidence = confidence[:top]
        nms_rank = nms_rank[:top]

    return coords, confidence, nms_rank


def _run_single(predictor: VNEGNNPredictor, args: argparse.Namespace) -> None:
    output_path = args.output or args.structure.with_name(args.structure.stem + "_predictions.csv")

    logger.info("Running inference on %s ...", args.structure)
    coords, confidence = predictor.predict(
        structure_path=args.structure,
        num_cycles=args.cycles,
        num_virtual_nodes=args.virtual_nodes,
        chains=args.chains,
        vn_seed=args.vn_seed,
    )
    coords, confidence, nms_rank = _rank_predictions(coords, confidence, args.nms_radius, args.top)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["confidence_rank", "nms_rank", "confidence", "x", "y", "z"])
        for conf_rank, (coord, conf, nr) in enumerate(zip(coords, confidence, nms_rank), 1):
            writer.writerow([conf_rank, nr, *_fmt_site(conf, coord)])

    logger.info(
        "Predictions written to %s  (%d total, NMS radius %.1f Å)",
        output_path,
        len(coords),
        args.nms_radius,
    )


def _run_folder(predictor: VNEGNNPredictor, args: argparse.Namespace) -> None:
    structure_files = sorted(
        p
        for p in args.folder.rglob("*")
        if p.is_file() and p.suffix.lower() in _STRUCTURE_EXTENSIONS
    )
    if not structure_files:
        logger.error("No .pdb / .cif / .mmcif files found in %s", args.folder)
        return

    output_path = args.output or args.folder.with_name(args.folder.name + "_predictions.csv")

    k = args.virtual_nodes * args.cycles
    if args.top is not None:
        k = min(k, args.top)

    header = ["structure"] + [
        f"vn{n}_{field}" for n in range(1, k + 1) for field in ("conf", "x", "y", "z", "nms_rank")
    ]

    n_ok = 0
    n_err = 0
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for struct_path in structure_files:
            logger.info("Processing %s ...", str(struct_path.relative_to(args.folder)))
            try:
                coords, confidence = predictor.predict(
                    structure_path=struct_path,
                    num_cycles=args.cycles,
                    num_virtual_nodes=args.virtual_nodes,
                    vn_seed=args.vn_seed,
                )
                coords, confidence, nms_rank = _rank_predictions(
                    coords, confidence, args.nms_radius, args.top
                )
                row = [str(struct_path.relative_to(args.folder))]
                for coord, conf, nr in zip(coords, confidence, nms_rank):
                    row += [*_fmt_site(conf, coord), nr]
                n_ok += 1
            except Exception as exc:
                logger.warning("  Failed: %s", exc)
                row = [str(struct_path.relative_to(args.folder))] + [""] * (len(header) - 1)
                n_err += 1

            writer.writerow(row)

    logger.info(
        "Predictions written to %s  (%d structures, %d failed, NMS radius %.1f Å)",
        output_path,
        n_ok,
        n_err,
        args.nms_radius,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Predict binding-site locations from a protein structure using apholo-EGNN."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--structure",
        "-s",
        type=Path,
        help="Path to a single structure file (.pdb, .cif, or .mmcif).",
    )
    source.add_argument(
        "--folder",
        "-f",
        type=Path,
        help="Path to a folder of structure files. One CSV row is written per file.",
    )

    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        help="Device for both VNEGNN and ESM-2 3B (default: cuda).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output CSV path (default: <structure stem>_predictions.csv or "
        "<folder name>_predictions.csv).",
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=None,
        help="Limit to the top-N predictions per structure ranked by confidence (default: all).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of iterative prediction cycles (default: 1).",
    )
    parser.add_argument(
        "--virtual-nodes",
        type=int,
        default=8,
        help="Number of virtual nodes / candidate binding sites per cycle (default: 8).",
    )
    parser.add_argument(
        "--nms-radius",
        type=float,
        default=2.0,
        help="NMS minimum separation radius in Å (default: 2.0).",
    )
    parser.add_argument(
        "--chains",
        default=None,
        help="Comma- or hyphen-separated chain IDs to use (single-structure mode only). "
        "E.g. 'A', 'A,B', or 'G-H'.",
    )
    parser.add_argument(
        "--vn-seed",
        type=int,
        default=None,
        help="Integer seed for virtual-node Fibonacci initialisation. "
        "When set, random rotations are enabled and results are reproducible. "
        "When omitted, random rotations are disabled (fully deterministic).",
    )
    args = parser.parse_args()
    if args.chains is not None:
        args.chains = parse_chains(args.chains)

    if args.cycles < 1:
        parser.error("--cycles must be >= 1")
    if args.virtual_nodes < 1:
        parser.error("--virtual-nodes must be >= 1")
    if args.top is not None and args.top < 1:
        parser.error("--top must be >= 1")
    if args.nms_radius <= 0:
        parser.error("--nms-radius must be > 0")
    if args.structure is not None and not args.structure.exists():
        parser.error(f"Structure file not found: {args.structure}")
    if args.folder is not None and not args.folder.is_dir():
        parser.error(f"Not a directory: {args.folder}")

    logger.info(
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n"
        "⣿⣿⣿⠛⢻⣿⣿⣿⣿⠿⠛⡿⠀⠀⠀⠀⠀⠀⢿⠛⠿⣟⣻⡿⠛⠻⣿⣿⣿⣿\n"
        "⣿⣿⣿⡆⣾⣿⣿⠟⠁⠀⣀⣇⣀⣠⣤⣤⣄⣀⣸⣀⠀⠈⠻⣧⣀⣀⣿⣿⣿⣿\n"
        "⣿⣿⣿⡇⣿⣿⠋⢀⣾⠟⠉⢉⣭⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠙⣿⣉⢿⣿⣿⣿\n"
        "⣿⣿⣿⡿⣿⠇⢠⣿⠁⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠸⣿⢿⣿⣿⣿\n"
        "⣿⣿⠇⢰⡟⠀⣾⡇⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⢻⡆⠸⣿⣿\n"
        "⣿⣿⠀⢸⡇⢰⣿⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⢸⡇⠀⣿⣿\n"
        "⣿⣧⡀⢸⡇⠘⣿⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⢸⡇⢀⣼⣿\n"
        "⣿⣿⣿⣾⣇⠀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣸⣷⣿⣿⣿\n"
        "⣿⣿⣿⣿⣿⣄⠈⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⣠⣿⣿⣿⣿⣿\n"
        "⣿⣿⣿⣿⣿⣿⣦⡀⠈⠛⠿⠿⣿⣿⣿⣿⣿⣿⠿⠿⠛⠁⢀⣴⣿⣿⣿⣿⣿⣿\n"
        "⣿⣿⣿⣿⣿⡃⠈⠛⠷⣦⣤⣀⣀⣀⠀⠀⣀⣀⣀⣤⣴⠾⠛⠁⢘⣿⣿⣿⣿⣿\n"
        "⣿⣿⠟⠋⠈⠙⠷⣦⣤⣀⡈⠉⢹⡟⠛⠛⢻⡏⠉⢁⣀⣤⣴⠾⠋⠁⠙⠻⣿⣿\n"
        "⣿⠀⠀⠀⠀⠀⠀⠀⠈⠉⠙⠛⠛⠷⠶⠶⠾⠛⠛⠋⠉⠁⠀⠀⠀⠀⠀⠀⠀⣿\n"
        "⣿⣤⣤⣼⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣧⣤⣤⣿"
    )

    logger.info("Loading model ...")
    predictor = VNEGNNPredictor(device=args.device)

    if args.structure is not None:
        _run_single(predictor, args)
    else:
        _run_folder(predictor, args)


if __name__ == "__main__":
    main()

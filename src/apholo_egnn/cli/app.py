"""Command-line interface for apholo-EGNN inference.

Usage:
    uv run python -m apholo_egnn.cli.app --structure protein.pdb
    uv run python -m apholo_egnn.cli.app --structure protein.cif --device cuda --top 3
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from apholo_egnn.predictor import VNEGNNPredictor

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Predict binding-site locations from a protein structure using apholo-EGNN."
    )
    parser.add_argument(
        "--structure",
        "-s",
        required=True,
        type=Path,
        help="Path to the input structure file (.pdb, .cif, or .mmcif).",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cpu",
        help="Device for both VNEGNN and ESM-2 3B (default: cpu). Use 'cuda' for GPU.",
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=None,
        help="Print only the top-N predictions ranked by confidence (default: all).",
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
    args = parser.parse_args()

    if args.cycles < 1:
        parser.error("--cycles must be >= 1")
    if args.virtual_nodes < 1:
        parser.error("--virtual-nodes must be >= 1")
    if args.top is not None and args.top < 1:
        parser.error("--top must be >= 1")

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

    if not args.structure.exists():
        parser.error(f"Structure file not found: {args.structure}")

    logger.info("Loading model ...")
    predictor = VNEGNNPredictor(device=args.device)

    logger.info("Running inference on %s ...", args.structure)
    coords, confidence = predictor.predict(
        structure_path=args.structure,
        num_cycles=args.cycles,
        num_virtual_nodes=args.virtual_nodes,
    )

    # Rank by confidence (descending)
    order = np.argsort(confidence)[::-1]
    coords = coords[order]
    confidence = confidence[order]

    if args.top is not None:
        coords = coords[: args.top]
        confidence = confidence[: args.top]

    logger.info(
        "\nPredicted binding sites (%d total, ranked by confidence):", len(coords)
    )
    logger.info("%-6s %12s  %10s %10s %10s", "Rank", "Confidence", "x", "y", "z")
    logger.info("-" * 55)
    for i, (coord, conf) in enumerate(zip(coords, confidence), start=1):
        logger.info(
            "%-6d %12.4f  %10.3f %10.3f %10.3f", i, conf, coord[0], coord[1], coord[2]
        )


if __name__ == "__main__":
    main()

"""Command-line interface for apholo-EGNN inference.

Usage:
    uv run python predict.py --pdb protein.pdb
    uv run python predict.py --pdb protein.pdb --device cuda --top 3
"""
import argparse
from pathlib import Path

import numpy as np

from predictor import VNEGNNPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Predict binding-site locations from a PDB file using VN-EGNN."
    )
    parser.add_argument(
        "--pdb", "-p",
        required=True,
        type=Path,
        help="Path to the input PDB file.",
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        help="Device for both VNEGNN and ESM-2 3B (default: cpu). Use 'cuda' for GPU.",
    )
    parser.add_argument(
        "--top", "-n",
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

    print(
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

    if not args.pdb.exists():
        parser.error(f"PDB file not found: {args.pdb}")

    print("Loading model ...")
    predictor = VNEGNNPredictor(device=args.device)

    print(f"Running inference on {args.pdb} ...")
    coords, confidence = predictor.predict(
        pdb_path=args.pdb,
        num_cycles=args.cycles,
        num_virtual_nodes=args.virtual_nodes,
    )

    # Rank by confidence (descending)
    order = np.argsort(confidence)[::-1]
    coords     = coords[order]
    confidence = confidence[order]

    if args.top is not None:
        coords     = coords[: args.top]
        confidence = confidence[: args.top]

    print(f"\nPredicted binding sites ({len(coords)} total, ranked by confidence):")
    print(f"{'Rank':<6} {'Confidence':>12}  {'x':>10} {'y':>10} {'z':>10}")
    print("-" * 55)
    for i, (coord, conf) in enumerate(zip(coords, confidence), start=1):
        print(f"{i:<6} {conf:>12.4f}  {coord[0]:>10.3f} {coord[1]:>10.3f} {coord[2]:>10.3f}")


if __name__ == "__main__":
    main()

"""Convert a Lightning .ckpt to a plain safetensors file.

The Lightning checkpoint stores the full training state (optimiser, hparams,
callbacks …) and embeds OmegaConf/Hydra objects that are not needed at
inference time.  This script extracts only the VNEGNN backbone weights and
saves them as a self-contained safetensors file.

Usage:
    python scripts/convert_checkpoint.py \
        --input  logs/train/runs/.../checkpoints/last.ckpt \
        --output model.safetensors
"""

import argparse
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file


def _load_lightning_checkpoint(path: Path) -> dict:
    """Load a Lightning .ckpt, stubbing omegaconf if not installed."""
    if "omegaconf" not in sys.modules:
        try:
            import omegaconf  # noqa: F401
        except ImportError:
            stub = types.ModuleType("omegaconf")
            stub.DictConfig = dict  # type: ignore[attr-defined]
            sys.modules["omegaconf"] = stub
            sys.modules["omegaconf.dictconfig"] = stub

    return torch.load(str(path), map_location="cpu", weights_only=False)


def convert(input_path: Path, output_path: Path) -> None:
    print(f"Loading  {input_path}")
    ckpt = _load_lightning_checkpoint(input_path)

    state_dict = ckpt["state_dict"]
    all_keys = list(state_dict.keys())
    print(f"  {len(all_keys)} keys in Lightning state_dict")
    print(f"  top-level prefixes: {sorted({k.split('.')[0] for k in all_keys})}")

    backbone = {
        k[len("backbone."):]: v.detach().cpu().contiguous()
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }
    print(f"  {len(backbone)} backbone keys extracted")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(backbone, str(output_path))
    print(f"Saved    {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  "-i", required=True, type=Path, help="Path to Lightning .ckpt file")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output .safetensors path")
    args = parser.parse_args()
    convert(args.input, args.output)

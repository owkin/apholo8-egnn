"""VNEGNNPredictor — load a safetensors checkpoint and run binding-site inference."""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch_geometric.data import Batch
from torch_geometric.nn import MeanAggregation

from apholo_egnn.data.graph import build_graph
from apholo_egnn.model.vnegnn import VNEGNN
from apholo_egnn.utils.grid import sample_fibonacci_grid

_HF_REPO_ID  = "apholo/vnegnn"      # TODO: update once uploaded
_HF_FILENAME = "model.safetensors"  # TODO: update once uploaded

_ESM_MODEL_NAME = "esm2_t36_3B_UR50D"
_ESM_REPR_LAYER = 36

# Architecture of the published VN-EGNN model trained with ESM-2 3B.
# input_features = 2560 (ESM-2 3B) + 21 (one-hot residue encoding)
_VNEGNN_KWARGS = dict(
    input_features=2581,
    node_features=100,
    edge_features=1,
    hidden_features=100,
    out_features=100,
    num_layers=5,
    dropout=0.1,
    residual=True,
    weight_share=False,
    norm_coords=True,
    norm_coors_scale_init=0.01,
    norm_feats=True,
    initialization_gain=1.0,
    act=nn.SiLU,
    node_aggr=MeanAggregation,
    cord_aggr=MeanAggregation,
)
_SCALING_FACTOR = 5.0


def _parse_pdb(pdb_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Cα coordinates and 3-letter residue names from a PDB file.

    Mirrors extract_binding_info.py: standard residues only, skips residues
    without a CA atom and non-standard amino acids that seq1() cannot convert.

    Returns:
        res_coords: [N, 3] float32 Cα coordinates in Angstroms.
        res_names:  [N]    3-letter residue codes (e.g. "ALA", "GLY").
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    coords = []
    res_names = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":  # skip HETATM / water
                    continue
                if "CA" not in residue:
                    continue
                try:
                    seq1(residue.resname)  # raises KeyError for non-standard AA
                except KeyError:
                    continue
                coords.append(residue["CA"].get_coord())
                res_names.append(residue.resname)

    return np.array(coords, dtype=np.float32), np.array(res_names)


class VNEGNNPredictor:
    """
    Load a VN-EGNN checkpoint and predict binding-site locations for a protein.

    ESM-2 3B embeddings are computed automatically from the protein sequence
    extracted from the PDB file.  ESM weights are downloaded on first use and
    cached by PyTorch Hub.

    Usage::

        predictor = VNEGNNPredictor(device="cuda")
        coords, confidence = predictor.predict("protein.pdb")
        # coords:     [K, 3]  predicted binding-site positions
        # confidence: [K]     rank descending — confidence.argmax() gives the top prediction
    """

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: Device for both VNEGNN and ESM-2 3B ("cpu", "cuda", "cuda:1", …).
        """
        self.device      = torch.device(device)
        self._esm_device = torch.device(device)

        checkpoint_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_FILENAME)
        model = VNEGNN(**_VNEGNN_KWARGS)
        state_dict = load_file(checkpoint_path, device="cpu")
        model.load_state_dict(state_dict)
        self.model = model.eval().to(self.device)

        # ESM-2 3B is loaded lazily on the first predict() call
        self._esm_model    = None
        self._esm_alphabet = None

    def _ensure_esm_loaded(self) -> None:
        if self._esm_model is not None:
            return
        try:
            import esm as esm_lib
        except ImportError:
            raise ImportError(
                "fair-esm is required. Install with:  pip install fair-esm"
            )
        esm_model, alphabet = esm_lib.pretrained.esm2_t36_3B_UR50D()
        self._esm_model    = esm_model.eval().to(self._esm_device)
        self._esm_alphabet = alphabet

    def _compute_esm(self, sequence: str) -> np.ndarray:
        """Return ESM-2 3B per-residue embeddings, shape [N, 2560]."""
        self._ensure_esm_loaded()
        batch_converter = self._esm_alphabet.get_batch_converter()
        _, _, tokens = batch_converter([("protein", sequence)])
        tokens = tokens.to(self._esm_device)
        with torch.no_grad():
            out = self._esm_model(tokens, repr_layers=[_ESM_REPR_LAYER])
        # Remove BOS and EOS tokens → [N, 2560]
        emb = out["representations"][_ESM_REPR_LAYER][0, 1 : len(sequence) + 1]
        return emb.cpu().numpy()

    @torch.no_grad()
    def predict(
        self,
        pdb_path: str | Path,
        num_cycles: int = 1,
        num_virtual_nodes: int = 8,
        max_neighbours: int = 10,
        dist_cutoff: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict binding-site locations for a protein from its PDB file.

        Args:
            pdb_path:          Path to the protein PDB file.  Only ATOM records
                               with a CA atom are used (HETATM and non-standard
                               residues are skipped), matching the training pipeline.
            num_cycles:        Iterative prediction cycles — each cycle re-seeds
                               virtual nodes around pockets found in the previous
                               cycle (default: 1).
            num_virtual_nodes: Candidate binding sites per cycle (default: 8).
            max_neighbours:    Max atom neighbours in the radius graph (default: 10).
            dist_cutoff:       Radius in Å for atom-atom edges (default: 10.0).

        Returns:
            coords:     np.ndarray [K, 3] — predicted binding-site coordinates.
            confidence: np.ndarray [K]   — confidence score per prediction.
                        K = num_virtual_nodes × num_cycles.
                        Sort descending by confidence to rank candidates.
        """
        res_coords, res_names = _parse_pdb(pdb_path)
        if len(res_coords) == 0:
            raise ValueError(f"No standard residues with CA atoms found in {pdb_path}")

        sequence = "".join(seq1(r) for r in res_names)
        esm_embeddings = self._compute_esm(sequence)

        graph = build_graph(
            res_coords=res_coords,
            esm_embeddings=esm_embeddings,
            res_names=res_names,
            num_virtual_nodes=num_virtual_nodes,
            max_neighbours=max_neighbours,
            dist_cutoff=dist_cutoff,
        )
        batch = Batch.from_data_list([graph]).to(self.device)

        all_coords: list[torch.Tensor] = []
        all_conf:   list[torch.Tensor] = []

        for cycle in range(num_cycles):
            x_atom, pos_gn, _, conf = self.model(
                x_atom=batch.x_dict["atom"],
                pos_atom=batch.pos_dict["atom"] / _SCALING_FACTOR,
                x_global_node=batch.x_dict["global_node"],
                pos_global_node=batch.pos_dict["global_node"] / _SCALING_FACTOR,
                edge_index_atom_atom=batch.edge_index_dict[("atom", "to", "atom")],
                edge_index_atom_global_node=batch.edge_index_dict[("atom", "to", "global_node")],
                edge_index_global_node_atom=batch.edge_index_dict[("global_node", "to", "atom")],
            )
            all_coords.append((pos_gn * _SCALING_FACTOR).cpu())
            all_conf.append(conf.squeeze(-1).cpu())

            if cycle < num_cycles - 1:
                batch["global_node"].pos = sample_fibonacci_grid(
                    batch.centroid,
                    batch.radius,
                    num_virtual_nodes,
                    random_rotations=True,
                ).to(self.device)

        coords     = torch.cat(all_coords, dim=0).numpy()
        confidence = torch.cat(all_conf,   dim=0).numpy()
        return coords, confidence

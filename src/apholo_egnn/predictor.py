"""VNEGNNPredictor: load a safetensors checkpoint and run binding-site inference."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio.SeqUtils import seq1
from safetensors.torch import load_file
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import MeanAggregation

from apholo_egnn.data.graph import build_graph
from apholo_egnn.model.vnegnn import VNEGNN
from apholo_egnn.utils.grid import sample_fibonacci_grid
from apholo_egnn.utils.hub import download_from_pretrained
from apholo_egnn.utils.structure import parse_structure, parse_structure_by_chain

logger = logging.getLogger(__name__)
_ESM_REPR_LAYER = 36

# Architecture of the published VN-EGNN model trained with ESM-2 3B.
# input_features = 2560 (ESM-2 3B) + 21 (one-hot residue encoding)
_VNEGNN_KWARGS = {
    "input_features": 2581,
    "node_features": 100,
    "edge_features": 1,
    "hidden_features": 100,
    "out_features": 100,
    "num_layers": 5,
    "dropout": 0.1,
    "residual": True,
    "weight_share": False,
    "norm_coords": True,
    "norm_coors_scale_init": 0.01,
    "norm_feats": True,
    "initialization_gain": 1.0,
    "act": nn.SiLU,
    "node_aggr": MeanAggregation,
    "cord_aggr": MeanAggregation,
}
_SCALING_FACTOR = 5.0


class VNEGNNPredictor:
    """
    Load a VN-EGNN checkpoint and predict binding-site locations for a protein.

    ESM-2 3B embeddings are computed automatically from the protein sequence
    extracted from the structure file.  ESM weights are downloaded on first
    use and cached by PyTorch Hub.

    Usage::

        predictor = VNEGNNPredictor(device="cuda")
        coords, confidence = predictor.predict("protein.pdb")  # or "protein.cif"
        # coords:     [K, 3]  predicted binding-site positions
        # confidence: [K]     rank descending: confidence.argmax() gives the top prediction
    """

    def __init__(self, device: str | None = None):
        """
        Args:
            device: Device for both VNEGNN and ESM-2 3B ("cpu", "cuda", "cuda:1", …).
                    Defaults to "cuda" if a GPU is available, otherwise "cpu".
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._esm_device = torch.device(device)
        logger.info("Using device: %s", self.device)

        checkpoint_path = download_from_pretrained("owkin/apholo-egnn")
        logger.debug("Loading VNEGNN checkpoint from %s", checkpoint_path)
        model = VNEGNN(**_VNEGNN_KWARGS)
        state_dict = load_file(checkpoint_path, device="cpu")
        model.load_state_dict(state_dict)
        self.model = model.eval().to(self.device)
        logger.debug(
            "VNEGNN model ready (%d parameters)",
            sum(p.numel() for p in model.parameters()),
        )

        # ESM-2 3B is loaded lazily on the first predict() call
        self._esm_model: Any = None
        self._esm_alphabet: Any = None

    def _ensure_esm_loaded(self) -> None:
        if self._esm_model is not None:
            return
        import esm as esm_lib

        logger.info("Loading ESM-2 3B model onto %s", self._esm_device)
        esm_model, alphabet = esm_lib.pretrained.esm2_t36_3B_UR50D()
        self._esm_model = esm_model.eval().to(self._esm_device)
        self._esm_alphabet = alphabet

    def _compute_esm(self, sequence: str) -> np.ndarray:
        """Return ESM-2 3B per-residue embeddings for a single sequence, shape [N, 2560]."""
        self._ensure_esm_loaded()
        logger.debug("Computing ESM-2 embeddings for sequence of length %d", len(sequence))
        batch_converter = self._esm_alphabet.get_batch_converter()
        _, _, tokens = batch_converter([("protein", sequence)])
        tokens = tokens.to(self._esm_device)
        with torch.no_grad():
            out = self._esm_model(tokens, repr_layers=[_ESM_REPR_LAYER])
        # Remove BOS and EOS tokens → [N, 2560]
        emb = out["representations"][_ESM_REPR_LAYER][0, 1 : len(sequence) + 1]
        return emb.float().cpu().numpy()

    def _compute_esm_per_chain(self, chain_seqs: list[tuple[str, str]]) -> np.ndarray:
        """Embed each chain independently with ESM-2 3B, return concatenated [N, 2560].

        Each chain is embedded then the per-residue representations are concatenated in chain order.
        """
        self._ensure_esm_loaded()
        batch_converter = self._esm_alphabet.get_batch_converter()
        parts = []
        for chain_id, seq in chain_seqs:
            logger.debug("Computing ESM-2 embeddings for chain %s (length %d)", chain_id, len(seq))
            _, _, tokens = batch_converter([(chain_id, seq)])
            tokens = tokens.to(self._esm_device)
            with torch.no_grad():
                out = self._esm_model(tokens, repr_layers=[_ESM_REPR_LAYER])
            emb = out["representations"][_ESM_REPR_LAYER][0, 1 : len(seq) + 1]
            parts.append(emb.float().cpu().numpy())
        return np.concatenate(parts, axis=0)

    @torch.no_grad()
    def predict(
        self,
        structure_path: str | Path,
        num_cycles: int = 1,
        num_virtual_nodes: int = 8,
        max_neighbours: int = 10,
        dist_cutoff: float = 10.0,
        chains: list[str] | None = None,
        vn_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict binding-site locations for a protein from a PDB or mmCIF file.

        Args:
            structure_path:    Path to the protein structure file (.pdb, .cif,
                               or .mmcif; parser is picked by extension). Only
                               ATOM records with a CA atom are used (HETATM
                               and non-standard residues are skipped), matching
                               the training pipeline.
            num_cycles:        Iterative prediction cycles: each cycle re-seeds
                               virtual nodes around pockets found in the previous
                               cycle (default: 1).
            num_virtual_nodes: Candidate binding sites per cycle (default: 8).
            max_neighbours:    Max atom neighbours in the radius graph (default: 10).
            dist_cutoff:       Radius in Å for atom-atom edges (default: 10.0).
            chains:            If given, only residues from these chain IDs are
                               used (e.g. ['G', 'H'] for a CryptoBench entry
                               '1fd4_G-H').
            vn_seed:           Integer seed for the random Fibonacci-sphere rotation
                               used to initialise the virtual nodes. When None, the
                               global torch random state is used as-is. Passing an
                               explicit integer makes a single run reproducible.

        Returns:
            coords:     np.ndarray [K, 3]: predicted binding-site coordinates,
                        sorted by descending confidence. K = num_virtual_nodes × num_cycles.
            confidence: np.ndarray [K]: sigmoid confidence score per prediction,
                        descending (values in (0, 1)).
        """
        if chains is not None:
            chain_data = parse_structure_by_chain(structure_path, chains)
            if not chain_data:
                raise ValueError(f"No standard residues with CA atoms found in {structure_path}")
            res_coords = np.concatenate([c for _, c, _ in chain_data])
            res_names = np.concatenate([n for _, _, n in chain_data])
            chain_seqs = [(cid, "".join(seq1(r) for r in names)) for cid, _, names in chain_data]
            esm_embeddings = self._compute_esm_per_chain(chain_seqs)
        else:
            res_coords, res_names = parse_structure(structure_path)
            if len(res_coords) == 0:
                raise ValueError(f"No standard residues with CA atoms found in {structure_path}")
            sequence = "".join(seq1(r) for r in res_names)
            esm_embeddings = self._compute_esm(sequence)

        if vn_seed is not None:
            torch.manual_seed(vn_seed)

        logger.debug(
            "Building graph (virtual_nodes=%d, max_neighbours=%d, dist_cutoff=%.1f Å)",
            num_virtual_nodes,
            max_neighbours,
            dist_cutoff,
        )
        graph = build_graph(
            res_coords=res_coords,
            esm_embeddings=esm_embeddings,
            res_names=res_names,
            num_virtual_nodes=num_virtual_nodes,
            max_neighbours=max_neighbours,
            dist_cutoff=dist_cutoff,
            random_rotations=(vn_seed is not None),
        )
        batch = Batch.from_data_list([graph]).to(self.device)

        all_coords: list[torch.Tensor] = []
        all_conf: list[torch.Tensor] = []

        for cycle in range(num_cycles):
            logger.debug("Inference cycle %d/%d", cycle + 1, num_cycles)
            _, pos_gn, _, conf = self.model(
                x_atom=batch.x_dict["atom"],
                pos_atom=batch.pos_dict["atom"] / _SCALING_FACTOR,
                x_global_node=batch.x_dict["global_node"],
                pos_global_node=batch.pos_dict["global_node"] / _SCALING_FACTOR,
                edge_index_atom_atom=batch.edge_index_dict[("atom", "to", "atom")],
                edge_index_atom_global_node=batch.edge_index_dict[("atom", "to", "global_node")],
                edge_index_global_node_atom=batch.edge_index_dict[("global_node", "to", "atom")],
            )
            all_coords.append((pos_gn * _SCALING_FACTOR).cpu())
            all_conf.append(torch.sigmoid(conf.squeeze(-1)).cpu())

            if cycle < num_cycles - 1:
                batch["global_node"].pos = sample_fibonacci_grid(
                    batch.centroid,
                    batch.radius,
                    num_virtual_nodes,
                    random_rotations=True,
                ).to(self.device)

        coords = torch.cat(all_coords, dim=0).numpy()
        confidence = torch.cat(all_conf, dim=0).numpy()

        order = np.argsort(confidence)[::-1]
        coords, confidence = coords[order], confidence[order]

        logger.info(
            "Done: %d binding-site predictions, top confidence: %.4f",
            len(coords),
            float(confidence[0]),
        )
        return coords, confidence

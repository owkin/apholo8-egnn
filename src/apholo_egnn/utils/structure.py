"""PDB / mmCIF parsing utilities."""

import logging
import re
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1

logger = logging.getLogger(__name__)


def _load_structure(path: str | Path) -> Structure:
    """Parse a PDB or mmCIF file, dispatching by extension."""
    path = Path(path)
    parser: MMCIFParser | PDBParser
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("protein", str(path))


def parse_chains(chains_str: str) -> list[str]:
    """Parse a chain string like 'A', 'A,B', or 'G-H' into a list of chain IDs.

    Hyphens and commas are treated as separators (matching the CryptoBench
    apo_chain convention where 'G-H' means chains G and H).
    """
    return [c for c in re.split(r"[-,]+", chains_str.strip()) if c]


def parse_structure(
    path: str | Path,
    chains: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Cα coordinates and 3-letter residue names from a PDB or mmCIF file.

    Args:
        path:   Path to the structure file (.pdb, .cif, or .mmcif).
        chains: If given, only residues from these chain IDs are included.
                Matches the CryptoBench apo_chain filtering used during
                VN-EGNN training (e.g. chains=['G','H'] for entry '1fd4_G-H').

    Returns:
        res_coords: [N, 3] float32 Cα coordinates in Angstroms.
        res_names:  [N]    3-letter residue codes (e.g. "ALA", "GLY").
    """
    structure = _load_structure(path)
    chain_set = set(chains) if chains is not None else None

    coords = []
    res_names = []

    # We only take the primary conformer (model 0); NMR ensembles would otherwise
    # accumulate residues from every conformer.
    for chain in structure[0]:
        if chain_set is not None and chain.id not in chain_set:
            continue
        for residue in chain:
            if residue.id[0] != " ":
                continue
            if "CA" not in residue:
                continue
            try:
                seq1(residue.resname)
            except KeyError:
                logger.warning("Skipping non-standard residue: %s", residue.resname)
                continue
            coords.append(residue["CA"].get_coord())
            res_names.append(residue.resname)

    return np.array(coords, dtype=np.float32), np.array(res_names)


def parse_structure_by_chain(
    path: str | Path,
    chains: list[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Extract Cα coordinates and residue names separately per chain, in chain order.

    Returns a list of (chain_id, coords [N_i, 3], res_names [N_i]) tuples,
    one per chain in the order specified by `chains`. Chains with no valid
    residues are omitted. Matches vnegnn's per-chain ESM embedding strategy
    where each chain is embedded independently before concatenation.
    """
    structure = _load_structure(path)
    chain_set = set(chains)
    buckets: dict[str, tuple[list, list]] = {c: ([], []) for c in chains}

    # We only take the primary conformer (model 0); NMR ensembles would otherwise
    # accumulate residues from every conformer.
    for chain in structure[0]:
        if chain.id not in chain_set:
            continue
        coords_list, names_list = buckets[chain.id]
        for residue in chain:
            if residue.id[0] != " ":
                continue
            if "CA" not in residue:
                continue
            try:
                seq1(residue.resname)
            except KeyError:
                logger.warning("Skipping non-standard residue: %s", residue.resname)
                continue
            coords_list.append(residue["CA"].get_coord())
            names_list.append(residue.resname)

    result = []
    for chain_id in chains:
        coords_list, names_list = buckets[chain_id]
        if coords_list:
            result.append(
                (
                    chain_id,
                    np.array(coords_list, dtype=np.float32),
                    np.array(names_list),
                )
            )
    return result

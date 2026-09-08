"""PDB / mmCIF parsing utilities."""

import logging
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


def parse_structure(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract Cα coordinates and 3-letter residue names from a PDB or mmCIF file.

    Returns:
        res_coords: [N, 3] float32 Cα coordinates in Angstroms.
        res_names:  [N]    3-letter residue codes (e.g. "ALA", "GLY").
    """
    structure = _load_structure(path)

    coords = []
    res_names = []

    for residue in structure.get_residues():
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

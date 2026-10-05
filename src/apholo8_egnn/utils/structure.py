"""PDB / mmCIF parsing utilities."""

import io
import logging
import re
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1

logger = logging.getLogger(__name__)


def _load_structure(content: str, fmt: str) -> Structure:
    """Parse PDB or mmCIF text, using the format returned by `detect_format`."""
    parser: MMCIFParser | PDBParser
    if fmt == "cif":
        parser = MMCIFParser(QUIET=True)
    elif fmt == "pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"unknown structure format: {fmt!r} (expected 'pdb' or 'cif')")
    return parser.get_structure("protein", io.StringIO(content))


def parse_chains(chains_str: str) -> list[str]:
    """Parse a chain string like 'A', 'A,B', or 'G-H' into a list of chain IDs.

    Hyphens and commas are treated as separators (matching the CryptoBench
    apo_chain convention where 'G-H' means chains G and H).
    """
    return [c for c in re.split(r"[-,]+", chains_str.strip()) if c]


def parse_structure(
    content: str,
    fmt: str,
    chains: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Cα coordinates and 3-letter residue names from PDB or mmCIF text.

    Args:
        content: Text content of the structure file (see `read_structure`).
        fmt:     "pdb" or "cif" (see `detect_format`).
        chains:  If given, only residues from these chain IDs are included.
                 Matches the CryptoBench apo_chain filtering used during
                 VN-EGNN training (e.g. chains=['G','H'] for entry '1fd4_G-H').

    Returns:
        res_coords: [N, 3] float32 Cα coordinates in Angstroms.
        res_names:  [N]    3-letter residue codes (e.g. "ALA", "GLY").
    """
    structure = _load_structure(content, fmt)
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
    content: str,
    fmt: str,
    chains: list[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Extract Cα coordinates and residue names separately per chain, in chain order.

    Returns a list of (chain_id, coords [N_i, 3], res_names [N_i]) tuples,
    one per chain in the order specified by `chains`. Chains with no valid
    residues are omitted. Matches vnegnn's per-chain ESM embedding strategy
    where each chain is embedded independently before concatenation.
    """
    structure = _load_structure(content, fmt)
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


PROTEIN_RESIDUES = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE",
    "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
    "THR", "VAL", "TRP", "TYR", "MSE", "SEC", "PYL",
}  # fmt: skip


def validate_pdb(content: str) -> tuple[bool, str]:
    """
    Check PDB content has atoms and at least one protein residue.

    Parameters
    ----------
    content : str
        Text content of the PDB file

    Returns
    -------
    bool
        True if valid protein structure
    str
        Status message
    """
    lines = content.splitlines()

    if not lines:
        return False, "empty file"

    has_atoms = False
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            has_atoms = True
            # Residue name is at columns 18-20 (0-indexed: 17-20)
            if len(line) >= 20 and line[17:20].strip() in PROTEIN_RESIDUES:
                return True, "valid PDB structure"

    if not has_atoms:
        return False, "no atom records found"
    return False, "no protein residues found"


def validate_cif(content: str) -> tuple[bool, str]:
    """
    Check mmCIF content has data block, atoms, and protein residues.

    Parameters
    ----------
    content : str
        Text content of the mmCIF file

    Returns
    -------
    bool
        True if valid protein structure
    str
        Status message
    """
    if not content:
        return False, "empty file"

    has_data = "data_" in content
    has_atoms = "_atom_site." in content or "ATOM" in content
    has_protein = any(res in content for res in PROTEIN_RESIDUES)

    if not has_data:
        return False, "no mmCIF data block"
    if not has_atoms:
        return False, "no atom records found"
    if not has_protein:
        return False, "no protein residues found"

    return True, "valid mmCIF structure"


def detect_format(path: Path, content: str) -> str | None:
    """Return "pdb" or "cif" from the file extension, else from the content; None if unknown."""
    suffix = path.suffix.lower()

    # Detect format by extension
    if suffix in [".pdb", ".ent"]:
        return "pdb"
    elif suffix in [".cif", ".mmcif"]:
        return "cif"

    # Try detecting from content
    if content.startswith("data_"):
        return "cif"
    elif any(content.startswith(p) for p in ["HEADER", "ATOM  ", "HETATM", "CRYST1"]):
        return "pdb"

    return None


def read_structure(file_path) -> tuple[str, str]:
    """Read a structure file once and detect its format.

    Returns:
        content: Text content of the file.
        fmt:     "pdb" or "cif".

    Raises:
        ValueError: If the file is missing, unreadable, or in an unknown format.
    """
    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"file not found: {file_path}")

    try:
        with open(path) as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"error reading file: {e}") from e

    fmt = detect_format(path, content)
    if fmt is None:
        raise ValueError(f"unknown format (extension: {path.suffix.lower()})")
    return content, fmt


def validate_content(content: str, fmt: str) -> tuple[bool, str]:
    """Run `validate_pdb` or `validate_cif` depending on `fmt`."""
    return validate_pdb(content) if fmt == "pdb" else validate_cif(content)


def validate_protein_structure(file_path) -> tuple[bool, str]:
    """
    Validate protein structure file in PDB or mmCIF format.

    Parameters
    ----------
    file_path : str
        Path to the structure file

    Returns
    -------
    bool
        True if valid protein structure
    str
        Status message describing validation result
    """
    try:
        content, fmt = read_structure(file_path)
    except ValueError as e:
        return False, str(e)
    return validate_content(content, fmt)

"""Tiny PDB / mmCIF structures shared by the tests."""

import pytest

# (record, atom name, residue name, chain, residue number)
PROTEIN_ATOMS = [
    ("ATOM", "N", "ALA", "A", 1),
    ("ATOM", "CA", "ALA", "A", 1),
    ("ATOM", "C", "ALA", "A", 1),
    ("ATOM", "N", "GLY", "A", 2),
    ("ATOM", "CA", "GLY", "A", 2),
    ("ATOM", "C", "GLY", "A", 2),
]
LIGAND_ATOMS = [("HETATM", "C1", "LIG", "A", 1), ("HETATM", "O1", "LIG", "A", 1)]


def _pdb_text(atoms: list[tuple[str, str, str, str, int]]) -> str:
    lines = []
    for serial, (record, name, resname, chain, resseq) in enumerate(atoms, 1):
        x, y, z = float(serial), 0.0, 0.0
        lines.append(
            f"{record:<6}{serial:>5}  {name:<3} {resname:>3} {chain}{resseq:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {name[0]:>2}"
        )
    return "\n".join(lines + ["END", ""])


def _cif_text(atoms: list[tuple[str, str, str, str, int]]) -> str:
    fields = [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
        "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
        "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
        "B_iso_or_equiv", "auth_seq_id", "auth_asym_id", "pdbx_PDB_model_num",
    ]  # fmt: skip
    lines = ["data_test", "loop_"] + [f"_atom_site.{f}" for f in fields]
    for serial, (record, name, resname, chain, resseq) in enumerate(atoms, 1):
        lines.append(
            f"{record} {serial} {name[0]} {name} . {resname} {chain} 1 {resseq} ? "
            f"{float(serial):.3f} 0.000 0.000 1.00 0.00 {resseq} {chain} 1"
        )
    return "\n".join(lines + ["#", ""])


@pytest.fixture
def protein_pdb() -> str:
    """PDB text of a 2-residue protein (ALA, GLY) on chain A."""
    return _pdb_text(PROTEIN_ATOMS)


@pytest.fixture
def protein_cif() -> str:
    """mmCIF text of the same 2-residue protein."""
    return _cif_text(PROTEIN_ATOMS)


@pytest.fixture
def ligand_pdb() -> str:
    """PDB text with only a ligand (HETATM records, no protein residue)."""
    return _pdb_text(LIGAND_ATOMS)

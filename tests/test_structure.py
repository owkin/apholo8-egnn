from pathlib import Path

import pytest

from apholo8_egnn.utils.structure import (
    detect_format,
    parse_structure,
    parse_structure_by_chain,
    validate_cif,
    validate_pdb,
    validate_protein_structure,
)

GARBAGE = "this is not a structure file\n"


def test_validate_pdb_valid(protein_pdb):
    assert validate_pdb(protein_pdb) == (True, "valid PDB structure")


def test_validate_pdb_empty():
    assert validate_pdb("") == (False, "empty file")


def test_validate_pdb_without_atoms():
    assert validate_pdb(GARBAGE) == (False, "no atom records found")


def test_validate_pdb_ligand_only(ligand_pdb):
    assert validate_pdb(ligand_pdb) == (False, "no protein residues found")


def test_validate_cif_valid(protein_cif):
    assert validate_cif(protein_cif) == (True, "valid mmCIF structure")


def test_validate_cif_empty():
    assert validate_cif("") == (False, "empty file")


def test_validate_cif_without_data_block():
    assert validate_cif(GARBAGE) == (False, "no mmCIF data block")


def test_missing_file(tmp_path):
    ok, msg = validate_protein_structure(tmp_path / "missing.pdb")
    assert not ok
    assert "file not found" in msg


@pytest.mark.parametrize(
    "name, expected",
    [("x.pdb", "pdb"), ("x.ent", "pdb"), ("x.cif", "cif"), ("x.mmcif", "cif"), ("x.CIF", "cif")],
)
def test_detect_format_by_extension(name, expected):
    assert detect_format(Path(name), GARBAGE) == expected


def test_detect_format_pdb_content(protein_pdb):
    assert detect_format(Path("x.txt"), protein_pdb) == "pdb"


def test_detect_format_cif_content(protein_cif):
    assert detect_format(Path("x.txt"), protein_cif) == "cif"


def test_detect_format_unknown():
    assert detect_format(Path("x.txt"), GARBAGE) is None


def test_parse_structure_pdb(protein_pdb):
    coords, names = parse_structure(protein_pdb, "pdb")
    assert coords.shape == (2, 3)
    assert list(names) == ["ALA", "GLY"]


def test_parse_structure_cif(protein_cif):
    coords, names = parse_structure(protein_cif, "cif")
    assert coords.shape == (2, 3)
    assert list(names) == ["ALA", "GLY"]


def test_parse_structure_by_chain(protein_pdb):
    [(chain_id, coords, names)] = parse_structure_by_chain(protein_pdb, "pdb", ["A"])
    assert chain_id == "A"
    assert coords.shape == (2, 3)
    assert list(names) == ["ALA", "GLY"]


def test_parse_structure_unknown_format(protein_pdb):
    with pytest.raises(ValueError, match="unknown structure format: 'mmcif'"):
        parse_structure(protein_pdb, "mmcif")

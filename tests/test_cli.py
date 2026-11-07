import subprocess


def test_cli_help():
    result = subprocess.run(["uv", "run", "apholo8-egnn", "--help"], capture_output=True)
    assert result.returncode == 0


def test_folder_dot_writes_default_output_next_to_folder(tmp_path, monkeypatch, protein_pdb):
    import argparse
    from pathlib import Path

    import numpy as np

    from apholo8_egnn.cli.app import _run_folder

    class FakePredictor:
        def predict(self, **kwargs):
            coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
            return coords, np.array([0.9, 0.5])

    structures = tmp_path / "structures"
    structures.mkdir()
    (structures / "1abc.pdb").write_text(protein_pdb)
    monkeypatch.chdir(structures)
    args = argparse.Namespace(
        folder=Path("."),
        output=None,
        cycles=1,
        virtual_nodes=2,
        top=None,
        nms_radius=2.0,
        vn_seed=None,
    )

    _run_folder(FakePredictor(), args)

    output = tmp_path / "structures_predictions.csv"
    lines = output.read_text().splitlines()
    assert lines[0].startswith("structure,vn1_conf")
    assert lines[1].startswith("1abc.pdb,0.900000")

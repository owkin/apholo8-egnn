import pytest

from apholo_egnn.cli.app import main


def test_cli_help_does_not_crash(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["apholo-egnn", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_requires_structure_or_folder(monkeypatch):
    monkeypatch.setattr("sys.argv", ["apholo-egnn"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_cli_structure_and_folder_are_mutually_exclusive(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["apholo-egnn", "--structure", "protein.pdb", "--folder", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2

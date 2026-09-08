import pytest

from apholo_egnn.cli.app import main


def test_cli_help_does_not_crash(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["apholo-egnn", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()

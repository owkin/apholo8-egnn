import subprocess


def test_cli_help():
    result = subprocess.run(["uv", "run", "apholo-egnn", "--help"], capture_output=True)
    assert result.returncode == 0

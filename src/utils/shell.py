"""Utility module for interacting with the shell."""
import shlex
from subprocess import PIPE, Popen, run


def run_shell(cmd: str) -> str:
    """Run a shell command using the subprocess module."""
    return run(shlex.split(cmd), capture_output=True, check=True, text=True)


def is_file_in_use(filename: str) -> bool:
    """Check if a file is being used by other processes."""
    lsout = Popen(shlex.split(f"lsof {filename}"), stdout=PIPE)
    output = run(shlex.split(f"grep {filename}"), stdin=lsout.stdout)
    return not output.returncode

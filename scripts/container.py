"""Utility CLI to run pygfx within an x86 ubuntu container
configured for offscreen software rendering with lavapipe.

Can be considered equivalent to the CI environment.

Run `python scripts/docker.py --help` to get usage help.
"""
import argparse
import platform
from shutil import which
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).parent
ROOT = Path(__file__).parent.parent
use_wsl = False


extra_build_args = extra_run_args = ""
if platform.machine().startswith(("arm", "aarch64")):
    extra_build_args = extra_run_args = "--platform linux/amd64"


def mount_path(path):
    """Convert a path to be absolute and posix, and mapped optionally
    to the WSL guest VM"""
    path = path.resolve()
    if use_wsl:
        path = Path(f"/mnt/{path.drive[:-1].lower()}/", *path.parts[1:])
    return path.as_posix()


def run_subprocess(cmd, *args, **kwargs):
    """subprocess.run wrapper that also echoes the issued command"""
    print(f"+ {cmd}")
    return subprocess.run(cmd, *args, **kwargs)


def get_docker_cli():
    """determine how to call the docker/podman CLI"""
    for tool in ["docker", "podman"]:
        if which(tool):
            return tool
    has_wsl = which("wsl") is not None
    global use_wsl
    for tool in ["docker", "podman"]:
        if has_wsl:
            if (
                subprocess.run(f"wsl which {tool}", stdout=subprocess.PIPE).returncode
                == 0
            ):
                use_wsl = True
                return f"wsl {tool}"
    raise ValueError("You need to have either podman or docker on PATH")


dockerfile = (HERE / "Dockerfile").relative_to(ROOT).as_posix()
docker_cli = get_docker_cli()


def build():
    """Build the docker image"""
    returncode = run_subprocess(
        f"{docker_cli} build {extra_build_args} -t pygfx -f {dockerfile} .",
        shell=True,
        cwd=ROOT,
    ).returncode
    if returncode != 0:
        sys.exit(returncode)


def run(use_volumes, cmd):
    """Run commands against the docker image"""
    cmd = " ".join(cmd)
    paths = ["pygfx", "tests", "examples", "docs", "conftest.py", "setup.cfg"]
    volumes = ""
    if use_volumes:
        volumes = " ".join(
            [f"-v {mount_path(ROOT / path)}:/app/{path}" for path in paths]
        )
    returncode = run_subprocess(
        f"{docker_cli} run --rm -ti {extra_run_args} {volumes} pygfx {cmd}",
        shell=True,
        cwd=ROOT,
    ).returncode
    if returncode != 0:
        sys.exit(returncode)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-b", "--build", action="store_true", help="build the docker image first"
)
parser.add_argument(
    "-v", "--volumes", action="store_true", help="mount repo into container"
)
parser.add_argument(
    "cmd",
    nargs=argparse.REMAINDER,
    help="remainder of the command will be run in the container",
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.build:
        build()
    run(args.volumes, args.cmd)

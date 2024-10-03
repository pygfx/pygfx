"""
Versioning for Pygfx. We use a hard-coded version number, because it's simple
and alwayd works. And for dev installs we add extra versioning info.
"""

import logging
import subprocess
from pathlib import Path


# This is the reference version number, to be bumped before each release.
# The build system (Flit) detects this definition when building a distribution.
__version__ = "0.5.0"


logger = logging.getLogger("pygfx")

# Get whether this is a repo. If so, repo_dir is the path, otherwise repo_dir is None.
repo_dir = Path(__file__).parents[1]
repo_dir = repo_dir if repo_dir.joinpath(".git").is_dir() else None


def get_version():
    """Get the version string."""
    if repo_dir:
        return get_extended_version()
    else:
        return __version__


def get_extended_version():
    """Get an extended version string with information from git."""

    release, post, labels = get_version_info_from_git()

    # Sample first 3 parts of __version__
    base_release = ".".join(__version__.split(".")[:3])

    # Check release
    if not release:
        release = base_release
    elif release != base_release:
        logger.warning("Pygfx version from git and __version__ don't match.")

    # Build the total version
    version = release
    if post and post != "0":
        version += f".post{post}"
    if labels:
        version += "+" + ".".join(labels)

    return version


def get_version_info_from_git():
    """Get (release, post, labels) from Git.

    With `release` the version number from the latest tag, `post` the
    number of commits since that tag, and `labels` a tuple with the
    git-hash and optionally a dirty flag.
    """

    # Call out to Git
    command = [
        "git",
        "describe",
        "--long",
        "--always",
        "--tags",
        "--dirty",
        "--first-parent",
    ]
    p = subprocess.run(
        command, cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    # Parse the result into parts
    output = p.stdout.decode(errors="ignore")
    if p.returncode:
        logger.warning("Could not get pygfx version: " + output)
        parts = (None, None, "unknown")
    else:
        parts = output.strip().lstrip("v").split("-")
        if len(parts) <= 2:
            # No tags (and thus also no post). Only git hash and maybe 'dirty'
            parts = (None, None, *parts)

    # Return unpacked parts
    release, post, *labels = parts
    return release, post, labels


__version__ = get_version()

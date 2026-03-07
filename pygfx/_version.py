"""
_version.py v1.6

Simple version string management, using a hard-coded version string
for simplicity and compatibility, while adding git info at runtime.
See https://github.com/pygfx/_version for more info.
This code is subject to The Unlicense (public domain).

Any updates to this file should be done in https://github.com/pygfx/_version

Usage in short:

* Add this file to the root of your library (next to the `__init__.py`).
* On a new release, you just update the __version__.
"""

# ruff: noqa: RUF100, S310, PLR2004, D212, D400, D415, S603, BLE001, COM812

import logging
import subprocess
from pathlib import Path

# This is the base version number, to be bumped before each release.
# The build system detects this definition when building a distribution.
__version__ = "0.16.0"

# Set this to your library name
project_name = "pygfx"


logger = logging.getLogger(project_name)

# Get whether this is a repo. If so, repo_dir is the path, otherwise None.
# .git is a dir in a normal repo and a file when in a submodule.
repo_dir = Path(__file__).parents[1]
repo_dir = repo_dir if repo_dir.joinpath(".git").exists() else None


def get_version() -> str:
    """Get the version string."""
    try:
        if repo_dir:
            release, post, tag, dirty = get_version_info_from_git()
            result = get_extended_version(release, post, tag, dirty)

            # Warn if release does not match base_version.
            # Can happen between bumping and tagging. And also when merging a
            # version bump into a working branch, because we use --first-parent.
            if release and release != base_version:
                release2, _post, _tag, _dirty = get_version_info_from_git(
                    first_parent=False
                )
                if release2 != base_version:
                    warning(
                        f"{project_name} version from git ({release})"
                        f" and __version__ ({base_version}) don't match."
                    )

            return result

    except Exception as err:
        # Failsafe.
        warning(f"Error getting refined version: {err}")

    return base_version


def get_extended_version(release: str, post: str, tag: str, dirty: str) -> str:
    """Get an extended version string with information from git."""
    # Start version string (__version__ string is leading).
    version = base_version
    labels = []

    if release and release != base_version:
        pre_label = "from_tag_" + release.replace(".", "_")
        labels = [pre_label, f"post{post}", tag, dirty]
    elif post and post != "0":
        version += f".post{post}"
        labels = [tag, dirty]
    elif dirty:
        labels = [tag, dirty]
    else:
        # If not post and not dirty, show 'clean' version without git tag.
        pass

    # Compose final version (remove empty labels, e.g. when not dirty).
    # Everything after the '+' is not sortable (does not get in version_info).
    label_str = ".".join(label for label in labels if label)
    if label_str:
        version += "+" + label_str
    return version


def get_version_info_from_git(
    *, first_parent: bool = True
) -> tuple[str, str, str, str]:
    """
    Get (release, post, tag, dirty) from Git.

    With `release` the version number from the latest tag, `post` the
    number of commits since that tag, `tag` the git hash, and `dirty` a string
    that is either empty or says 'dirty'.
    """
    # Call out to Git.
    command = ["git", "describe", "--long", "--always", "--tags", "--dirty"]
    if first_parent:
        command.append("--first-parent")
    try:
        p = subprocess.run(command, check=False, cwd=repo_dir, capture_output=True)
    except Exception as e:
        warning(f"Could not get {project_name} version: {e}")
        p = None

    # Parse the result into parts.
    if p is None:
        parts = ("", "", "unknown")
    else:
        output = p.stdout.decode(errors="ignore")
        if p.returncode:
            stderr = p.stderr.decode(errors="ignore")
            warning(
                f"Could not get {project_name} version.\n\nstdout: "
                + output
                + "\n\nstderr: "
                + stderr
            )
            parts = ("", "", "unknown")
        else:
            parts = output.strip().lstrip("v").split("-")
            if len(parts) <= 2:
                # No tags (and thus no post). Only git hash and maybe 'dirty'.
                parts = ("", "", *parts)

    # Return unpacked parts.
    release = parts[0]
    post = parts[1]
    tag = parts[2]
    dirty = "dirty" if len(parts) > 3 else ""
    return release, post, tag, dirty


def version_to_tuple(v: str) -> tuple:
    parts = []
    for part in v.split("+", maxsplit=1)[0].split("."):
        if not part:
            pass
        elif part.startswith("post"):
            try:
                parts.extend(["post", int(part[4:])])
            except ValueError:
                parts.append(part)
        else:
            try:
                parts.append(int(part))
            except ValueError:
                parts.append(part)
    return tuple(parts)


def warning(m: str) -> None:
    logger.warning(m)


# Apply the versioning.
base_version = __version__
__version__ = get_version()
version_info = version_to_tuple(__version__)


# The CLI part.

CLI_USAGE = """
_version.py

help            - Show this message.
version         - Show the current version.
bump VERSION    - Bump the __version__ to the given VERSION.
update          - Self-update the _version.py module by downloading the
                  reference code and replacing version number and project name.
""".lstrip()

if __name__ == "__main__":
    import sys
    import urllib.request

    def prnt(m: str) -> None:
        sys.stdout.write(m + "\n")
        sys.stdout.flush()

    _, *args = sys.argv
    this_file = Path(__file__)

    if not args or args[0] == "version":
        prnt(f"{project_name} v{__version__}")

    elif args[0] == "bump":
        if len(args) != 2:
            sys.exit("Expected a version number to bump to.")
        new_version = args[1].lstrip("v")  # allow '1.2.3' and 'v1.2.3'
        if new_version.count(".") != 2:
            sys.exit("Expected two dots in new version string.")
        if not all(s.isnumeric() for s in new_version.split(".")):
            sys.exit("Expected only numbers in new version string.")
        with this_file.open("rb") as f:
            text = ref_text = f.read().decode()
        text = text.replace(base_version, new_version, 1)
        with this_file.open("wb") as f:
            f.write(text.encode())
        prnt(f"Bumped version from '{base_version}' to '{new_version}'.")

    elif args[0] == "update":
        u = "https://raw.githubusercontent.com/pygfx/_version/main/_version.py"
        with urllib.request.urlopen(u) as f:
            text = ref_text = f.read().decode()
        text = text.replace("0.0.0", base_version, 1)
        text = text.replace("PROJECT_NAME", project_name, 1)
        with this_file.open("wb") as f:
            f.write(text.encode())
        prnt("Updated to the latest _version.py.")

    elif args[0].lstrip("-") in ["h", "help"]:
        prnt(CLI_USAGE)

    else:
        prnt(f"Unknown command for _version.py: {args[0]!r}")
        prnt("Use ``python _version.py help`` to see a list of options.")
        sys.exit(1)

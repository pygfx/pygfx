import os
import sys
import atexit
import shutil
import tempfile
import importlib.resources


try:
    HOME = os.path.expanduser("~")
except Exception:  # Exceptions thrown by home() are not specified...
    HOME = "/home"  # Just an arbitrary path


def get_resources_dir():
    """Get the path to the directory of builtin resources."""
    if sys.version_info < (3, 9):
        context = importlib.resources.path("pygfx.data_files", "__init__.py")
    else:
        ref = importlib.resources.files("pygfx.data_files") / "__init__.py"
        context = importlib.resources.as_file(ref)
    with context as path:
        pass
    # Return the dir. We assume that the data files are on a normal dir on the fs.
    return str(path.parent)


def get_cache_dir():
    """Get path were we can store our cache data."""
    return _get_data_dir("cache")


def _get_data_dir(xdg_name):
    # Set by user
    dir = os.getenv("PYGFX_DATA_DIR")
    if dir:
        return os.path.abspath(dir)

    # Get user dir
    user_dir = os.path.expanduser("~")
    if not os.path.isdir(user_dir):
        user_dir = "/var/tmp"

    # Get base cache dir
    if sys.platform.startswith("win"):
        roaming = False
        path1, path2 = os.getenv("LOCALAPPDATA"), os.getenv("APPDATA")
        base_dir = (path2 or path1) if roaming else (path1 or path2)
    elif sys.platform.startswith("darwin"):
        base_dir = os.path.join(user_dir, "Library", "Application Support")
    elif sys.platform.startswith(("linux", "freebsd")):
        # https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
        path1 = os.getenv(f"XDG_{xdg_name.upper()}_HOME")
        path2 = os.path.join(HOME, "." + xdg_name.lower())
        base_dir = path1 or path2

    # Fall back to user dir
    if not (base_dir and os.path.isdir(base_dir)):
        base_dir = user_dir

    # Make directory for pygfx
    dir = os.path.join(base_dir, ".pygfx" if base_dir == user_dir else "pygfx")
    try:
        os.makedirs(dir, exist_ok=True)
        if not (os.access(dir, os.W_OK) and os.path.isdir(dir)):
            raise OSError()
    except OSError:
        # If the config or cache directory cannot be created or is not a writable
        # directory, create a temporary one.
        dir = os.environ["PYGFX_DATA_DIR"] = tempfile.mkdtemp(prefix="pygfx-")
        atexit.register(shutil.rmtree, dir)

    return dir

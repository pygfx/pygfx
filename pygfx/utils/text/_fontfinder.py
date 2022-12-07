"""
Detect fonts on the system.
Some of this code was "inspired" by the Matplotlib font manager.
"""

import os
import sys
import json
import time
import secrets

from .. import logger, get_resources_dir, get_cache_dir


def find_fonts_from_dir(directory, recursive):
    """
    Return a list of all fonts matching any of the extensions, found
    recursively under the directory.
    """
    if not os.path.isdir(directory):
        raise OSError(f"Not a directory: {directory}")
    extensions = ".ttf", ".otf"
    dir_paths = set()
    file_paths = set()
    if not recursive:
        dir_paths.add(directory)
        for fname in os.listdir(directory):
            if fname.lower().endswith(extensions):
                file_paths.add(os.path.join(directory, fname))
    else:
        for dirpath, _, filenames in os.walk(directory):
            dir_paths.add(dirpath)
            for fname in filenames:
                if fname.lower().endswith(extensions):
                    file_paths.add(os.path.join(dirpath, fname))
    return dir_paths, file_paths


def get_builtin_fonts():
    """Get a list of fonts that is shipped with PyGfx."""
    dir_paths, file_paths = find_fonts_from_dir(get_resources_dir(), False)
    return file_paths


def get_entrypoints_fonts():
    """Get a list of fonts defined via the setuptools entrypoint. TODO someday."""
    raise NotImplementedError()


def get_system_fonts():
    """Get a list of system fonts. The list is cached for speed. When
    this function is called, it checks whether any of the directories
    that may contain fonts have changed (files are deleted, added, or
    renamed). If so, we search that directory again. This way we are
    able to detect new fonts without an exaustive search each process
    startup (which can be slow).
    """

    # Load cache from file system
    filename = os.path.join(get_cache_dir(), "font_cache.json")
    try:
        with open(filename, "rt", encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    reset = False
    dirs_to_update = set()
    new_dir_paths = set()
    new_file_paths = set()

    # Check if any of the directories have updated. Also basic sanity checks.
    if cache:
        try:
            for p, info in cache["dirs"].items():
                if not isinstance(p, str):
                    raise RuntimeError("Dir path not a str.")
                elif not isinstance(info, dict):
                    raise RuntimeError("Dir info not a dict.")
                elif not os.path.isdir(p):
                    reset = True
                elif info["mtime"] < os.path.getmtime(p):
                    dirs_to_update.add(p)
            for p, info in cache["files"].items():
                if not isinstance(p, str):
                    raise RuntimeError("File path not a str.")
                elif not isinstance(info, dict):
                    raise RuntimeError("File info not a dict.")
                # elif not os.path.isfile(p):  # slow, and should be covered by dir mtime
                #     dirs_to_update(os.path.dirname(p))
        except Exception as err:
            logger.error("Error loading font cache: %s", err)
            cache = {}

    # Reset directories that need an update
    for dir_path in dirs_to_update:
        # Clear fonts in this dir
        files = cache["files"]
        for p in list(files.keys()):
            if os.path.dirname(p) == dir_path:
                files.pop(p)
        # Detect fonts again, schedule for inclusion in the cache
        _, file_paths = find_fonts_from_dir(dir_path, False)
        new_dir_paths.add(dir_path)
        new_file_paths.update(file_paths)

    # Do a full search if needed
    if reset or not cache:
        new_dir_paths, new_file_paths = find_system_fonts()
        cache = {"dirs": {}, "files": {}}

    # Do we need to update?
    if new_dir_paths or new_file_paths:
        logger.info(
            f"Searched for fonts in: {new_dir_paths}",
        )

        # Put new items in the cache
        dirs = cache["dirs"]
        for p in new_dir_paths:
            info = {"mtime": os.path.getmtime(p)}
            dirs[p.replace("\\", "/")] = info
        files = cache["files"]
        for p in new_file_paths:
            info = {"mtime": os.path.getmtime(p)}
            files[p.replace("\\", "/")] = info

        # Sort the keys, just being clean
        cache = {"dirs": {}, "files": {}}
        for key in sorted(dirs.keys()):
            cache["dirs"][key] = dirs[key]
        for key in sorted(files.keys()):
            cache["files"][key] = files[key]

        # Write to file system, first to a uniquely named file
        filename2 = filename + ".part." + secrets.token_urlsafe(4)
        with open(filename2, "wt", encoding="utf-8") as f:
            json.dump(cache, f)

        # Now we must replace the font cache on disk with the result.
        # But we must take into account that another process might be
        # reading from it. The os.replace() call is atomic. And on Unix
        # that's that - whatever is reading is can keep doing so,
        # because this is simply a new inode with the same name. On
        # Windows, however, the replace will fail if something else has
        # the file open. So we wait and try again, until we time out
        # and give up.
        etime = time.time() + 10
        while time.time() < etime:
            try:
                os.replace(filename2, filename)
            except OSError:
                time.sleep(0.1)
            else:
                break
        else:
            logger.error("Failed to write font cache")
            try:
                os.remove(filename2)
            except Exception:
                pass

    return list(cache["files"].keys())


def find_system_fonts():
    """Find fonts on the file system. Returns (dir_paths, file_paths).
    This function might be relatively slow.
    """
    dirs = get_system_font_directories()
    dir_paths = set()
    file_paths = set()
    for d in dirs:
        new_dirs, new_files = find_fonts_from_dir(d, True)
        dir_paths.update(new_dirs)
        file_paths.update(new_files)

    return dir_paths, file_paths


# %% OS-specific logic


def get_system_font_directories():
    # Simple triage, easy to replace in tests.
    if sys.platform.startswith("win"):
        return get_windows_font_directories()
    elif sys.platform.startswith("darwin"):
        return get_osx_font_directories()
    else:
        return get_unix_font_directories()


def get_windows_font_directories():
    import winreg

    dirs = []
    dirs.extend(WinFontDirs)
    dirs.append(os.path.join(os.environ["WINDIR"], "Fonts"))

    # Get win32 font directory from the registry
    ms_folders = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, ms_folders) as user:
            dirs.append(winreg.QueryValueEx(user, "Fonts")[0])
    except OSError:
        pass

    # filter out duplicates and non-existants
    dirs = {os.path.abspath(d) for d in dirs if os.path.isdir(d)}
    return list(dirs)


def get_osx_font_directories():
    dirs = []
    dirs.extend(X11FontDirectories)
    dirs.extend(OSXFontDirectories)
    return [os.path.abspath(d) for d in dirs if os.path.isdir(d)]


def get_unix_font_directories():
    # MPL also calls out to fontconfig (fc-list) to get a list of fonts.
    # But if we assume that our list of possible directories is
    # complete, this should not be necessary. Let's see how it goes.
    dirs = []
    dirs.extend(X11FontDirectories)
    return [os.path.abspath(d) for d in dirs if os.path.isdir(d)]


try:
    HOME = os.path.expanduser("~")
except Exception:  # Exceptions thrown by home() are not specified...
    HOME = "/home"  # Just an arbitrary path

WinFontDirs = [
    os.path.join(os.environ["LOCALAPPDATA"], "Microsoft/Windows/Fonts"),
    os.path.join(os.environ["APPDATA"], "Microsoft/Windows/Fonts"),
    os.path.join(os.environ["ALLUSERSPROFILE"], "AppData/Local/Microsoft/Windows/Fonts"),
    os.path.join(os.environ["ALLUSERSPROFILE"], "AppData/Roaming/Microsoft/Windows/Fonts"),
]

X11FontDirectories = [
    # an old standard installation point
    "/usr/X11R6/lib/X11/fonts/TTF/",
    "/usr/X11/lib/X11/fonts",
    # here is the new standard location for fonts
    "/usr/share/fonts/",
    # documented as a good place to install new fonts
    "/usr/local/share/fonts/",
    # common application, not really useful
    "/usr/lib/openoffice/share/fonts/truetype/",
    # user fonts
    os.path.join(os.getenv("XDG_DATA_HOME") or HOME, ".local/share/fonts"),
    os.path.join(HOME, ".fonts"),
]

OSXFontDirectories = [
    "/Library/Fonts/",
    "/Network/Library/Fonts/",
    "/System/Library/Fonts/",
    # fonts installed via MacPorts
    "/opt/local/share/fonts",
    # user fonts
    os.path.join(HOME, "Library/Fonts"),
]

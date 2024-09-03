"""
Detect fonts on the system. This code was inspired by the Matplotlib
font manager. It differs in that we rely fully on finding fonts in
certain directories. We do not use the Windows registry or anything
like that to get a list of "official" fonts. The downside is that we
may find more fonts that the OS feels are registered. The big advantage
is that we can check the mtime of a handful of directories to know
whether we need to update our cache (and can also do partial updates).
"""

import os
import sys
import json
import time
import secrets

import freetype

from .. import logger, get_resources_dir, get_cache_dir


# Weight names according to CSS and the OpenType spec.
weight_dict = {
    "thin": 100,
    "hairline": 100,
    "ultralight": 200,
    "extralight": 200,
    "light": 300,
    "normal": 400,
    "regular": 400,
    "medium": 500,
    "semibold": 600,
    "demibold": 600,
    "bold": 700,
    "extrabold": 800,
    "ultrabold": 800,
    "black": 900,
    "heavy": 900,
}


style_dict = {
    "normal": "normal",
    "regular": "normal",
    "italic": "italic",
    "oblique": "oblique",
}


class FontFile:
    """Object to represent a font file."""

    def __init__(self, filename, family=None, variant=None, codepoints=None):
        assert isinstance(filename, str)
        assert family is None or isinstance(family, str)
        assert variant is None or isinstance(variant, str)
        assert codepoints is None or isinstance(codepoints, set)

        self._filename = filename
        self._family = family
        self._variant = variant
        self._name = None
        self._weight = None
        self._style = None
        self._codepoints = codepoints

    def __repr__(self):
        return f"<FontFile {self.name} at 0x{hex(id(self))}>"

    def __hash__(self):
        return hash(self.name)

    def _get_face(self):
        # This was factored out so it can be overloaded in tests
        return freetype.Face(self._filename)

    @property
    def filename(self):
        """The path to this font file."""
        return self._filename

    @property
    def family(self):
        """The family name of this font, e.g. 'Noto Sans' or 'Arial'.
        This value is defined in the font file.
        """
        if not self._family:
            self._family = self._get_face().family_name.decode()
            if not self._family:
                name = os.path.basename(self._filename).split(".")[0]
                family, _, _ = name.partition("-")
                self._family = family or "Unknown"
        return self._family

    @property
    def variant(self):
        """The variant name of this font, e.g. 'Regular', 'Bold',
        'Italic', 'Thin Italic'. This is a value defined in the font
        file, and from this the weight and style are derived.
        """
        if not self._variant:
            self._variant = self._get_face().style_name.decode()
            if not self._variant:
                name = os.path.basename(self._filename).split(".")[0]
                _, _, variant = name.partition("-")
                self._variant = variant or "Regular"
        return self._variant

    @property
    def weight(self):
        """The font weight, as a number between 100-900."""
        if not self._weight:
            variant_name = self._variant.lower()
            if variant_name == "regular":  # make common cases fast
                self._weight = 400
            else:
                for weight_name, weight_nr in weight_dict.items():
                    if weight_name in variant_name:
                        self._weight = weight_nr
                        break
                else:
                    self._weight = 400
        return self._weight

    @property
    def style(self):
        """The style, one of "normal", "italic", or "oblique"."""
        if not self._style:
            variant_name = self._variant.lower()
            for style_name1, style_name2 in style_dict.items():
                if style_name1 in variant_name:
                    self._style = style_name2
                    break
            else:
                self._style = "normal"
        return self._style

    @property
    def name(self):
        """A normalized name that includes the family and variant. This
        therefore uniquely identifies the font. This typically is the
        same as the filename (without extension), but this is not
        guaranteed to be the case.
        """
        if not self._name:
            family = "".join(x[0].upper() + x[1:] for x in self.family.split())
            style = "".join(x[0].upper() + x[1:] for x in self.variant.split())
            self._name = family + "-" + style
        return self._name

    @property
    def codepoints(self):
        """A set of Unicode code points (ints) supported by this font.
        To test whether a certain codepoint is supported, use has_codepoint() instead.
        """
        # note: we could use a data structure that stores codepoints more efficiently
        if self._codepoints is None:
            self._codepoints = set(i for i, _ in self._get_face().get_chars())
        return self._codepoints

    def has_codepoint(self, codepoint):
        """Check whether a codepoint is supported by this font."""
        return codepoint in self.codepoints


def get_all_fonts():
    """Get a set of all available fonts."""
    # Disabling system fonts can be very helpful in testing to
    # ensure that we get repeatable results nomatter the version of the
    # fonts installed on the machine.
    disable_system_fonts = os.environ.get("PYGFX_DISABLE_SYSTEM_FONTS", "0").lower()
    if disable_system_fonts in ("1", "true", "yes"):
        return get_builtin_fonts()
    else:
        return get_builtin_fonts() | get_system_fonts()


def get_builtin_fonts():
    """Get a list of fonts that are shipped with pygfx."""
    dir_paths, file_paths = find_fonts_paths(get_resources_dir(), False)
    return {FontFile(p) for p in file_paths}


def get_entrypoints_fonts():
    """Get a list of fonts defined via the setuptools entrypoint. TODO someday."""
    raise NotImplementedError()


def get_system_fonts():
    """Get a list of system fonts. The list is cached for speed. When
    this function is called, it checks whether any of the directories
    that may contain fonts have changed (files are deleted, added, or
    renamed). If so, we search that directory again. This way we are
    able to detect new fonts without an exhaustive search each process
    startup (which would be slow).
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
                elif {"mtime", "family", "variant"}.difference(info.keys()):
                    reset = True  # We probably added new info to the cache
                # elif not os.path.isfile(p):  # slow, and should be covered by dir mtime
                #     dirs_to_update(os.path.dirname(p))
        except Exception as err:
            logger.error("Error loading font cache: %s", err)
            cache = {}

    # Reset directories that need an update
    for dir_path in dirs_to_update:
        dir_path = os.path.normpath(dir_path)
        # Clear fonts in this dir
        files = cache["files"]
        for p in list(files.keys()):
            if os.path.normpath(os.path.dirname(p)) == dir_path:
                files.pop(p)
        # Detect fonts again, schedule for inclusion in the cache
        _, file_paths = find_fonts_paths(dir_path, False)
        new_dir_paths.add(dir_path)
        new_file_paths.update(file_paths)

    # Do a full search if needed
    if reset or not cache:
        new_dir_paths, new_file_paths = find_system_fonts()
        cache = {"dirs": {}, "files": {}}

    # Do we need to update?
    if new_dir_paths or new_file_paths:
        logger.info(f"Searched for fonts in: {new_dir_paths}")

        # Put new dirs in the cache
        dirs = cache["dirs"]
        for p in new_dir_paths:
            dirs[p.replace("\\", "/")] = {
                "mtime": os.path.getmtime(p),
            }

        # Put new files in the cache
        files = cache["files"]
        for p in new_file_paths:
            ff = FontFile(p)
            try:
                ff.name  # This makes FreeType open the file
            except Exception:
                continue
            files[p.replace("\\", "/")] = {
                "mtime": os.path.getmtime(ff.filename),
                "family": ff.family,
                "variant": ff.variant,
            }

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
        etime = time.time() + 5
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

    # Return set of FontFile objects
    return {
        FontFile(filename, info["family"], info["variant"])
        for filename, info in cache["files"].items()
    }


def find_system_fonts():
    """Find fonts on the file system. Returns (dir_paths, file_paths).
    This function might be relatively slow.
    """
    dirs = get_system_font_directories()
    dir_paths = set()
    file_paths = set()
    for d in dirs:
        new_dirs, new_files = find_fonts_paths(d, True)
        dir_paths.update(new_dirs)
        file_paths.update(new_files)
    return dir_paths, file_paths


def find_fonts_paths(directory, recursive):
    """Return two sets (dir_paths, file_paths) representing the dirs and
    fonts matching any of the extensions, found in the given directory.
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


# %% OS-specific logic


def get_system_font_directories():
    """Get a set of system font directories."""
    # Simple triage, easy to replace in tests.
    if sys.platform.startswith("win"):
        return get_windows_font_directories()
    elif sys.platform.startswith("darwin"):
        return get_osx_font_directories()
    else:
        return get_unix_font_directories()


def get_windows_font_directories():
    import winreg

    dirs = set()
    dirs.update(WinFontDirs)
    dirs.add(os.path.join(os.getenv("WINDIR", ""), "Fonts"))

    # Get win32 font directory from the registry
    ms_folders = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, ms_folders) as user:
            dirs.add(winreg.QueryValueEx(user, "Fonts")[0])
    except OSError:
        pass

    return {os.path.abspath(d) for d in dirs if os.path.isdir(d)}


def get_osx_font_directories():
    dirs = set()
    dirs.update(X11FontDirectories)
    dirs.update(OSXFontDirectories)
    return {os.path.abspath(d) for d in dirs if os.path.isdir(d)}


def get_unix_font_directories():
    # MPL also calls out to fontconfig (fc-list) to get a list of fonts.
    # But if we assume that our list of possible directories is
    # complete, this should not be necessary. Let's see how it goes.
    dirs = set()
    dirs.update(X11FontDirectories)
    return {os.path.abspath(d) for d in dirs if os.path.isdir(d)}


try:
    HOME = os.path.expanduser("~")
except Exception:  # Exceptions thrown by home() are not specified...
    HOME = "/home"  # Just an arbitrary path

WinFontDirs = [
    os.path.join(os.getenv("LOCALAPPDATA", ""), "Microsoft/Windows/Fonts"),
    os.path.join(os.getenv("APPDATA", ""), "Microsoft/Windows/Fonts"),
    os.path.join(
        os.getenv("ALLUSERSPROFILE", ""), "AppData/Local/Microsoft/Windows/Fonts"
    ),
    os.path.join(
        os.getenv("ALLUSERSPROFILE", ""), "AppData/Roaming/Microsoft/Windows/Fonts"
    ),
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

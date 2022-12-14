import os
import time
import shutil
import tempfile

from pytest import raises

from pygfx.utils.text import _fontfinder


def test_find_fonts_paths():

    # Prepare a clean temp dir
    tmpdir = os.path.join(tempfile.gettempdir(), "pygfx_test")
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)

    # Put a directory structure with stub files in place
    files = [
        "/aa.ttf",
        "/bb.ttf",
        "/cc.otf",
        "/dd.png",
        "/sub/ee.ttf",
        "/sub/ff.ttf",
        "/sub/gg.otf",
        "/sub/hh.png",
        "/sub/deeper/ii.ttf",
        "/sub/deeper/jj.ttf",
        "/sub/deeper/kk.otf",
        "/sub/deeper/ll.png",
    ]
    for fname in files:
        filename = tmpdir + fname
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb"):
            pass

    try:
        # Test non-recursive
        dirs, files = _fontfinder.find_fonts_paths(tmpdir, False)
        assert isinstance(dirs, set)
        assert isinstance(files, set)
        assert dirs == {tmpdir}
        files = set(p[len(tmpdir) + 1 : -4] for p in files)
        assert files == {"aa", "bb", "cc"}

        # Again but deeper
        dirs, files = _fontfinder.find_fonts_paths(tmpdir + "/sub", False)
        assert dirs == {tmpdir + "/sub"}
        files = set(p[len(tmpdir) + 1 : -4] for p in files)
        assert files == {"sub/ee", "sub/ff", "sub/gg"}

        # Recursive
        dirs, files = _fontfinder.find_fonts_paths(tmpdir, True)
        assert isinstance(dirs, set)
        assert isinstance(files, set)
        assert dirs == {tmpdir, tmpdir + "/sub", tmpdir + "/sub/deeper"}
        files = set(p[len(tmpdir) + 1 : -4] for p in files)
        assert files == {
            "aa",
            "bb",
            "cc",
            "sub/ee",
            "sub/ff",
            "sub/gg",
            "sub/deeper/ii",
            "sub/deeper/jj",
            "sub/deeper/kk",
        }

        # Recursive
        dirs, files = _fontfinder.find_fonts_paths(tmpdir + "/sub", True)
        assert dirs == {tmpdir + "/sub", tmpdir + "/sub/deeper"}
        files = set(p[len(tmpdir) + 1 : -4] for p in files)
        assert files == {
            "sub/ee",
            "sub/ff",
            "sub/gg",
            "sub/deeper/ii",
            "sub/deeper/jj",
            "sub/deeper/kk",
        }

        # Not a directory
        with raises(OSError):
            _fontfinder.find_fonts_paths(tmpdir + "/nope", False)
        with raises(OSError):
            _fontfinder.find_fonts_paths(tmpdir + "/nope", True)

    finally:
        # Clean up
        shutil.rmtree(tmpdir, ignore_errors=True)


class StubFace:
    def __init__(self, filename):
        self._filename = filename

    @property
    def family_name(self):
        return b""  # The FontFile will construct a family name from the filename

    @property
    def style_name(self):
        if "broken" in self._filename:
            raise RuntimeError()
        return b""

    def get_chars(self):
        return []


def test_get_system_fonts():

    # Prepare a clean temp dir
    tmpdir = os.path.join(tempfile.gettempdir(), "pygfx_test")
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)

    # Put a directory structure with stub files in place
    files = [
        "/d1/aa.ttf",
        "/d1/bb.ttf",
        "/d1/cc.otf",
        "/d1/dd.png",
        "/d1/broken.ttf",  # Our stubFace will fail on this one
        "/d1/sub/hh.png",
        "/d2/ii.ttf",
        "/d2/jj.ttf",
        "/d2/kk.otf",
        "/d2/ll.png",
    ]
    for fname in files:
        filename = tmpdir + fname
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb"):
            pass

    # Monkey-patch the function that returns platform-specific dirs
    counter = []
    sysdirs = [tmpdir + "/d1", tmpdir + "/d2"]
    _fontfinder.ori_get_system_font_directories = (
        _fontfinder.get_system_font_directories
    )
    _fontfinder.get_system_font_directories = lambda: counter.append("") or sysdirs

    # Monkey_patch FontFile.
    ori_get_face = _fontfinder.FontFile._get_face
    _fontfinder.FontFile._get_face = lambda self: StubFace(self.filename)

    # Make the cache be stored in this dir as well
    os.environ["PYGFX_DATA_DIR"] = tmpdir
    cache_filename = tmpdir + "/font_cache.json"

    try:

        # No cache
        assert len(counter) == 0
        assert not os.path.isfile(cache_filename)

        # Get fonts
        files = _fontfinder.get_system_fonts()
        assert isinstance(files, set)
        initial_files = {p.family for p in files}
        assert initial_files == {"aa", "bb", "cc", "ii", "jj", "kk"}

        # Had to query them
        assert len(counter) == 1

        # Now there is a cache
        assert os.path.isfile(cache_filename)

        # And it contains all files
        with open(cache_filename, "rt", encoding="utf-8") as f:
            cache_text = f.read()
        for p in files:
            assert p.filename in cache_text

        # Do it again
        files = {p.family for p in _fontfinder.get_system_fonts()}
        assert files == initial_files
        # No need to query!
        assert len(counter) == 1

        # Break the file
        with open(cache_filename, "wb") as f:
            f.write(b"not valid json")

        # No worries, will re-create
        files = {p.family for p in _fontfinder.get_system_fonts()}
        assert files == initial_files

        # Had to query them, of course
        assert len(counter) == 2

        # Yes, the cache works
        files = {p.family for p in _fontfinder.get_system_fonts()}
        assert files == initial_files
        # No need to query!
        assert len(counter) == 2

        # Add a font file. This will update the mtime of the directory, triggering
        # a call to find_fonts_paths on that dir, and thus finding the new font
        with open(tmpdir + "/d1/sub/ee.ttf", "wb"):
            pass
        time.sleep(0.1)

        # So now if we get the fonts ...
        files = {p.family for p in _fontfinder.get_system_fonts()}
        assert files != initial_files
        assert files.difference(initial_files).pop() == "ee"

    finally:
        # Clean up
        _fontfinder.FontFile._get_face = ori_get_face
        _fontfinder.get_system_font_directories = (
            _fontfinder.ori_get_system_font_directories
        )
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")

import os
import json
import time
import shutil
import tempfile
from pathlib import Path

from pytest import raises

from pygfx.utils.text import _fontfinder


def test_font_file():
    ff1 = _fontfinder.FontFile("x", "Foo Sans", "Regular", {1, 2, 3})
    assert ff1.filename == "x"
    assert ff1.name == "FooSans-Regular"
    assert ff1.family == "Foo Sans"
    assert ff1.variant == "Regular"
    assert ff1.weight == 400
    assert ff1.style == "normal"
    assert ff1.codepoints.intersection((2, 3, 4, 5)) == {2, 3}

    ff2 = _fontfinder.FontFile("x", "Foo Sans", "Bold", {1, 2, 3})
    assert ff2.name == "FooSans-Bold"
    assert ff2.family == "Foo Sans"
    assert ff2.variant == "Bold"
    assert ff2.weight == 700
    assert ff2.style == "normal"

    ff3 = _fontfinder.FontFile("x", "Foo Sans", "Bold Italic", {1, 2, 3})
    assert ff3.name == "FooSans-BoldItalic"
    assert ff3.family == "Foo Sans"
    assert ff3.variant == "Bold Italic"
    assert ff3.weight == 700
    assert ff3.style == "italic"

    assert hash(ff1) != hash(ff2)
    assert hash(ff1) != hash(ff3)
    assert hash(ff2) != hash(ff3)


def test_find_fonts_paths(request):
    # Prepare a clean temp dir
    tmpdir = Path(tempfile.gettempdir()) / "pygfx_test"
    shutil.rmtree(tmpdir, ignore_errors=True)
    request.addfinalizer(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    # Put a directory structure with stub files in place
    tmpdir.mkdir()
    files = [
        Path("aa.ttf"),
        Path("bb.ttf"),
        Path("cc.otf"),
        Path("dd.png"),
        Path("sub") / "ee.ttf",
        Path("sub") / "ff.ttf",
        Path("sub") / "gg.otf",
        Path("sub") / "hh.png",
        Path("sub") / "deeper" / "ii.ttf",
        Path("sub") / "deeper" / "jj.ttf",
        Path("sub") / "deeper" / "kk.otf",
        Path("sub") / "deeper" / "ll.png",
    ]
    for fname in files:
        filename = tmpdir / fname
        filename.parent.mkdir(exist_ok=True)
        filename.touch()

    # Test non-recursive
    dirs, files = _fontfinder.find_fonts_paths(tmpdir, False)
    assert isinstance(dirs, set)
    assert isinstance(files, set)
    assert dirs == {tmpdir}
    files = set(Path(p).stem for p in files)
    assert files == {"aa", "bb", "cc"}

    # Again but deeper
    root = tmpdir / "sub"
    dirs, files = _fontfinder.find_fonts_paths(root, False)
    assert dirs == {root}
    files = set(
        str(Path(p).relative_to(tmpdir).with_suffix("").as_posix()) for p in files
    )
    assert files == {"sub/ee", "sub/ff", "sub/gg"}

    # Recursive
    dirs, files = _fontfinder.find_fonts_paths(tmpdir, True)
    assert isinstance(dirs, set)
    assert isinstance(files, set)
    dirs = set(Path(p) for p in dirs)
    assert dirs == {tmpdir, tmpdir / "sub", tmpdir / "sub" / "deeper"}
    files = set(
        str(Path(p).relative_to(tmpdir).with_suffix("").as_posix()) for p in files
    )
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
    root = tmpdir / "sub"
    dirs, files = _fontfinder.find_fonts_paths(root, True)
    dirs = set(Path(p) for p in dirs)
    assert dirs == {root, root / "deeper"}
    files = set(
        str(Path(p).relative_to(tmpdir).with_suffix("").as_posix()) for p in files
    )
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
        _fontfinder.find_fonts_paths(tmpdir / "nope", False)
    with raises(OSError):
        _fontfinder.find_fonts_paths(tmpdir / "nope", True)


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

        # Break the file again, but in a more subtle way
        with open(cache_filename, "rt", encoding="utf-8") as f:
            cache = json.load(f)
        cache["files"]["not_a_path"] = 42  # not a dict
        with open(cache_filename, "wt", encoding="utf-8") as f:
            json.dump(cache, f)

        # No worries, will re-create
        files = {p.family for p in _fontfinder.get_system_fonts()}
        assert files == initial_files
        # Had to query them, of course
        assert len(counter) == 3

        # Yes, the cache works
        files = {p.family for p in _fontfinder.get_system_fonts()}
        assert files == initial_files
        # No need to query!
        assert len(counter) == 3

        # Add a font file. This will update the mtime of the directory, triggering
        # a call to find_fonts_paths on that dir, and thus finding the new font
        time.sleep(0.2)
        with open(tmpdir + "/d1/sub/ee.ttf", "wb"):
            pass

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
        os.environ["PYGFX_DATA_DIR"] = ""


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")

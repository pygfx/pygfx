import sys
import functools
import importlib.resources


@functools.lru_cache(maxsize=None)
def load_wgsl(shader_name):
    """Load wgsl code from the file system."""
    package_name = "pygfx.renderers.wgpu.wgsl"
    if sys.version_info < (3, 9):
        context = importlib.resources.path(package_name, shader_name)
    else:
        ref = importlib.resources.files(package_name) / shader_name
        context = importlib.resources.as_file(ref)
    with context as path:
        with open(path, "rb") as f:
            return f.read().decode()

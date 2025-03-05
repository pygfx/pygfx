"""
Test suite utilities.
"""

import re
import ast
from pathlib import Path

import pygfx as gfx

ROOT = Path(__file__).parents[2]  # repo root
examples_dir = ROOT / "examples"
screenshots_dir = examples_dir / "screenshots"
diffs_dir = screenshots_dir / "diffs"

# Initialize the device, to avoid Rust warnings from showing in the first example
adapter = gfx.renderers.wgpu.get_shared().adapter
is_lavapipe = adapter.info["vendor"] == "llvmpipe"


def find_examples():
    """Return a list of (filename, fname, config)."""
    examples = []
    for filename in examples_dir.glob("**/*.py"):
        if any(x in filename.parts for x in ["tests", "data", "screenshots"]):
            continue
        fname = str(filename.relative_to(examples_dir))
        example_code = filename.read_text(encoding="UTF-8")
        config_test = get_example_test_config(fname, example_code)
        config_docs = get_example_docs_config(fname, example_code)
        examples.append((filename, fname, config_test, config_docs))
    return examples


gallery_comment_pattern = re.compile(r"^# *sphinx_gallery_pygfx_docs *\=(.+)$", re.M)
test_comment_pattern = re.compile(r"^# *sphinx_gallery_pygfx_test *\=(.+)$", re.M)


def get_example_docs_config(fname, example_code):
    match = gallery_comment_pattern.search(example_code)
    config = None
    if match:
        config_s = match.group(1).strip()
        try:
            config = ast.literal_eval(config_s)
        except Exception:
            raise RuntimeError(
                f"In '{fname}' the sphinx_gallery_pygfx_docs value is not valid Python: {config_s}"
            ) from None
    return config


def get_example_test_config(fname, example_code):
    match = test_comment_pattern.search(example_code)
    config = None
    if match:
        config_s = match.group(1).strip()
        try:
            config = ast.literal_eval(config_s)
        except Exception:
            raise RuntimeError(
                f"In '{fname}' the sphinx_gallery_pygfx_test value is not valid Python: {config_s}"
            ) from None
    return config

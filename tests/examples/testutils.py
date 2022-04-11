"""
Test suite utilities.
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parent.parent.parent  # repo root
examples_dir = ROOT / "examples"
screenshots_dir = examples_dir / "screenshots"
diffs_dir = screenshots_dir / "diffs"


def wgpu_backend_endswith(query):
    """
    Query the configured wgpu backend driver.
    """
    code = "import wgpu.utils; d = wgpu.utils.get_default_device(); print(d.adapter.properties['adapterType'], d.adapter.properties['backendType'])"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=ROOT,
    )
    return (
        result.stdout.strip().endswith(query)
        and "traceback" not in result.stderr.lower()
    )


is_lavapipe = wgpu_backend_endswith("CPU Vulkan")


def find_examples(query=None, negative_query=None, return_stems=False):
    result = []
    for example_path in examples_dir.glob("*.py"):
        example_code = example_path.read_text()
        query_match = query is None or query in example_code
        negative_query_match = (
            negative_query is None or negative_query not in example_code
        )
        if query_match and negative_query_match:
            if return_stems:
                example_return = example_path.stem
            else:
                example_return = example_path
            result.append(example_return)
    return result

import sys
import subprocess


def _determine_can_use_wgpu_lib():
    # For some reason, since wgpu-native 5c304b5ea1b933574edb52d5de2d49ea04a053db
    # the process' exit code is not zero, so we test more pragmatically.
    code = "import wgpu.utils; wgpu.utils.get_default_device(); print('ok')"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    print("_determine_can_use_wgpu_lib() status code:", result.returncode)
    return (
        result.stdout.strip().endswith("ok")
        and "traceback" not in result.stderr.lower()
    )


can_use_wgpu_lib = _determine_can_use_wgpu_lib()

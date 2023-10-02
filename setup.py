import re

from setuptools import find_packages, setup


NAME = "pygfx"
SUMMARY = "A threejs-like render engine based on wgpu"

with open(f"{NAME}/__init__.py", "rb") as fh:
    init_text = fh.read().decode()
    VERSION = re.search(r"__version__ = \"(.*?)\"", init_text).group(1)
    match = re.search(r"__wgpu_version_range__ = \"(.*?)\", \"(.*?)\"", init_text)
    wgpu_min_ver, wgpu_max_ver = match.group(1), match.group(2)
    match = re.search(r"__pylinalg_version_range__ = \"(.*?)\", \"(.*?)\"", init_text)
    pylinalg_min_ver, pylinalg_max_ver = match.group(1), match.group(2)


runtime_deps = [
    "numpy",
    f"wgpu>={wgpu_min_ver},<{wgpu_max_ver}",
    f"pylinalg>={pylinalg_min_ver},<{pylinalg_max_ver}",
    "freetype-py",
    "uharfbuzz",
    "Jinja2",
]

extras_require = {
    "dev": [
        "black",
        "flake8",
        "flake8-black",
        "pep8-naming",
        "pytest",
        "setuptools",
        "wheel",
        "twine",
        "imageio",
        "psutil",
        "pyinstaller>=4",
    ],
    "examples": [
        "pytest",
        "PySide6-Essentials",
        "imageio",
        "imageio-ffmpeg>=0.4.7",
        "scikit-image",
        "trimesh",
    ],
    "docs": [
        "sphinx>7.2",
        "numpy",
        "wgpu",
        "jinja2",
        "sphinx-gallery",
        "imageio",
        "matplotlib",
        # duplicate example dependencies to avoid pyside6
        "pytest",
        "imageio",
        "imageio-ffmpeg>=0.4.7",
        "scikit-image",
        "trimesh",
    ],
}

resources_globs = ["*.ttf", "*.otf", "*.json"]

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(
        exclude=["tests", "tests.*", "examples", "examples.*", "exp", "exp.*"]
    ),
    package_data={f"{NAME}.data_files": resources_globs},
    python_requires=">=3.8.0",
    install_requires=runtime_deps,
    extras_require=extras_require,
    license="BSD 2-Clause",
    description=SUMMARY,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Almar Klein",
    author_email="almar.klein@gmail.com",
    url="https://github.com/pygfx/pygfx",
    data_files=[("", ["LICENSE"])],
    zip_safe=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    entry_points={
        "pyinstaller40": [
            "hook-dirs = pygfx.__pyinstaller:get_hook_dirs",
            "tests = pygfx.__pyinstaller:get_test_dirs",
        ],
    },
)

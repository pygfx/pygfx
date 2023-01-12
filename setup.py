import re

from setuptools import find_packages, setup


NAME = "pygfx"
SUMMARY = "A threejs-like render engine based on wgpu"

with open(f"{NAME}/__init__.py") as fh:
    VERSION = re.search(r"__version__ = \"(.*?)\"", fh.read()).group(1)


runtime_deps = [
    "numpy",
    "wgpu>=0.8.4,<0.9.0",
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
        "sphinx",
        "numpy",
        "wgpu",
        "jinja2",
        "sphinx-gallery",
        "imageio",
        "matplotlib",
    ],
}

resources_globs = ["*.ttf", "*.otf", "*.json"]

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(
        exclude=["tests", "tests.*", "examples", "examples.*", "exp", "exp.*"]
    ),
    package_data={f"{NAME}.pkg_resources": resources_globs},
    python_requires=">=3.7.0",
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
)

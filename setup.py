import re

from setuptools import find_packages, setup


NAME = "visvis2"
SUMMARY = "Experimental succesor to visvis"

with open(f"{NAME}/__init__.py") as fh:
    VERSION = re.search(r"__version__ = \"(.*?)\"", fh.read()).group(1)


runtime_deps = [
    "python_shader",
    "wgpu",
    "numpy",
]


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(
        exclude=["tests", "tests.*", "examples", "examples.*", "exp", "exp.*"]
    ),
    python_requires=">=3.6.0",
    install_requires=runtime_deps,
    license="BSD 2-Clause",
    description=SUMMARY,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Almar Klein",
    author_email="almar.klein@gmail.com",
    url="https://github.com/almarklein/visvis2",
    data_files=[("", ["LICENSE"])],
)

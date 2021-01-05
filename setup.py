#!/usr/bin/env python

"""Setup pytorch-boilerplate."""

from itertools import dropwhile
from os import path
from setuptools import find_packages, setup


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("pbp", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def setup_package():
    with open("README.md") as f:
        long_description = f.read()
    meta = collect_metadata()
    setup(
        name="pytorch-boilerplate",
        version=meta["version"],
        description=meta["description"],
        long_description=long_description,
        long_description_content_type="text/x-markdown",
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
        ],
        packages=find_packages(exclude=["docs", "tests", "scripts", "examples"])
    )


if __name__ == "__main__":
    setup_package()

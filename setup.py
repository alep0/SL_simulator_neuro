"""
setup.py
========
Build script for the C++ pybind11 extension ``stuart_landau_simulator``.

Build instructions
------------------
    pip install pybind11
    python setup.py build_ext --inplace

Or as an editable install:
    pip install -e .

The module falls back to a pure-Python implementation when the compiled
extension is unavailable (see ``source/core/simulation_engine.py``).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# ---------------------------------------------------------------------------
# pybind11 include path (resolved lazily so the package needn't be installed
# before setup.py is invoked)
# ---------------------------------------------------------------------------
class _Pybind11Include:
    """Defer pybind11.get_include() until the build phase."""

    def __str__(self) -> str:
        import pybind11  # type: ignore
        return pybind11.get_include()


# ---------------------------------------------------------------------------
# Compiler-flag helpers
# ---------------------------------------------------------------------------
def _has_flag(compiler, flag: str) -> bool:
    """Return True if *compiler* accepts *flag*."""
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("int main(int, char**) { return 0; }")
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flag])
    except Exception:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def _build_compile_args() -> tuple[list[str], list[str]]:
    """Return (extra_compile_args, extra_link_args) for the current platform."""
    if sys.platform == "win32":
        compile_args = ["/O2", "/openmp", "/std:c++17", "/fp:fast", "/EHsc"]
        link_args    = []

    elif sys.platform == "darwin":
        compile_args = ["-O3", "-std=c++17", "-Xpreprocessor",
                        "-fopenmp", "-ffast-math", "-march=native"]
        # Locate libomp installed via Homebrew
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"], text=True
            ).strip()
            compile_args += [f"-I{prefix}/include"]
            link_args = ["-lomp", f"-L{prefix}/lib"]
        except (FileNotFoundError, subprocess.CalledProcessError):
            link_args = ["-lomp"]

    else:  # Linux / other POSIX
        compile_args = ["-O3", "-std=c++17", "-fopenmp",
                        "-ffast-math", "-march=native"]
        link_args    = ["-fopenmp"]

    return compile_args, link_args


# ---------------------------------------------------------------------------
# Custom build_ext to inject compiler-type flags
# ---------------------------------------------------------------------------
class _BuildExt(build_ext):
    """Add visibility and exception-handling flags per compiler type."""

    _UNIX_FLAGS = {
        "darwin": ["-stdlib=libc++", "-mmacosx-version-min=10.14"],
        "other":  [],
    }

    def build_extensions(self) -> None:
        ct = self.compiler.compiler_type
        opts:   list[str] = []
        lopts:  list[str] = []

        if ct == "unix":
            plat_flags = self._UNIX_FLAGS.get(
                sys.platform, self._UNIX_FLAGS["other"]
            )
            opts += plat_flags
            lopts += plat_flags
            ver = f'-DVERSION_INFO="{self.distribution.get_version()}"'
            opts.append(ver)
            if _has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append(f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"')

        for ext in self.extensions:
            ext.extra_compile_args = opts + ext.extra_compile_args
            ext.extra_link_args    = lopts + ext.extra_link_args

        build_ext.build_extensions(self)


# ---------------------------------------------------------------------------
# Extension definition
# ---------------------------------------------------------------------------
_compile_args, _link_args = _build_compile_args()

ext_modules = [
    Extension(
        "stuart_landau_simulator",              # importable name
        sources=["source/core/stuart_landau_simulator.cpp"],
        include_dirs=[_Pybind11Include()],
        language="c++",
        extra_compile_args=_compile_args,
        extra_link_args=_link_args,
    )
]

# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------
long_description = Path("docs/README.md").read_text() if Path("docs/README.md").exists() else ""

setup(
    name="neuro-sl-simulator",
    version="2.0.0",
    author="Alejandro Aguado",
    description="C++ accelerated Stuart-Landau neural network simulator for resting-state fMRI modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": _BuildExt},
    packages=["source", "source.core", "source.analysis", "source.utils"],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "pybind11>=2.10",
    ],
    extras_require={
        "dev": ["pytest>=7", "pytest-cov", "black", "ruff"],
    },
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
    keywords="neural-network simulation stuart-landau fMRI functional-connectivity",
)

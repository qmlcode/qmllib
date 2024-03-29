from setuptools import setup

try:
    import _compile
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))
    import _compile

if __name__ == "__main__":
    _compile.main()
    setup(
        description="Python/Fortran toolkit for representation of molecules and solids for machine learning of properties of molecules and solids.",
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Chemistry",
        ],
        keywords=["qml", "quantum chemistry", "machine learning"],
    )

[![Test Ubuntu](https://github.com/qmlcode/qmllib/actions/workflows/test.ubuntu.yml/badge.svg)](https://github.com/qmlcode/qmllib/actions/workflows/test.ubuntu.yml)
[![Test MacOS](https://github.com/qmlcode/qmllib/actions/workflows/test.macos.yml/badge.svg)](https://github.com/qmlcode/qmllib/actions/workflows/test.macos.yml)
[![PyPI version](https://img.shields.io/pypi/v/qmllib)](https://pypi.org/project/qmllib/)
[![Python Versions](https://img.shields.io/pypi/pyversions/qmllib?logo=python&logoColor=white)](https://pypi.org/project/qmllib/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)](https://github.com/qmlcode/qmllib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is qmllib?

`qmllib` is a Python/Fortran toolkit for representation of molecules and solids for machine learning of properties of molecules and solids. The library is not a high-level framework where you can do `model.train()`, but supplies the building blocks to carry out efficient and accurate machine learning. As such, the goal is to provide usable and efficient implementations of concepts such as representations and kernels.

## QML or qmllib?

`qmllib` represents the core library functionality derived from the original QML package, providing a powerful toolkit for quantum machine learning applications, but without the high-level abstraction, for example SKLearn.

This package is and should stay free-function design oriented.

If you are moving from `qml` to `qmllib`, note that there are breaking changes to the interface to make it more consistent with both argument orders and function naming.

## How to install

Install from PyPI — pre-built wheels are available for Linux and macOS. They are pre-compiled with optimized BLAS libraries and OpenMP support.

For most users, you can just install with pip:
```bash
pip install qmllib
```
This installs pre-compiled wheels with optimized BLAS libraries:
- **Linux**: OpenBLAS
- **macOS**: Apple Accelerate framework


## Installing from source

If you are installing from source (e.g. directly from GitHub), you will need a Fortran compiler, OpenMP and a BLAS library. On Linux:

```bash
sudo apt install gfortran libomp-dev libopenblas-dev
```

On macOS via Homebrew:

```bash
brew install gcc libomp llvm 
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/qmlcode/qmllib
```

Or a specific branch:

```bash
pip install git+https://github.com/qmlcode/qmllib@feature_branch
```

## How to contribute

[uv](https://docs.astral.sh/uv/) is required for the development workflow.

Fork and clone the repo, then set up the environment and run the tests:

```bash
git clone your_repo qmllib.git
cd qmllib.git
make install-dev
make test
```
Fork it, clone it, make it, test it!

## How to use

Notebook examples are coming. For now, see test files in `tests/*`.

## How to cite

Please cite the representation that you are using accordingly.

- **Implementation**

  qmllib: A Python Toolkit for Quantum Chemistry Machine Learning,
  https://github.com/qmlcode/qmllib, \<version or git commit\>

- **FCHL19** `generate_fchl19`

  FCHL revisited: Faster and more accurate quantum machine learning,
  Christensen, Bratholm, Faber, Lilienfeld,
  J. Chem. Phys. 152, 044107 (2020),
  https://doi.org/10.1063/1.5126701

- **FCHL18** `generate_fchl18`

  Alchemical and structural distribution based representation for universal quantum machine learning,
  Faber, Christensen, Huang, Lilienfeld,
  J. Chem. Phys. 148, 241717 (2018),
  https://doi.org/10.1063/1.5020710

- **Coulomb Matrix** `generate_coulomb_matrix_*`

  Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning,
  Rupp, Tkatchenko, Müller, Lilienfeld,
  Phys. Rev. Lett. 108, 058301 (2012)
  DOI: https://doi.org/10.1103/PhysRevLett.108.058301

- **Bag of Bonds (BoB)** `generate_bob`

  Assessment and Validation of Machine Learning Methods for Predicting Molecular Atomization Energies,
  Hansen, Montavon, Biegler, Fazli, Rupp, Scheffler, Lilienfeld, Tkatchenko, Müller,
  J. Chem. Theory Comput. 2013, 9, 8, 3404–3419
  https://doi.org/10.1021/ct400195d

- **SLATM** `generate_slatm`

  Understanding molecular representations in machine learning: The role of uniqueness and target similarity,
  Huang, Lilienfeld,
  J. Chem. Phys. 145, 161102 (2016)
  https://doi.org/10.1063/1.4964627

- **ACSF** `generate_acsf`

  Atom-centered symmetry functions for constructing high-dimensional neural network potentials,
  Behler,
  J Chem Phys 21;134(7):074106 (2011)
  https://doi.org/10.1063/1.3553717

- **AARAD** `generate_aarad`

  Alchemical and structural distribution based representation for universal quantum machine learning,
  Faber, Christensen, Huang, Lilienfeld,
  J. Chem. Phys. 148, 241717 (2018),
  https://doi.org/10.1063/1.5020710

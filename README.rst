===============
What is qmllib?
===============

``qmllib`` is a Python/Fortran toolkit for representation of molecules and solids
for machine learning of properties of molecules and solids. The library is not
a high-level framework where you can do ``model.train()``, but supplies the
building blocks to carry out efficient and accurate machine learning. As such,
the goal is to provide usable and efficient implementations of concepts such as
representations and kernels.

==============
QML or qmllib?
==============

``qmllib`` represents the core library functionality derived from the original
QML package, providing a powerful toolkit for quantum machine learning
applications, but without the high-level abstraction, for example SKLearn.

This package is and should stay free-function design oriented.

If you are moving from ``qml`` to ``qmllib``, note that there are breaking
changes to the interface to make it more consistent with both argument orders
and function naming.


==============
How to install
==============

You need a fortran compiler, OpenMP and a math library. Default is `gfortran` and `openblas`.

.. code-block:: bash

    sudo apt install gcc libomp-dev libopenblas-dev

If you are on mac, you can install `gcc`, OpenML and BLAS/Lapack via `brew`

.. code-block:: bash

    brew install gcc libomp openblas lapack

You can then install via PyPi

.. code-block:: bash

   pip install qmllib

or directly from github

.. code-block:: bash

    pip install git+https://github.com/qmlcode/qmllib

or if you want a specific feature branch

.. code-block:: bash

    pip install git+https://github.com/qmlcode/qmllib@feature_branch


=================
How to contribute
=================

Know a issue and want to get started developing? Fork it, clone it, make it , test it.

.. code-block:: bash

    git clone your_repo qmllib.git
    cd qmllib.git
    make # setup env
    make compile # compile

You know have a conda environment in `./env` and are ready to run

.. code-block:: bash

    make test

happy developing


==========
How to use
==========

Notebook examples are coming. For now, see test files in ``tests/*``.

===========
How to cite
===========

Please cite the representation that you are using accordingly.

- **Implementation**

  Toolkit for Quantum Chemistry Machine Learning,
  https://github.com/qmlcode/qmllib, <version or git commit>

- **FCHL19** ``generate_fchl19``

  FCHL revisited: Faster and more accurate quantum machine learning,
  Christensen, Bratholm, Faber, Lilienfeld,
  J. Chem. Phys. 152, 044107 (2020),
  https://doi.org/10.1063/1.5126701

- **FCHL18** ``generate_fchl18``

  Alchemical and structural distribution based representation for universal quantum machine learning,
  Faber, Christensen, Huang, Lilienfeld,
  J. Chem. Phys. 148, 241717 (2018),
  https://doi.org/10.1063/1.5020710

- **Columb Matrix** ``generate_columnb_matrix_*``

  Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning,
  Rupp, Tkatchenko, Müller, Lilienfeld,
  Phys. Rev. Lett. 108, 058301 (2012)
  DOI: https://doi.org/10.1103/PhysRevLett.108.058301

- **Bag of Bonds (BoB)** ``generate_bob``

  Assessment and Validation of Machine Learning Methods for Predicting Molecular Atomization Energies,
  Hansen, Montavon, Biegler, Fazli, Rupp, Scheffler, Lilienfeld, Tkatchenko, Müller,
  J. Chem. Theory Comput. 2013, 9, 8, 3404–3419
  https://doi.org/10.1021/ct400195d

- **SLATM** ``generate_slatm``

  Understanding molecular representations in machine learning: The role of uniqueness and target similarity,
  Huang, Lilienfeld,
  J. Chem. Phys. 145, 161102 (2016)
  https://doi.org/10.1063/1.4964627

- **ACSF** ``generate_acsf``

  Atom-centered symmetry functions for constructing high-dimensional neural network potentials,
  Behler,
  J Chem Phys 21;134(7):074106 (2011)
  https://doi.org/10.1063/1.3553717

- **AARAD** ``generate_aarad``

  Alchemical and structural distribution based representation for universal quantum machine learning,
  Faber, Christensen, Huang, Lilienfeld,
  J. Chem. Phys. 148, 241717 (2018),
  https://doi.org/10.1063/1.5020710


===================
What is left to do?
===================

- Compile based on ``FCC`` env variable
- if ``ifort`` find the right flags
- Find MKL from env (for example conda)
- Find what numpy has been linked too (lapack or mkl)
- Notebook examples

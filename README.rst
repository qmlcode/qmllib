====
What
====

``qmllib`` is a Python/Fortran toolkit for representation of molecules and solids
for machine learning of properties of molecules and solids. The library is not
a high-level framework where you can do ``model.train()``, but supplies the
building blocks to carry out efficient and accurate machine learning. As such,
the goal is to provide usable and efficient implementations of concepts such as
representations and kernels.

==============
QML or QMLLib?
==============

``qmllib`` represents the core library functionality derived from the original
QML package, providing a powerful toolkit for quantum machine learning
applications, but without the high-level abstraction, for example SKLearn.

This package is and should stay free-function design oriented.

Breaking changes from ``qml``:

* FCHL representations callable interface to be consistent with other representations (e.i. atoms, coordinates)

==============
How to install
==============

You need a fortran compiler and math library. Default is `gfortran` and `openblas`.


.. code-block:: bash

    sudo apt install libopenblas-dev gcc

You can install it via PyPi

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

.. code-block:: python

    raise NotImplementedError

===========
How to cite
===========

.. code-block:: python

    raise NotImplementedError

=========
What TODO
=========

* Setup ifort flags
* Setup based on FCC env variable or --global-option flags
* Find MKL from env (for example conda)
* Find what numpy has been linked too (lapack or mkl)

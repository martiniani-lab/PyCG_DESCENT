.. image:: https://travis-ci.org/smcantab/PyCG_DESCENT.svg?branch=master
    :target: https://travis-ci.org/smcantab/PyCG_DESCENT

PyCG_DESCENT : Python Conjugate Gradient Descent
++++++++++++++++++++++++++++++++++++++++++++++++

Source code: https://github.com/smcantab/PyCG_DESCENT

Documentation: http://smcantab.github.io/PyCG_DESCENT/

Python wrapper for the Hager and Zang CG_DESCENT algorithm.
CG_DESCENT is a conjugate gradient algorithm for solving an unconstrained minimization
problem of the form::

    min[f(x)]

The algorithm was developed in the following papers (see `W. Hager website <http://users.clas.ufl.edu/hager/papers/CG/>`_):

[1] W. W. Hager and H. Zhang, A new conjugate gradient method with guaranteed descent and an efficient line search,
SIAM Journal on Optimization, 16 (2005), 170-192.

[2] W. W. Hager and H. Zhang, Algorithm 851: CG_DESCENT, A conjugate gradient method with guaranteed descent,
ACM Transactions on Mathematical Software, 32 (2006), 113-137.

[3] W. W. Hager and H. Zhang, A survey of nonlinear conjugate gradient methods, 
Pacific Journal of Optimization, 2 (2006), pp. 35-58.

[4] W. W. Hager and H. Zhang, Limited memory conjugate gradients, 
www.math.ufl.edu/~hager/papers/CG/lcg.pdf

This project wraps the CG_DESCENT C-library (Version 6.8) released by William Hager
under the GNU general public license and adds some more functionalities.
CG_DESCENT wraps the `cg_descent` method first into a c++ optimizer class on the model of the
`pele <https://github.com/pele-python/pele>`_ project and then through Cython into
the `CGDescent` Python class.

The current release requires that the objective function to optimize derives from
the `pele`_ `BasePotential` data structure, future releases will remove this dependency.

Required packages
-----------------

for compilation:

1. c++ compiler (must support c++11, GCC > 4.6 or similar)

python packages:

1. numpy:
     We use numpy everywhere for doing numerical work.

#. `pele`_:
    python energy landscape explorer for potential, minimizers etc.

non-python packages:

1. cmake: optional
    to compile using cmake (much faster)

All the above packages can be installed via the python package manager pip (or
easy_install), with the exception of pele.  However, some of the packages (numpy, scipy)
have additional dependencies and it can be more convenient to use the linux package manager
(apt, yum, ...).

Compilation
-----------
Compilation is required as many of the computationally intensive parts (especially potentials)
are written in fortran and c++. This package uses the standard python setup utility (distutils).
There are lots of options for how and where to install. For more information::

  $ python setup.py --help
  $ python setup.py --help-commands

Developers probably want to install "in-place", i.e. build the extension
modules in their current directories::

  $ python setup.py build_ext -i

Users can install PyCG_DESCENT in the standard python package location::

  $ python setup.py build
  $ python setup.py install [--user]

where --user installs it in $HOME/.local/

We also have have an alternate form of compilation that uses CMake to compile the c++
libraries.  This is *much* faster because it can be done in parallel and can
take advantage of common libraries.  Simply use the file `setup_with_cmake.py`
in place of `setup.py`

Installing on OS X
------------------
On Macbook Air OS X Version 10.9 for an in-place build run

    MACOSX_DEPLOYMENT_TARGET=10.9 python setup.py build_ext -i

PYTHONPATH
----------
If you do an in-place install, make sure to add the install directory to your
PYTHONPATH environment variable.  This is not necessary if you install to a
standard location.

Tests
-----
PyCG_DESCENT has a suite of unit tests.  They can be run using the nose testing
framework (which can be installed using pip).  The tests are run from the top
directory with this command::

  nosetests PyCG_DESCENT

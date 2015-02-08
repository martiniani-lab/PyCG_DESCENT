PyCG_DESCENT : Python Conjugate Gradient Descent
++++++++++++++++++++++++++++++++++++++++++++++++

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

This project wraps the CG_DESCENT C-library (Version 6.7) released by William Hager
under the GNU general public license.
CG_DESCENT wraps the `cg_descent` method first into a c++ optimizer class on the model of the
`pele <https://github.com/pele-python/pele>`_ project and then through Cython into
the :class:`CGDescent <PyCG_DESCENT:PyCG_DESCENT.CGDescent>` Python class.

The current release requires that the objective function to optimize derives from
the :class:`BasePotential <pele:pele.potentials.BasePotential>` data structure, future releases will remove this dependency.

PyCG_DESCENT has been authored by Stefano Martiniani at the University of Cambridge.
The project is publicly available under the GNU general public licence.
   
Reference
---------

.. toctree::
   :maxdepth: 2
	
   CGDescent

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


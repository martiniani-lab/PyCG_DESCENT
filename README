Users interested in applying cg_descent in a parallel setting, please see:

http://www5.cs.fau.de/en/our-team/grimm-robert/downloads/
=======================================================================
cg_descent is a conjugate gradient algorithm for solving
an unconstrained minimization problem of the form:

                    min f (x)

The algorithm is developed in the following papers
(see www.math.ufl.edu/~hager/papers/CG):

[1] W. W. Hager and H. Zhang, A new conjugate gradient method
    with guaranteed descent and an efficient line search,
    SIAM Journal on Optimization, 16 (2005), 170-192.

[2] W. W. Hager and H. Zhang, Algorithm 851: CG_DESCENT,
    A conjugate gradient method with guaranteed descent,
    ACM Transactions on Mathematical Software, 32 (2006), 113-137.

[3] W. W. Hager and H. Zhang, A survey of nonlinear conjugate
    gradient methods, Pacific Journal of Optimization,
    2 (2006), pp. 35-58.

[4] W. W. Hager and H. Zhang, Limited memory conjugate gradients,
         www.math.ufl.edu/~hager/papers/CG/lcg.pdf

This directory contains a C implementation of cg_descent.
A C code which calls cg_descent should include the header file
cg_user.h.  Examples showing how to call cg_descent are given
in driver1.c through driver5.c.  The user must provide routines
to evaluate the objective function and its gradient.  Performance
is often improved if the user also provides a routine to simultaneously
evaluate the objective function and its gradient (see drive1.c).
In the simplest case, cg_descent is invoked with a statement
of the form:

cg_descent (x, n, NULL, NULL, tol, myvalue, mygrad, NULL, NULL) ;

where x is a pointer to an array which contains the starting
guess on input and the solution on output, n is the problem
dimension, tol is the computing tolerance (max norm of the
gradient), myvalue is a routine to evaluate the user's
objective function, and mygrad is a routine to evaluate
the gradient of the user's objective function. The 4 NULL
arguments could be replaced by the following (in order):
a structure to store execution statistics, a structure containing
algorithm parameters, a pointer to a routine which evaluates the
objective function and its gradient, and a pointer to a work
array. If the work array is not provided, then the code
allocates and frees memory. If the routine to simultaneously evaluate
the objective function and its gradient is not provided, then the
code will use myvalue and mygrad to compute the value and
gradient independently. When the algorithm parameters are not
provided, then the default parameter values will be used
(see cg_default for their values).

The subdirectory MATLAB provides the mex functions for using
cg_descent in MATLAB. To compile the mex functions, descend
into the MATLAB directory and type make. See the README file
in the MATLAB directory for information concerning the use of
the code in MATLAB.

If CUTEr or CUTEst are installed, then cg_descent can be installed
in it with the commands cg_cuter_install or cg_cutest_install found
in the directory where cg_descent is located.  Other information
concerning CUTE is given in the README inside the interface
directories. Information concerning the CUTEst testing environment is
available at the following web site:

http://ccpforge.cse.rl.ac.uk/gf/project/cutest/wiki/

cg_descent does loop unrolling, so there is likely no benefit
from using unrolled BLAS. There could be a benefit from using
threaded BLAS if the problems is really big.  To use the BLAS
with cg_descent, comment out the following statement in the
cg_blas.h file:

#define NOBLAS

Also, make any needed adjustments to the BLAS_UNDERSCORE and the
BLAS_START parameters as explained in the cg_blas.h file. In the
Makefile, use the compiler option that includes "-lpthread"
The README file in the CUTEr_interface directory explains how
to use the BLAS with CUTEr.

cg_descent does loop unrolling, so there is likely no benefit
from using unrolled BLAS. There could be a benefit from using
threaded BLAS if the problems is really big.  To use the BLAS
with cg_descent, comment out the following statement in the
cg_blas.h file:

#define NOBLAS

Also, make any needed adjustments to the BLAS_UNDERSCORE and the
BLAS_START parameters as explained in the cg_blas.h file. In the
Makefile, use the compiler option that includes "-lpthread"
The README file in the interface directory explains how to use the
BLAS with CUTE.

# distutils: language = c++
import sys
import numpy as np
from pele.optimize import Result
from pele.potentials._pythonpotential import as_cpp_potential
cimport PyCG_DESCENT._pycgd as _pycgd
cimport cython

@cython.boundscheck(False)
cdef pele_array_to_np_array(_pele.Array[double] v):
    """copy a pele Array into a new numpy array"""
    cdef np.ndarray[double, ndim=1] vnew = np.zeros(v.size(), dtype=float)
    cdef int i
    cdef int N = vnew.size
    for i in xrange(N):
        vnew[i] = v[i]
    
    return vnew

cdef class _Cdef_CGDescent(object):
    """Python interface for c++ CG_DESCENT cpp wrapper

    This class wraps the CG_DESCENT c++ wrapper into a cython class
    for use in Python. This version of the wrapper relies on the
    `pele <pele:pele>` library.

    Parameters
    ----------
    x0 : numpy.array
        these are the initial coordinates for the system
    potential : :class:`BasePotential <pele:pele.potentials.BasePotential>`
        the potential (or cost function) return energy, gradient and hessian
        information given a set of coordinates
    M : int
        number of vectors stored in memory. M=0 corresponds to the standard
        CG_DESCENT method, while M>0 corresponds to the (preconditioned)
        Limited Memory version of the algorithm
    tol: double
        minimisation is terminated when the L^infty-Norm of the gradient is less
        than tol
    nsteps : int
        maximum number of iterations
    print_level: int 0 to 3
        CG_DESCENT verbosity:
        0 no output
        1 to 3 different amount of details about stepping and line search
    verbosity: int
        level of verbosity for `potential <pele:pele.potentials.BasePotential>`

    Attributes
    ----------
    potential : :class:`BasePotential <pele:pele.potentials.BasePotential>`
        the potential (or cost function) return energy, gradient and hessian
        information given a set of coordinates

    Notes
    -----
    for direct access to the underlying c++ optimizer use self.thisptr
    """
    
    def __cinit__(self, x0, potential, double tol=1e-5, int M=0, int print_level=0, int nsteps=10000, int verbosity=0):
        potential = as_cpp_potential(potential, verbose=verbosity>0)
        self.pot = potential
        cdef np.ndarray[double, ndim=1] x0c = np.array(x0, dtype=float)
        self.thisptr = shared_ptr[_pycgd.cCGDescent](new cCGDescent(self.pot.thisptr, 
                             _pele.Array[double](<double*> x0c.data, x0c.size), tol, print_level))
        self.set_memory(M)
        self.set_maxiter(nsteps)
        
    def run(self, niter=None):
        """run CG_DESCENT
        :param niter: maximum number of iterations
        :return: `Result <pele:pele.optimize.result>` container
        """
        if niter is None:
            self.thisptr.get().run()
        else:
            self.thisptr.get().run(niter)
                            
        return self.get_result()
            
    def reset(self, coords):
        """reset coordinates to a new state for a new run

        :param coords: new coordinates
        :return: void
        """
        cdef np.ndarray[double, ndim=1] ccoords = np.array(coords, dtype=float)
        self.thisptr.get().reset(_pele.Array[double](<double*> ccoords.data, ccoords.size))

    def get_iter(self):
        """ get number of iterattions
        :return: int
        """
        return self.thisptr.get().get_iter()
    
    def get_result(self):
        """return a results object
        :return: `Result <pele:pele.optimize.result>` container
            *res.energy : function value
            *res.coords : final coordinates
            *res.grad : gradient vector
            *res.gnorm : L^infy-norm of gradient vector
            *res.rms : root mean square of gradient vector
            *res.nsteps : number of steps
            *res.itersub : number of iterations in subspace
            *res.numsub :
            *res.nfunc
            *res.ngrad
            *res.nfev
            *res.success
        """
        res = Result()
        
        cdef _pele.Array[double] xi = self.thisptr.get().get_x()
        cdef _pele.Array[double] gi = self.thisptr.get().get_g()
        x = pele_array_to_np_array(xi)
        g = pele_array_to_np_array(gi)
        
        res.energy = self.thisptr.get().get_f()
        res.coords = x
        res.grad = g
        
        res.gnorm = self.thisptr.get().get_gnorm()
        res.rms = self.thisptr.get().get_rms()        
        res.nsteps = self.thisptr.get().get_iter()
        res.itersub = self.thisptr.get().get_IterSub()
        res.numsub = self.thisptr.get().get_NumSub()
        res.nfunc = self.thisptr.get().get_nfunc()
        res.ngrad = self.thisptr.get().get_ngrad()
        res.nfev = self.thisptr.get().get_nfev()
        res.success = bool(self.thisptr.get().success())
        return res
    
    def set_print_level(self, val):
        self.thisptr.get().set_PrintLevel(val)
    
    def set_maxiter(self, val):
        self.thisptr.get().set_maxit(val)
    
    def set_memory(self, val):
        self.thisptr.get().set_memory(val)
    
    def set_approx_wolfe(self, val):
        self.thisptr.get().set_AWolfe(val)
    
    def set_approx_wolfe_factor(self, val):
        self.thisptr.get().set_AWolfeFac(val)
    
    def set_lbfgs(self, val):
        self.thisptr.get().set_lbfgs(val)
    
    def set_quad_step(self, val):
        self.thisptr.get().set_QuadStep(val)
        
    def set_use_cubic(self, val):
        self.thisptr.get().set_UseCubic(val)

    def set_step(self, val):
        self.thisptr.get().set_step(val)

    def set_nslow(self, val):
        self.thisptr.get().set_nslow(val)

    def set_stop_rule(self, val):
        self.thisptr.get().set_StopRule(val)
    
class CGDescent(_Cdef_CGDescent):
    """This class defines the python interface for c++ CG_DESCENT cpp wrapper
    """
        
        
        
    

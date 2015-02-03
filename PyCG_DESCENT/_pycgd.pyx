# distutils: language = c++
import sys
import numpy as np

from pele.potentials import _pele
from pele.potentials cimport _pele
from pele.optimize import Result

cimport numpy as np
from libcpp cimport bool as cbool
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
    """this class defines the python interface for c++ CG_DESCENT cpp wrapper 
    
    Notes
    -----
    for direct access to the underlying c++ optimizer use self.thisptr
    """
    
    cdef _pele.BasePotential pot
    
    def __cinit__(self, x0, potential, double tol=1e-5, int M=0, int print_level=0,
                  int nsteps=10000, logger=None):
        potential = as_cpp_potential(potential, verbose=verbosity>0)
        self.pot = potential
        self.thisptr = shared_ptr[_pycgd.cCGGradient](new cCGDescent(self.pot.thisptr, 
                             _pele.Array[double](<double*> x0c.data, x0c.size), tol, print_level))
        self.set_memory(M)
        self.set_maxiter(nsteps)
        
    def run(self, niter=None):
        if niter is None:
            self.thisptr.get().run()
        else:
            self.thisptr.get().run(niter)
                            
        return self.get_result()
            
    def reset(self, coords):
        cdef np.ndarray[double, ndim=1] ccoords = np.array(coords, dtype=float)
        self.thisptr.get().reset(_pele.Array[double](<double*> ccoords.data, ccoords.size))

    def get_iter(self):
        return self.thisptr.get().get_iter()
    
    def get_result(self):
        """return a results object"""
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
        res.itersub = self.thisptr.get().IterSub()
        res.numsub = self.thisptr.get().NumSub()
        res.nfunc = self.thisptr.get().nfunc()
        res.ngrad = self.thisptr.get().ngrad()
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
    
class CGDescent(_Cdef_CGDescent):
    """This class is the python interface for the c++ LBFGS implementation
    """
        
        
        
    

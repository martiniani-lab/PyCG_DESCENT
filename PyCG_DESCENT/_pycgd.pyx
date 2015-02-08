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
        """Level 0 = no printing, ... , Level 3 = maximum printing

        :param val: int
        :return: void
        """
        self.thisptr.get().set_PrintLevel(val)
    
    def set_maxiter(self, val):
        """abort cg after maxiter iterations

        :param val: int
        :return: void
        """
        self.thisptr.get().set_maxit(val)
    
    def set_memory(self, val):
        """number of vectors stored in memory

        :param val: int
        :return: void
        """
        self.thisptr.get().set_memory(val)
    
    def set_approx_wolfe(self, val, factor=None):
        """set True to use approximate Wolfe line search
        if False use ordinary Wolfe line search, switch to approximate Wolfe when
        |f_k+1-f_k| < AWolfeFac*C_k, C_k = average size of cost

        :param val: bool
        :param factor: double
        :return: void
        """
        self.thisptr.get().set_AWolfe(val)
        if factor is not None:
            self.thisptr.get().set_AWolfeFac(val)
    
    def set_lbfgs(self, val):
        """set True to use LBFGS

        :param val: bool
        :return: void
        """
        self.thisptr.get().set_lbfgs(val)

    def set_quad_step(self, val, cutoff=None, quadsafe=None):
        """set true to attempt quadratic interpolation in line search when
        |f_k+1 - f_k|/f_k <= QuadCutoff

        :param val: bool
        :param cutoff: double
        :param quadsafe: double
            maximum factor by which a quad step can reduce the step size
        :return: void
        """
        self.thisptr.get().set_QuadStep(val)
        if cutoff is not None:
            self.thisptr.get().set_QuadCutOff(cutoff)
        if quadsafe is not None:
            self.thisptr.get().set_QuadSafe(quadsafe)
        
    def set_use_cubic(self, val, cutoff=None):
        """set True to use cubic step when |f_k+1 - f_k|/|f_k| > CubicCutOff

        :param val: bool
        :param cutoff: double
        :return: void
        """
        self.thisptr.get().set_UseCubic(val)
        if cutoff is not None:
            self.thisptr.get().set_CubicCutOff(cutoff)

    def set_small_cost(self, val):
        """when |f| < SmallCost*starting cost then skip QuadStep and set PertRule = False

        :param val: double
        :return: void
        """
        self.thisptr.get().set_SmallCost(val)

    def set_step(self, val):
        """if step is nonzero, it is the initial step of the initial line search

        :param val: double
        :return: void
        """
        self.thisptr.get().set_step(val)

    def set_nslow(self, val):
        """terminate after nslow iterations without strict improvement in either function value or gradient

        :param val: int
        :return: void
        """
        self.thisptr.get().set_nslow(val)

    def set_stop_rule(self, val, factor=None):
        """set True to stop when ||proj_grad||_infty <= max(grad_tol,initial ||grad||_infty*StopFact)
        if False then stop when ||proj_grad||_infty <= grad_tol*(1 + |f_k|)

        :param val: bool
        :param factor: double
        :return: void
        """
        self.thisptr.get().set_StopRule(val)
        if factor is not None:
            self.thisptr.get().set_StopFac(factor)

    def set_subspace_check(self, check_freq, skip=None):
        """controls the frequency with which the subspace condition is checked.

        It is checked for SubCheck*mem iterations and if not satisfied,
        then it is skipped for Subskip*mem iterations and Subskip is doubled.
        Whenever the subspace condition is statisfied, SubSkip is returned to
        its original value.

        :param check_freq: int
        :param skip: int
        :return: void
        """
        self.thisptr.get().set_SubCheck(check_freq)
        if skip is not None:
            self.thisptr.get().set_SubSkip(skip)

    def set_eta0(self, val):
        """when relative distance from current gradient to subspace <= eta0,
        enter subspace if subspace dimension = mem

        :param val: double
        :return: void
        """
        self.thisptr.get().set_eta0(val)

    def set_eta1(self, val):
        """when relative distance from current gradient to subspace >= eta1,
        leave subspace

        :param val: double
        :return: void
        """
        self.thisptr.get().set_eta1(val)

    def set_eta2(self, val):
        """when relative distance from current direction to subspace <= eta2,
        always enter subspace (invariant space)

        :param val: double
        :return: void
        """
        self.thisptr.get().set_eta2(val)

    def set_qdecay(self, val):
        """factor in [0, 1] used to compute average cost magnitude C_k as follows:
        Q_k = 1 + (Qdecay)Q_k-1, Q_0 = 0,  C_k = C_k-1 + (|f_k| - C_k-1)/Q_k

        :param val: double
        :return: void
        """
        self.thisptr.get().set_Qdecay(val)

    def set_pert_rule(self, val, eps=None):
        """if True estimated error in function value is eps*Ck
        otherwise estimated error in function value is eps

        :param val: bool
        :param eps: double
            error in function value
        :return: void
        """
        self.thisptr.get().set_PertRule(val)
        if eps is not None:
            self.thisptr.get().set_eps(eps)

    def set_egrow(self, val):
        """factor by which eps grows when line search fails during contraction

        :param val: double
        :return: void
        """
        self.thisptr.get().set_egrow(val)

    def set_debug(self, val, tol=None):
        """set True to check that f_k+1 - f_k <= debugtol*C_k
        otherwise don't check function values

        :param val: bool
        :param tol: double
        :return: void
        """
        self.thisptr.get().set_debug(val)
        if tol is not None:
            self.thisptr.get().set_debugtol(tol)

    def set_ntries(self, val):
        """maximum number of times the bracketing interval grows during expansion

        :param val: int
        :return: void
        """
        self.thisptr.get().set_ntries(val)

    def set_expand_safe(self, val):
        """maximum factor secant step increases stepsize in expansion phase

        :param val: double
        :return: void
        """
        self.thisptr.get().set_ExpandSafe(val)

    def set_secant_amplification(self, val):
        """factor by which secant step is amplified during expansion phase
        where minimizer is bracketed

        :param val: double
        :return: void
        """
        self.thisptr.get().set_SecantAmp(val)

    def set_rho_grow(self, val):
        """factor by which rho grows during expansion phase where minimizer is bracketed

        :param val: double
        :return: void
        """
        self.thisptr.get().set_RhoGrow(val)

    def set_neps(self, val):
        """maximum number of times that eps is updated

        :param val: int
        :return: void
        """
        self.thisptr.get().set_neps(val)

    def set_nshrink(self, val):
        """maximum number of times the bracketing interval shrinks

        :param val: int
        :return: void
        """
        self.thisptr.get().set_nshrink(val)

    def set_nline(self, val):
        """maximum number of iterations in line search

        :param val: int
        :return: void
        """
        self.thisptr.get().set_nline(val)

    def set_restart_factor(self, val):
        """conjugate gradient method restarts after (n*restart_fac) iterations

        :param val: double
        :return: void
        """
        self.thisptr.get().set_restart_fac(val)

    def set_feps(self, val):
        """stop when -alpha*dphi0 (estimated change in function value) <= feps*|f|

        :param val: double
        :return: void
        """
        self.thisptr.get().set_feps(val)

    def set_nan_rho(self, val):
        """after encountering nan, growth factor when searching for a bracketing interval

        :param val: double
        :return: void
        """
        self.thisptr.get().set_nan_rho(val)

    def set_nan_decay(self, val):
        """after encountering nan, decay factor for stepsize

        :param val: double
        :return: void
        """
        self.thisptr.get().set_nan_decay(val)

    def set_delta(self, val):
        """Wolfe line search parameter

        :param val: double
        :return: void
        """
        self.thisptr.get().set_delta(val)

    def set_sigma(self, val):
        """Wolfe line search parameter

        :param val: double
        :return: void
        """
        self.thisptr.get().set_sigma(val)

    def set_gamma(self, val):
        """decay factor for bracket interval width

        :param val: double
        :return: void
        """
        self.thisptr.get().set_gamma(val)

    def set_rho(self, val):
        """growth factor when searching for initial bracketing interval

        :param val: double
        :return: void
        """
        self.thisptr.get().set_rho(val)

    def set_psi0(self, val):
        """factor used in starting guess for iteration 1

        :param val: double
        :return: void
        """
        self.thisptr.get().set_psi0(val)

    def set_psi_bounds(self, low=None, high=None):
        """in performing a QuadStep, we evaluate at point betweeen
        [psi_lo, psi_hi]*psi2*previous step

        :param low: double
        :param high: double
        :return: void
        """
        if low is not None:
            self.thisptr.get().set_psi_lo(low)
        if high is not None:
            self.thisptr.get().set_psi_hi(high)

    def set_psi1(self, val):
        """for approximate quadratic, use gradient at psi1*psi2*previous step
        for initial stepsize

        :param val: double
        :return: void
        """
        self.thisptr.get().set_psi1(val)

    def set_psi2(self, val):
        """when starting a new cg iteration, our initial guess for the line search
        stepsize is psi2*previous step

        :param val: double
        :return: void
        """
        self.thisptr.get().set_psi2(val)

    def set_adaptive_beta(self, val):
        """set True to choose beta adaptively

        :param val: bool
        :return: void
        """
        self.thisptr.get().set_AdaptiveBeta(val)

    def set_beta_lower(self, val):
        """set lower bound factor for beta

        :param val: double
        :return: void
        """
        self.thisptr.get().set_BetaLower(val)

    def set_theta(self, val):
        """parameter describing the cg_descent family

        :param val: double
        :return: void
        """
        self.thisptr.get().set_theta(val)

    def set_qeps(self, val):
        """parameter in cost error for quadratic restart criterion

        :param val: double
        :return: void
        """
        self.thisptr.get().set_qeps(val)

    def set_qrule(self, val):
        """parameter used to decide if cost is quadratic

        :param val: double
        :return: void
        """
        self.thisptr.get().set_qrule(val)

    def set_qrestart(self, val):
        """number of iterations the function should be nearly quadratic before a restart

        :param val: int
        :return: void
        """
        self.thisptr.get().set_qrestart(val)

class CGDescent(_Cdef_CGDescent):
    """Python interface for c++ CG_DESCENT cpp/cython wrapper

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
    """
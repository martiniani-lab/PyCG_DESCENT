cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr
from libcpp cimport bool as cbool
cimport numpy as np

cdef extern from "PyCG_DESCENT/cg_descent_wrapper.h" namespace "pycgd":
    cdef cppclass  cCGDescent "pycgd::cg_descent":
        cCGDescent(shared_ptr[_pele.cBasePotential], _pele.Array[double], double, size_t) except +
        void run(int) except+
        void run() except+
        void reset(_pele.Array[double]&) except+
        void set_PrintLevel(int) except+
        void set_maxit(size_t) except+
        void set_memory(int) except+
        void set_AWolfeFac(double) except+
        void set_AWolfe(cbool) except+
        void set_lbfgs(cbool) except+
        void set_QuadStep(cbool) except+
        void set_QuadCutOff(double) except+
        void set_QuadSafe(double) except+
        void set_UseCubic(cbool) except+
        void set_CubicCutOff(double) except+
        void set_SmallCost(double) except+
        void set_step(double) except+
        void set_nslow(int) except+
        void set_StopRule(cbool) except+
        void set_StopFac(double) except+
        #void set_x(_pele.Array[double]) except+
        void set_SubCheck(int) except+
        void set_SubSkip(int) except+
        void set_eta0(double) except+
        void set_eta1(double) except+
        void set_eta2(double) except+
        void set_Qdecay(double) except+
        void set_PertRule(cbool) except+
        void set_eps(double) except+
        void set_egrow(double) except+
        void set_debug(cbool) except+
        void set_debugtol(double) except+
        void set_ntries(int) except+
        void set_ExpandSafe(double) except+
        void set_SecantAmp(double) except+
        void set_RhoGrow(double) except+
        void set_neps(int) except+
        void set_nshrink(int) except+
        void set_nline(int) except+
        void set_restart_fac(double) except+
        void set_feps(double) except+
        void set_nan_rho(double) except+
        void set_nan_decay(double) except+
        void set_delta(double) except+
        void set_sigma(double) except+
        void set_gamma(double) except+
        void set_rho(double) except+
        void set_psi0(double) except+
        void set_psi_lo(double) except+
        void set_psi_hi(double) except+
        void set_psi1(double) except+
        void set_psi2(double) except+
        void set_AdaptiveBeta(cbool) except+
        void set_BetaLower(double) except+
        void set_theta(double) except+
        void set_qeps(double) except+
        void set_qrule(double) except+
        void set_qrestart(int) except+
        double get_f() except+
        double get_gnorm() except+
        size_t get_iter() except+
        size_t get_IterSub() except+
        size_t get_NumSub() except+
        size_t get_nfunc() except+
        size_t get_ngrad() except+
        size_t get_nfev() except+
        double get_rms() except+
        _pele.Array[double] get_x() except+
        _pele.Array[double] get_g() except+
        cbool success() except+
        
cdef class _Cdef_CGDescent:
    cdef shared_ptr[cCGDescent] thisptr      # hold a C++ instance which we're wrapping
    cdef _pele.BasePotential pot

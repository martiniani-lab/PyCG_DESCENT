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
        void set_PrintLevel(size_t) except+
        void set_maxit(size_t) except+
        void set_memory(size_t) except+
        void set_AWolfeFac(double) except+
        void set_AWolfe(cbool) except+
        void set_lbfgs(cbool) except+
        void set_QuadStep(cbool) except+
        void set_UseCubic(cbool) except+
        void set_x(_pele.Array[double]) except+
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

#ifndef _PELE_CGD_WRAPPER_H__
#define _PELE_CGD_WRAPPER_H__

#include <math.h>
#include <memory>
#include "pele/array.h"
#include "pele/base_potential.h"
#include "pele/optimizer.h"
#include "cg_user.h"

/* entries in cg_descent
 * cg_descent (x, n, NULL, NULL, grad_tol, user_value, user_gradient, user_value_gradient, NULL) ;
 * where n is x.size(), grad_tol is a float
 * must have signature
 *
 *  double user_value(double *x, INT n);
 *  void mygrad(double *g, double *x, INT n);
 *
 * Performance is often improved if the user also provides a routine to
 * simultaneously evaluate the objective function and its gradient
 *
 *  double myvalgrad(double *g, double *x, INT n);
 *
 *  NOTE:   this version relies on the pele library
 *
*/

std::shared_ptr<pele::BasePotential> glob_pot;
size_t glob_nfev=0;

inline double value(double* x, INT n){
    pele::Array<double> xarray(x, (size_t) n);
    ++glob_nfev;
    return glob_pot->get_energy(xarray);
}

inline void gradient(double* g, double* x, INT n){
    pele::Array<double> xarray(x, (size_t) n);
    pele::Array<double> garray(g, (size_t) n);
    ++glob_nfev;
    double f = glob_pot->get_energy_gradient(xarray, garray);
}

inline double value_gradient(double* g, double* x, INT n){
    pele::Array<double> xarray(x, (size_t) n);
    pele::Array<double> garray(g, (size_t) n);
    ++glob_nfev;
    return glob_pot->get_energy_gradient(xarray, garray);
}

class pycg_descent{
protected:
    pele::BasePotential m_pot;
    cg_parameter m_parm;
    cg_stats m_stats;
    pele::Array<double> m_x0, m_x, m_g;
    double m_tol;
    size_t m_nfev;
public:
    pycg_descent(std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0, size_t iprint=0, double tol=1e-4):
        m_pot(potential),
        m_parm(),
        m_stats(),
        m_x0(x0.copy()),
        m_x(x0.copy()),
        m_tol(tol),
        m_nfev(0)
    {
        cg_default(&m_parm); /*set default parameter values*/
        m_parm.PrintFinal = FALSE;
        m_parm.AWolfeFac = 0.;
        m_parm.AWolfe = FALSE;
        m_parm.memory = 0;
        m_parm.maxit = 1e5;
    };
    ~pycg_descent(){}

    //run

    inline void run(){
        INT cgout = cg_descent(xcopy.data(), xcopy.size(), &Stats, &Parm, tol, value, gradient, value_gradient, NULL);
        m_nfev = glob_nfev;
        glob_nfev = 0; //reset global variable
    }

    inline void run(size_t maxiter){
        this->set_max_iter(maxiter);
        this->run();
    }

    inline void set_memory(size_t memory){ m_parm.memory = (INT) memory; }
    inline void use_lbfgs(bool val){ m_parm.LBFGS = (int) val; }
    inline void set_AWolfFac(bool val){ m_parm.AWolfe = (int) val; }
    inline void set_AWolfFac(size_t val){ m_parm.AWolfeFac = (INT) val; }
    inline void set_maxit(size_t val){ m_parm.maxit = (int) val; }
    inline void set_QuadStep(bool val){ m_parm.QuadStep = (int) val; }
    inline void set_UseCubic(bool val){ m_parm.UseCubic = (int) val; }

    /*function value at solution */
    inline double get_f(){ return m_stats.f; };
    /* max abs component of gradient */
    inline double get_gnorm(){ return m_stats.gnorm; };
    /* number of iterations */
    inline size_t get_iter(){ return m_stats.iter; };
    /* number of subspace iterations */
    inline size_t get_IterSub(){ return m_stats.IterSub; };
    /* total number subspaces */
    inline size_t get_NumSub(){ return m_stats.NumSub; };
    /* number of function evaluations */
    inline size_t get_nfunc(){ return m_stats.nfunc; };
    /* number of gradient evaluations */
    inline size_t get_ngrad(){ return m_stats.ngrad; };
    /* total number of function evaluations from global counter*/
    inline size_t get_nfev(){ return m_nfev; };

    inline bool success(INT cgout){
        if (cgout == 0){
            return true;
        }
        else{
            switch (cgout){
            case -2:
                std::cout<<"function value became nan"<<std::endl;
                break;
            case -1:
                std::cout<<"starting function value is nan"<<std::endl;
                break;
            case 1:
                std::cout<<"change in func <= feps*|f|"<<std::endl;
                break;
            case 2:
                std::cout<<"total iterations exceeded maxit"<<std::endl;
                break;
            case 3:
                std::cout<<"slope always negative in line search"<<std::endl;
                break;
            case 4:
                std::cout<<"number secant iterations exceed nsecant"<<std::endl;
                break;
            case 5:
                std::cout<<"search direction not a descent direction"<<std::endl;
                break;
            case 6:
                std::cout<<"line search fails in initial interval"<<std::endl;
                break;
            case 7:
                std::cout<<"line search fails during bisection"<<std::endl;
                break;
            case 8:
                std::cout<<"line search fails during interval update"<<std::endl;
                break;
            case 9:
                std::cout<<"debugger is on and the function value increases"<<std::endl;
                break;
            case 10:
                std::cout<<"out of memory"<<std::endl;
                break;
            default:
                std::cout<<"failed, value not known"<<std::endl;
                break;
            };
        return false;
        }
    }
};

#endif

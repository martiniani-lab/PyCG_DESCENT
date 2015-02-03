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
*/

std::shared_ptr<pele::BasePotential> glob_pot;
size_t nfev=0;

inline double value(double* x, INT n){
    pele::Array<double> xarray(x, (size_t) n);
    ++nfev;
    return glob_pot->get_energy(xarray);
}

inline void gradient(double* g, double* x, INT n){
    pele::Array<double> xarray(x, (size_t) n);
    pele::Array<double> garray(g, (size_t) n);
    ++nfev;
    double f = glob_pot->get_energy_gradient(xarray, garray);
}

inline double value_gradient(double* g, double* x, INT n){
    pele::Array<double> xarray(x, (size_t) n);
    pele::Array<double> garray(g, (size_t) n);
    ++nfev;
    return glob_pot->get_energy_gradient(xarray, garray);
}

size_t cgd_wrapper(std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0, double tol=1e-4)
{
    glob_pot = potential;
    pele::Array<double> xcopy(x0.copy());
    cg_parameter Parm;
    cg_default (&Parm);    /* set default parameter values */
    cg_stats Stats;

    Parm.PrintFinal = FALSE;
    //Parm.LBFGS = TRUE;
    Parm.AWolfeFac = 0.;
    Parm.AWolfe = FALSE;
    Parm.memory = 0;
    //Parm.QuadStep = FALSE ; /* change QuadStep to FALSE */
    Parm.maxit = 1e5;
    //Parm.PrintLevel = 1;
    //Parm.QuadStep = FALSE;
    //Parm.UseCubic = FALSE;

    INT cgout = cg_descent(xcopy.data(), xcopy.size(), &Stats, &Parm, tol, value, gradient, value_gradient, NULL);
    std::cout<<"cgout "<<cgout<<std::endl;
    pele::Array<double> grad(x0.size());
    double f = potential->get_energy_gradient(xcopy, grad);
    std::cout<<"cgd f "<<f<<std::endl;
    std::cout<<"rms "<<pele::norm(grad)/sqrt(x0.size()/3)<<std::endl;

    return Stats.nfunc;
}

#endif

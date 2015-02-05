#include <random>
#include <iostream>
#include <fstream>
#include <string>

#include "pele/lj.h"
#include "pele/lbfgs.h"
#include "pele/matrix.h"
#include "pele/array.h"
#include "PyCG_DESCENT/cg_descent_wrapper.h"

using std::string;

int main(int argc, char ** argv)
{
    std::cout << std::setprecision(16);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0,1);

    size_t natoms = 500;
    pele::Array<double> x(3*natoms);

    auto lj = std::make_shared<pele::LJ>(4., 4.);

    size_t totiter = 10;
    /*size_t tot_nfev_lbfgs = 0;
    for (size_t j=0;j<totiter;++j)
    {
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = 2 * (1 - 0.5 * distribution(generator));
        }

        pele::LBFGS lbfgs(lj, x, 1e-4, 11);
        lbfgs.set_max_iter(100000);
        //lbfgs.set_iprint(500);
        lbfgs.set_maxstep(10);
        lbfgs.run();
        tot_nfev_lbfgs += lbfgs.get_nfev();
        std::cout<<"lbfgs f"<<lbfgs.get_f()<<std::endl;
        std::cout<<"lbfgs nfev"<<lbfgs.get_nfev()<<std::endl;
    }
    std::cout<<"lbfgs avg nfev "<<tot_nfev_lbfgs/totiter<<std::endl;*/

//conjugate gradient descent
    pycgd::cg_descent cg_descent(lj, x, 1e-4);
    cg_descent.set_memory(0);

//    size_t tot_nfev_cgd = 0;
    size_t tot_nfev_cgd = 0;
    for (size_t j=0;j<totiter;++j)
    {
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = 2 * (1 - 0.5 * distribution(generator));
        }
        cg_descent.reset(x);
        cg_descent.run();
        std::cout<<"energy: "<<cg_descent.get_f()<<std::endl;
        std::cout<<"nfev: "<<cg_descent.get_nfev()<<std::endl;
        std::cout<<"RMS: "<<cg_descent.get_rms()<<std::endl;
        tot_nfev_cgd += cg_descent.get_nfev();
    }
    std::cout<<"cgd avg glob nfev "<<tot_nfev_cgd/totiter<<std::endl;
//    std::cout<<"cgd avg nfev "<<tot_nfev_cgd/totiter<<std::endl;
}

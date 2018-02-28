#include <iostream>
#include <sstream>
#include "common.h"
#include "wireframe.h"
#include "mesh.h"
#include "master.h"
#include "function.h"

template<int N, int P>
void projection_test()
{
    std::cout << "P = " << P << std::endl;
    std::stringstream ss;
    for (int i=4; i<=32; i*=2) {
        ss << "data/square_" << i << ".mesh";
        DG::Wireframe<N> wireframe(ss.str());
        DG::Mesh<N,P> mesh(wireframe);
        auto f = [](DG::Tuple<double,N> x) { return cos(2*M_PI*x[0])*cos(2*M_PI*x[1]) + x[0] + x[1]; };
        DG::Function<N,P> f_true(mesh, f);
        DG::Function<N,P> f_proj = mesh.l2_project(f);
        std::cout << (f_proj.vec() - f_true.vec()).template lpNorm<Eigen::Infinity>() / f_true.max_norm() << std::endl;
        ss.str("");
        ss.clear();
    }
}

int main()
{
    const int N = 2;
    DG::Timer::off();
    projection_test<N,1>();
    projection_test<N,2>();
    projection_test<N,3>();
    projection_test<N,4>();
    projection_test<N,5>();
}

#include <functional>
#include <cmath>
#include <iostream>
#include "common.h"
#include "quadtree.h"
#include "mesh.h"
#include "function.h"
#include "boundaryconditions.h"
#include "ldgpoisson.h"

int main()
{
    const int p = 2;    // Polynomial order
    const int N = 2;    // Dimension
    const int P = p+1;  // Nodes per dimension
    double tau0 = 0;    // Interior penalty parameter
    double tauD = 1000; // Dirichlet penalty parameter

    std::function<DG::Tuple<double,N>(DG::Tuple<double,N>)> h = [](const DG::Tuple<double,N>) { return DG::Tuple<double,N>(0.125); };
    DG::Quadtree<N> qt(h);
    DG::Mesh<P,N> mesh(qt);

    auto bcs = DG::BoundaryConditions<P,N>::Dirichlet(mesh);
    DG::LDGPoisson<P,N> poisson(mesh, bcs, tau0, tauD);
    poisson.dump();

    DG::Function<P,N> u(mesh, [](DG::Tuple<double,N> x) { return sin(M_PI*x[0])*sin(M_PI*x[1]); } );
    DG::Function<P,N> f(mesh, [](DG::Tuple<double,N> x) { return 2*M_PI*M_PI*sin(M_PI*x[0])*sin(M_PI*x[1]); } );

    u.write("u.fun");
    f.write("f.fun");

    //DG::Fun<P,N> u = poisson.solve(f);

    return 0;
}

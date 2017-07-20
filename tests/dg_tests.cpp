#include <functional>
#include <cmath>
#include <iostream>
#include "common.h"
#include "quadtree.h"
#include "mesh.h"
#include "function.h"
#include "boundaryconditions.h"
#include "ldgpoisson.h"

int main(int argc, char* argv[])
{
    const int p = 2;    // Polynomial order
    const int N = 2;    // Dimension
    const int P = p+1;  // Nodes per dimension

    double dx = 0.125;
    bool periodic = false;
    double tau0 = 0;    // Interior penalty parameter
    double tauD = 1000; // Dirichlet penalty parameter
    int coarsening = 0;

    if (argc >= 2) dx = atof(argv[1]);
    if (argc >= 3) periodic = (bool)atoi(argv[2]);
    if (argc >= 4) tau0 = atof(argv[3]);
    if (argc >= 4) tauD = atof(argv[4]);

    std::function<DG::Tuple<double,N>(DG::Tuple<double,N>)> h = [dx](const DG::Tuple<double,N>) { return DG::Tuple<double,N>(dx); };
    DG::Quadtree<N> qt(h, periodic);
    DG::Mesh<P,N> mesh(qt, coarsening);

    auto bcs = periodic ? DG::BoundaryConditions<P,N>::Periodic(mesh) : DG::BoundaryConditions<P,N>::Dirichlet(mesh);
    DG::LDGPoisson<P,N> poisson(mesh, bcs, tau0, tauD);
    poisson.dump();

    auto ufun = periodic ? [](DG::Tuple<double,N> x) { return cos(2*M_PI*x[0])*cos(2*M_PI*x[1]); } :
                           [](DG::Tuple<double,N> x) { return sin(2*M_PI*x[0])*sin(2*M_PI*x[1]); };
    auto ffun = periodic ? [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*cos(2*M_PI*x[0])*cos(2*M_PI*x[1]); } :
                           [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*sin(2*M_PI*x[0])*sin(2*M_PI*x[1]); };

    DG::Function<P,N> u(mesh, ufun);
    DG::Function<P,N> f(mesh, ffun);

    u.write("u.fun");
    f.write("f.fun");

    //DG::Fun<P,N> u = poisson.solve(f);

    return 0;
}

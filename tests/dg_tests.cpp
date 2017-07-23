#include <functional>
#include <cmath>
#include <iostream>
#include <stdexcept>
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
    DG::BoundaryType bctype = DG::kDirichlet;
    double tau0 = 0;    // Interior penalty parameter
    double tauD = 1000; // Dirichlet penalty parameter
    int coarsening = 0;

    if (argc >= 2) dx     = atof(argv[1]);
    if (argc >= 3) bctype = (DG::BoundaryType)atoi(argv[2]);
    if (argc >= 4) tau0   = atof(argv[3]);
    if (argc >= 4) tauD   = atof(argv[4]);

    std::function<DG::Tuple<double,N>(DG::Tuple<double,N>)> h = [dx](const DG::Tuple<double,N>) { return DG::Tuple<double,N>(dx); };
    DG::Quadtree<N> qt(h, bctype == DG::kPeriodic);
    DG::Mesh<P,N> mesh(qt, coarsening);

    DG::BoundaryConditions<P,N> bcs(mesh);
    std::function<double(DG::Tuple<double,N>)> ufun, ffun;
    switch (bctype) {
        case DG::kDirichlet:
            ufun = [](DG::Tuple<double,N> x) { return cos(2*M_PI*x[0])*cos(2*M_PI*x[1]) + x[0] + x[1]; };
            ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*cos(2*M_PI*x[0])*cos(2*M_PI*x[1]); };
            bcs = DG::BoundaryConditions<P,N>::Dirichlet(mesh, ufun);
            break;
        case DG::kNeumann:
            ufun = [](DG::Tuple<double,N> x) { return cos(2*M_PI*x[0])*cos(2*M_PI*x[1]) + x[0]*(1-x[0]) + x[1]*(1-x[1]); };
            ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*cos(2*M_PI*x[0])*cos(2*M_PI*x[1])+ 4.0; };
            bcs = DG::BoundaryConditions<P,N>::Neumann(mesh, -1.0);
            break;
        case DG::kPeriodic:
            ufun = [](DG::Tuple<double,N> x) { return sin(2*M_PI*x[0])*sin(2*M_PI*x[1]); };
            ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*sin(2*M_PI*x[0])*sin(2*M_PI*x[1]); };
            DG::BoundaryConditions<P,N>::Periodic(mesh);
            break;
        default:
            throw std::invalid_argument("Unknown boundary condition.");
    }

    // Set up the test
    DG::Function<P,N> u_true(mesh, ufun);
    DG::Function<P,N> f(mesh, ffun);
    if (bctype == DG::kNeumann || bctype == DG::kPeriodic) u_true.meanZero();

    // Discretize and solve
    DG::LDGPoisson<P,N> poisson(mesh, bcs, tau0, tauD);
    DG::Function<P,N> u = poisson.solve(f);

    // Output the data
    u.write("data/u.fun");
    u_true.write("data/u_true.fun");
    poisson.dump();

    return 0;
}

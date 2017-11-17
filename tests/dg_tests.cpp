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
#include "timer.h"
#include "multigrid.h"

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
    if (argc >= 5) tauD   = atof(argv[4]);

    auto h = [dx](const DG::Tuple<double,N>) { return DG::Tuple<double,N>(dx); };

    DG::Timer::tic();
    DG::Quadtree<N> qt(h, bctype == DG::kPeriodic);
    DG::Timer::toc("Build quadtree");

    DG::Timer::tic();
    DG::Mesh<N,P> mesh(qt, coarsening);
    DG::Timer::toc("Build mesh");

    DG::BoundaryConditions<N,P> bcs(mesh);
    std::function<double(DG::Tuple<double,N>)> ufun, ffun;
    switch (bctype) {
        case DG::kDirichlet:
            ufun = [](DG::Tuple<double,N> x) { return cos(2*M_PI*x[0])*cos(2*M_PI*x[1]) + x[0] + x[1]; };
            ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*cos(2*M_PI*x[0])*cos(2*M_PI*x[1]); };
            bcs = DG::BoundaryConditions<N,P>::Dirichlet(mesh, ufun);
            break;
        case DG::kNeumann:
            ufun = [](DG::Tuple<double,N> x) { return cos(2*M_PI*x[0])*cos(2*M_PI*x[1]) + x[0]*(1-x[0]) + x[1]*(1-x[1]); };
            ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*cos(2*M_PI*x[0])*cos(2*M_PI*x[1])+ 4.0; };
            bcs = DG::BoundaryConditions<N,P>::Neumann(mesh, -1.0);
            break;
        case DG::kPeriodic:
            ufun = [](DG::Tuple<double,N> x) { return sin(2*M_PI*x[0])*sin(2*M_PI*x[1]); };
            ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*sin(2*M_PI*x[0])*sin(2*M_PI*x[1]); };
            bcs = DG::BoundaryConditions<N,P>::Periodic(mesh);
            break;
        default:
            throw std::invalid_argument("Unknown boundary condition.");
    }

    // Set up the test
    DG::Function<N,P> u_true(mesh, ufun);
    DG::Function<N,P> f(mesh, ffun);
    if (bctype == DG::kNeumann || bctype == DG::kPeriodic) u_true.meanZero();

    // Discretize the Poisson problem
    DG::LDGPoisson<N,P> poisson(mesh, bcs, tau0, tauD);

    // Build the multigrid hierarchy
    DG::Timer::tic();
    DG::InterpolationHierarchy<N,P> hierarchy(qt);
    DG::Multigrid<N,P> mg(poisson.ops(), hierarchy);
    auto precon = [&mg](const DG::Vector& b) {
        mg.solution().setZero();
        mg.rhs() = b;
        mg.vcycle();
        DG::Vector x = mg.solution();
        return x;
    };
    DG::Timer::toc("Build multigrid hierarchy");

    // Add the forcing function to the RHS
    DG::Function<N,P> rhs = poisson.computeRHS(f);

    // Solve with MGPCG
    DG::Timer::tic();
    DG::Function<N,P> u(mesh);
    DG::pcg(poisson.ops()->A, rhs.vec(), u.vec(), precon);
    DG::Timer::toc("Solve using MGPCG");

    // Output the data
    DG::Timer::tic();
    u.write("data/u.fun");
    u_true.write("data/u_true.fun");
    poisson.dump();
    DG::Timer::toc("Output data");

    return 0;
}

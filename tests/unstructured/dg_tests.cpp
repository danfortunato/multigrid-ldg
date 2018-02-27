#include <iostream>
#include <cmath>
#include "common.h"
#include "wireframe.h"
#include "mesh.h"
#include "master.h"
#include "ldgpoisson.h"

int main(int argc, char* argv[])
{
    const int N = 2;   // Dimension
    const int p = 2;   // Polynomial order
    const int P = p+1; // Number of nodes per dimension

    DG::Wireframe<N> wireframe("data/circle.mesh");
    std::cout << "Points = " << wireframe.np << ", Triangles = " << wireframe.nt << std::endl;

    auto bnd_circle = [] (DG::Tuple<double,N> p) { return std::sqrt(p.square().sum())>1-1e-3; };
    // auto bnd_square = [] (DG::Tuple<double,N> p) { return (p<1e-3 || p>1-1e-3).any(); };
    DG::Mesh<N,P> mesh(wireframe, {bnd_circle});

    // Circle tests

    // Zero Dirichlet
    auto ufun = [](DG::Tuple<double,N> x) { return (1.0-x.square().sum())/(2.0*N); };
    auto ffun = [](DG::Tuple<double,N> x) { return 1.0; };
    DG::BoundaryConditions<N,P> bcs = DG::BoundaryConditions<N,P>::Dirichlet(mesh);

    // Nonzero Dirichlet
    // auto ufun = [](DG::Tuple<double,N> x) { return cos(2*M_PI*x[0])*cos(2*M_PI*x[1]) + x[0] + x[1]; };
    // auto ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*cos(2*M_PI*x[0])*cos(2*M_PI*x[1]); };
    // DG::BoundaryConditions<N,P> bcs = DG::BoundaryConditions<N,P>::Dirichlet(mesh, ufun);

    // Square tests

    // Zero Dirichlet
    // auto ufun = [](DG::Tuple<double,N> x) { return sin(2*M_PI*x[0])*sin(2*M_PI*x[1]); };
    // auto ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*sin(2*M_PI*x[0])*sin(2*M_PI*x[1]); };

    DG::Function<N,P> u(mesh, ufun);
    DG::Function<N,P> f(mesh, ffun);
    u.write("data/u.fun");

    double tau0 = 100, tauD = 1000;
    DG::LDGPoisson<N,P> poisson(mesh, bcs, tau0, tauD);
    DG::Function<N,P> rhs = poisson.computeRHS(f);
    rhs.write("data/f_rhs.fun");
    poisson.dump();

    return 0;
}

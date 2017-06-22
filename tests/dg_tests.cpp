#include "quadtree.h"
#include "mesh.h"
#include "function.h"
#include "boundaryconditions.h"
#include "ldgpoisson.h"

int main()
{
    const int p = 3;   // Polynomial order
    const int N = 2;   // Dimension
    const int P = p+1; // Nodes per dimension

    DG::Quadtree<P,N> qt(3);
    DG::Mesh<P,N> mesh(qt);
    //DG::Cartesian<P,N> mesh(32,32);

    DG::Function<P,N> f(mesh);
    auto bc = DG::BoundaryConditions<P,N>::Dirichlet(mesh);

    f = 1;
    DG::LDGPoisson<P,N> poisson(mesh, bc);
    DG::Function<P,N> u = poisson.solve(f);

    return 0;
}

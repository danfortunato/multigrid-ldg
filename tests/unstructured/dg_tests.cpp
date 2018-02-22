#include <iostream>
#include "common.h"
#include "wireframe.h"
#include "mesh.h"
#include "master.h"

int main(int argc, char* argv[])
{
    const int N = 3;   // Dimension
    const int p = 3;   // Polynomial order
    const int P = p+1; // Number of nodes per dimension

    DG::Wireframe<N> wireframe("data/ball.mesh");
    std::cout << "Points = " << wireframe.np << ", Triangles = " << wireframe.nt << std::endl;

    DG::Mesh<N,P> mesh(wireframe);

    double vol=0, area=0;
    for (int i=0; i<mesh.ne; ++i) {
        vol += mesh.elements[i].volume();
    }
    for (int i=0; i<mesh.nf; ++i) {
        if (mesh.faces[i].boundaryQ()) {
            area += mesh.faces[i].area();
        }
    }
    std::cout << "Volume = " << vol << std::endl;
    std::cout << "Surface area = " << area << std::endl;

    double qv=0, qa=0;
    for (const double& w : DG::Quadrature<N,P>::weights) qv += w;
    for (const double& w : DG::Quadrature<N-1,P>::weights) qa += w;

    vol=0, area=0;
    for (int i=0; i<mesh.ne; ++i) {
        double J = mesh.elements[i].simplex.volume();
        vol += J * qv;
    }
    for (int i=0; i<mesh.nf; ++i) {
        if (mesh.faces[i].boundaryQ()) {
            double J = mesh.faces[i].simplex.volume();
            area += J * qa;
        }
    }
    std::cout << "Volume (q) = " << vol << std::endl;
    std::cout << "Surface area (q) = " << area << std::endl;

    return 0;
}

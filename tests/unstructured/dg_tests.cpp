#include <iostream>
#include "common.h"
#include "range.h"
#include "ndarray.h"
#include "wireframe.h"

int main(int argc, char* argv[])
{
    const int N = 3;   // Dimension
    const int p = 3;   // Polynomial order
    const int P = p+1; // Number of nodes per dimension

    DG::SimplexArray<DG::Tuple<double,N>,N,P> nodes;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        *it = it.index().reverse().cast<double>()/(P-1);
        std::cout << (*it).format(DG::TupleFormat) << std::endl;
    }
    std::cout << std::endl;

    DG::Wireframe<2> wireframe("data/circle.mesh");
    std::cout << wireframe.np << " " << wireframe.nt << std::endl << std::endl;

    for (int i=0; i<wireframe.np; ++i) {
        std::cout << wireframe.p[i].format(DG::TupleFormat) << std::endl;
    }
    std::cout << std::endl;

    for (int i=0; i<wireframe.nt; ++i) {
        std::cout << wireframe.t[i].format(DG::TupleFormat) << std::endl;
    }

    return 0;
}

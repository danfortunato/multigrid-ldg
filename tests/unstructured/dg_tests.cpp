#include <iostream>
#include "common.h"
#include "range.h"
#include "ndarray.h"
#include "wireframe.h"
#include "master.h"

int main(int argc, char* argv[])
{
    const int N = 3;   // Dimension
    const int p = 3;   // Polynomial order
    const int P = p+1; // Number of nodes per dimension

    DG::Wireframe<N> wireframe("data/ball.mesh");
    std::cout << wireframe.np << " " << wireframe.nt << std::endl << std::endl;

    for (int i=0; i<wireframe.nt; ++i) {
        DG::Tuple<int,N+1> ti = wireframe.t[i];
        std::array<DG::Tuple<double,N>,N+1> p;
        for (int j=0; j<N+1; ++j) {
            p[j] = wireframe.p[ti[j]];
        }
        DG::Simplex<N> simplex(p);
        std::cout << "Simplex " << i << ": ";
        for (int j=0; j<N+1; ++j) {
            std::cout << simplex.p[j].format(DG::TupleFormat) << " ";
        }
        std::cout << std::endl;
        for (DG::SimplexRangeIterator<N,P> it; it != DG::SimplexRange<N,P>::end(); ++it) {
            std::cout << DG::Master<N,P>::dgnodes(it.linearIndex(), simplex).format(DG::TupleFormat) << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}

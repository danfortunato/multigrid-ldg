#include <iostream>
#include "common.h"
#include "range.h"
#include "ndarray.h"

int main(int argc, char* argv[])
{
    const int N = 3;   // Dimension
    const int p = 3;   // Polynomial order
    const int P = p+1; // Number of nodes per dimension

    DG::SimplexArray<DG::Tuple<double,N>,N,P> nodes;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        *it = it.index().reverse().cast<double>()/(P-1);
        std::cout << *it << std::endl;
    }

    return 0;
}

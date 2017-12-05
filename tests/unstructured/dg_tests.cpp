#include <iostream>
#include "common.h"
#include "range.h"

int main(int argc, char* argv[])
{
    const int N = 3;   // Dimension
    const int p = 3;   // Polynomial order
    const int P = p+1; // Number of nodes per dimension

    for (DG::SimplexRangeIterator<N,P> it; it != DG::SimplexRange<N,P>::end(); ++it) {
        for (int i=N-1; i>=0; --i) {
           std::cout << (double)it(i)/(P-1) << " ";
        }
        std::cout << it.linearIndex() << std::endl;
    }

    return 0;
}

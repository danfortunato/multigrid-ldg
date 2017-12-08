#ifndef MASTER_H
#define MASTER_H

#include <array>
#include <vector>
#include "common.h"
#include "range.h"
#include "ndarray.h"
#include "wireframe.h"

namespace DG
{
    /** @brief The master simplex */
    template<int N, int P>
    struct Master
    {
        /** Order of polynomial */
        static const int p = P-1;
        /** Number of nodes per element */
        static const int npl = ichoose(P+N-1,N);
        /** The local DG nodes */
        static const SimplexArray<Tuple<double,N>,N,P> nodes;
        /** The linearIndex-th node in a given simplex */
        static Tuple<double,N> dgnodes(double linearIndex, const Simplex<N>& simplex = Simplex<N>())
        {
            assert(0 <= linearIndex && linearIndex < npl);
            Tuple<double,N> local = nodes(linearIndex);
            return simplex.p[0].matrix() + simplex.jacobian_mat()*local.matrix();
        }
    };

    /** @brief Equispaced nodes on a simplex */
    template<int N, int P>
    SimplexArray<Tuple<double,N>,N,P> simplexNodes()
    {
        SimplexArray<Tuple<double,N>,N,P> nodes;
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            *it = it.index().reverse().template cast<double>() / (P-1);
        }
        return nodes;
    }

    template<int N, int P>
    const SimplexArray<Tuple<double,N>,N,P> Master<N,P>::nodes = simplexNodes<N,P>();
}

#endif

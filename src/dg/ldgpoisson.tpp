#include <stdexcept>
#include "range.h"

namespace DG
{
    template<int P, int N>
    void LDGPoisson<P,N>::discretize()
    {
        construct_mass_matrix();
        construct_broken_gradient();
        construct_lifting_terms();
    }

    /** @brief Construct the block diagonal mass matrices */
    template<int P, int N>
    void LDGPoisson<P,N>::construct_mass_matrix()
    {
        for (const auto& e : mesh->elements) {
            int j = e.lid;
            M_builder.setBlock(j, j, e.mass());
            for (int i=0; i<N; ++i) {
                int k = N*j+i;
                MM_builder.setBlock(k, k, e.mass());
            }
        }
    }

    /** @brief Construct the broken gradient operator */
    template<int P, int N>
    void LDGPoisson<P,N>::construct_broken_gradient()
    {
        KronMat<P,N> diff;
        // Create the differentiation matrix in each dimension
        for (int i=0; i<N; ++i) {
            diff = KronMat<P,N>::Zero();
            // Compute the tensor product of the 1D differentiation matrix
            // with the appropriate identity matrices
            for (RangeIterator<P,N> it; it != Range<P,N>::end(); ++it) {
                for (RangeIterator<P,N> jt; jt != Range<P,N>::end(); ++jt) {
                    // There is only a nonzero entry when the indices for all
                    // the identity matrices are on the diagonal
                    bool nonzero = true;
                    for (int k=0; k<N; ++k) {
                        if (k!=i && it(k)!=jt(k)) {
                            nonzero = false;
                            break;
                        }
                    }
                    if (nonzero) {
                        int bi = it.linearIndex();
                        int bj = jt.linearIndex();
                        diff(bi,bj) = GaussLobatto<P>::diff(it(i),jt(i));
                    }
                }
            }

            // Place this gradient component in the global matrix for all
            // elements
            for (const auto& e : mesh->elements) {
                int j = e.lid;
                double jac = 1/e.cell.width(i);
                G_builder.addToBlock(N*j+i, j, jac*diff);
            }
        }
    }

    /** @brief Construct the lifting operator and penalty terms */
    template<int P, int N>
    void LDGPoisson<P,N>::construct_lifting_terms()
    {
        FaceQuadMat<P,Q,N> L, R;
        KronMat<P,N> LL, RR, RL, MinvR, MinvL;

        for (const auto& f : mesh->faces) {

            // Compute the lifting matrices
            mesh->lift(f, L, R, LL, RR, RL);

            // Since the normals are the coordinate vectors, the only nonzero
            // component of the normal dot product is the in dimension this face
            // lives on
            double n = f.normal[f.dim];
            MinvL = mesh->elements[f.left].invmass();
            MinvR = mesh->elements[f.right].invmass();

            if (f.interiorQ()) {
                // Lifting operator
                G_builder.addToBlock(N*f.right+f.dim, f.right, n * MinvR * RR);
                G_builder.addToBlock(N*f.right+f.dim, f.left, -n * MinvR * RL);
                // Penalty terms
                T_builder.addToBlock(f.right, f.right,  tau0 * RR);
                T_builder.addToBlock(f.left,  f.left,   tau0 * LL);
                T_builder.addToBlock(f.right, f.left,  -tau0 * RL);
                T_builder.addToBlock(f.left,  f.right, -tau0 * RL.transpose());
            } else {
                switch (bcs.bcmap.at(f.boundary()).type) {

                    case kDirichlet:
                    {
                        // Lifting operator
                        G_builder.addToBlock(N*f.left+f.dim, f.left, -n * MinvL * LL);
                        // Penalty terms
                        T_builder.addToBlock(f.left, f.left, tauD * LL);
                        break;
                    }
                    case kNeumann:
                    {
                        const auto& hfun = bcs.bcmap.at(f.boundary()).f;
                        // Evaluate hfun at the quadrature points
                        KronVec<Q,N-1> h;
                        for (RangeIterator<Q,N-1> it; it != Range<Q,N-1>::end(); ++it) {
                            Tuple<double,N> q;
                            for (int k=0; k<N-1; ++k) {
                                double x = Quadrature<Q>::nodes[it(k)];
                                int kk = k + (k>=f.dim);
                                q[kk] = f.cell.lower[kk] + f.cell.width(kk)*x;
                            }
                            q[f.dim] = f.cell.lower[f.dim];
                            int i = it.linearIndex();
                            h[i] = hfun(q);
                        }
                        Jh.segment<npl>(npl*f.left) += MinvL * L * h;
                        break;
                    }
                    default:
                        throw std::invalid_argument("Unknown boundary condition.");
                }
            }
        }
    }

    template<int P, int N>
    void LDGPoisson<P,N>::assemble_system()
    {
        M = M_builder.build();
        G = G_builder.build();
        T = T_builder.build();

        // Create the discrete Laplacian
        // A = G^T M G + T ...
    }
}

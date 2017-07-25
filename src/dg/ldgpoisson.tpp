#include <stdexcept>
#include "range.h"
#include "krylov.h"
#include "timer.h"

namespace DG
{
    /** @brief Discretize the Poisson operator on the mesh using LDG
     *
     *  The discretization will construct matrices for the gradient,
     *  lifting, and penalty terms and assemble the global linear system
     *  to be solved.
     */
    template<int P, int N>
    void LDGPoisson<P,N>::discretize()
    {
        Timer::tic();
        construct_mass_matrix();
        construct_broken_gradient();
        construct_lifting_terms();
        construct_laplacian();
        Timer::toc("Discretize");
    }

    /** @brief Construct the block diagonal mass matrices */
    template<int P, int N>
    void LDGPoisson<P,N>::construct_mass_matrix()
    {
        Timer::tic();

        for (const auto& e : mesh->elements) {
            int j = e.lid;
            M_builder.setBlock(j, j, e.mass());
            for (int i=0; i<N; ++i) {
                int k = N*j+i;
                MM_builder.setBlock(k, k, e.mass());
            }
        }

        Timer::toc("Construct mass matrix");
    }

    /** @brief Construct the broken gradient operator */
    template<int P, int N>
    void LDGPoisson<P,N>::construct_broken_gradient()
    {
        Timer::tic();

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

        Timer::toc("Construct broken gradient");
    }

    /** @brief Construct the lifting operator and penalty terms */
    template<int P, int N>
    void LDGPoisson<P,N>::construct_lifting_terms()
    {
        Timer::tic();

        KronMat<P,N> LL, RR, RL, MinvR, MinvL;

        for (const auto& f : mesh->faces) {

            // Compute the lifting matrices
            mesh->lift(f, nullptr, nullptr, &LL, &RR, &RL);

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
            } else if (bcs.bcmap.at(f.boundary()).type == kDirichlet) {
                // Lifting operator
                G_builder.addToBlock(N*f.left+f.dim, f.left, -n * MinvL * LL);
                // Penalty terms
                T_builder.addToBlock(f.left, f.left, tauD * LL);
            }
        }

        Timer::toc("Construct lifting terms");
    }

    /** @brief Construct the discrete Laplacian operator */
    template<int P, int N>
    void LDGPoisson<P,N>::construct_laplacian()
    {
        Timer::tic();

        Timer::tic();
        MM = MM_builder.build();
        G  = G_builder.build();
        T  = T_builder.build();
        Timer::toc("Compress matrices");

        // Compute the discrete Laplacian: A = G^T M G + T
        Timer::tic();
        SparseBlockMatrix<npl> temp;
        multiply_mm_t(G, MM, A);
        multiply_mm(A, G, temp);
        add_mm(1.0, temp, T, A);
        Timer::toc("Matrix arithmetic");

        Timer::toc("Construct Laplacian");
    }

    /** @brief Compute the contribution of the boundary conditions to the RHS */
    template<int P, int N>
    void LDGPoisson<P,N>::add_source_terms()
    {
        Timer::tic();

        FaceQuadMat<P,Q,N> L;
        KronVec<Q,N-1> bc;

        for (const auto& f : mesh->faces) {

            if (f.boundaryQ()) {

                // Compute the lifting matrix
                mesh->lift(f, &L, nullptr, nullptr, nullptr, nullptr);
                double n = f.normal[f.dim];

                // Evaluate the boundary condition at the quadrature points
                const auto& bcfun = bcs.bcmap.at(f.boundary()).f;
                for (RangeIterator<Q,N-1> it; it != Range<Q,N-1>::end(); ++it) {
                    Tuple<double,N> q;
                    for (int k=0; k<N-1; ++k) {
                        double x = Quadrature<Q>::nodes[it(k)];
                        int kk = k + (k>=f.dim);
                        q[kk] = f.cell.lower[kk] + f.cell.width(kk)*x;
                    }
                    q[f.dim] = f.cell.lower[f.dim];
                    int i = it.linearIndex();
                    bc[i] = bcfun(q);
                }

                switch (bcs.bcmap.at(f.boundary()).type) {
                    case kDirichlet:
                        // Dirichlet contribution to RHS
                        Jg.segment<npl>(npl*(N*f.left+f.dim)) += n * L * bc;
                        // Dirichlet penalty contribution to RHS
                        rhs.vec(f.left) += tauD * L * bc;
                        break;
                    case kNeumann:
                        // Neumann contribution to RHS
                        rhs.vec(f.left) += L * bc;
                        break;
                    default:
                        throw std::invalid_argument("Unknown boundary condition.");
                }
            }
        }

        Timer::toc("Add source terms");
    }

    /** @brief Solve the linear system
     *
     *  @pre discretize() must be called before solve().
     *
     *  @param[in] f : The forcing function
     */
    template<int P, int N>
    Function<P,N> LDGPoisson<P,N>::solve(Function<P,N>& f)
    {
        // Compute the contribution of the boundary conditions to the RHS
        add_source_terms();

        // Add the forcing function and Dirichlet contribution to the RHS
        Timer::tic();
        M = M_builder.build();
        multiply_add_mv(   1.0, M,  f.data(), 1.0, rhs.data());
        multiply_add_mv_t(-1.0, G, Jg.data(), 1.0, rhs.data());
        Timer::toc("Computing RHS");

        // Solve using PCG
        Function<P,N> u(*mesh);
        auto uvec = u.vec();
        auto rvec = rhs.vec();
        pcg(A, rvec, uvec);

        return u;
    }

}

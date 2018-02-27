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
    template<int N, int P>
    void LDGPoisson<N,P>::discretize()
    {
        Timer::tic();
        construct_mass_matrix();
        construct_broken_gradient();
        construct_lifting_terms();
        ops_->construct_laplacian();
        add_source_terms();
        Timer::toc("Discretize");
    }

    /** @brief Construct the block diagonal mass matrices */
    template<int N, int P>
    void LDGPoisson<N,P>::construct_mass_matrix()
    {
        Timer::tic();

        for (const auto& e : mesh->elements) {
            int j = e.lid;
            ops_->M.setBlock(j, j, e.mass());
            ops_->Minv.emplace_back(e.mass());
        }

        Timer::toc("Construct mass matrix");
    }

    /** @brief Construct the broken gradient operator */
    template<int N, int P>
    void LDGPoisson<N,P>::construct_broken_gradient()
    {
        Timer::tic();

        for (int i=0; i<N; ++i) {
            // Create the differentiation matrix in each dimension
            KronMat<N,P> diff = Master<N,P>::diff(i);
            // Place this gradient component in the global matrix for all
            // elements
            for (const auto& e : mesh->elements) {
                int j = e.lid;
                double jac = 1.0/e.cell.width(i);
                ops_->G[i].addToBlock(j, j, jac*diff);
            }
        }

        Timer::toc("Construct broken gradient");
    }

    /** @brief Construct the lifting operator and penalty terms */
    template<int N, int P>
    void LDGPoisson<N,P>::construct_lifting_terms()
    {
        Timer::tic();

        KronMat<N,P> LL, RR, RL, MinvR, MinvL;

        for (const auto& f : mesh->faces) {

            // Compute the lifting matrices
            mesh->lift(f, nullptr, nullptr, &LL, &RR, &RL);

            // Since the normals are the coordinate vectors, the only nonzero
            // component of the normal dot product is in the dimension this face
            // lives on
            double n = f.normal[f.dim];

            if (f.interiorQ()) {
                MinvR = mesh->elements[f.right].invmass();
                // Lifting operator
                ops_->G[f.dim].addToBlock(f.right, f.right, n * MinvR * RR);
                ops_->G[f.dim].addToBlock(f.right, f.left, -n * MinvR * RL);
                // Penalty terms
                ops_->T.addToBlock(f.right, f.right,  tau0 * RR);
                ops_->T.addToBlock(f.left,  f.left,   tau0 * LL);
                ops_->T.addToBlock(f.right, f.left,  -tau0 * RL);
                ops_->T.addToBlock(f.left,  f.right, -tau0 * RL.transpose());
            } else if (bcs.bcmap.at(f.boundary()).type == kDirichlet) {
                MinvL = mesh->elements[f.left].invmass();
                // Lifting operator
                ops_->G[f.dim].addToBlock(f.left, f.left, -n * MinvL * LL);
                // Penalty terms
                ops_->T.addToBlock(f.left, f.left, tauD * LL);
            }
        }

        Timer::toc("Construct lifting terms");
    }

    /** @brief Compute the contribution of the boundary conditions to the RHS */
    template<int N, int P>
    void LDGPoisson<N,P>::add_source_terms()
    {
        Timer::tic();

        FaceQuadMat<N,P,Q> L;
        KronVec<N-1,Q> bc;

        for (const auto& f : mesh->faces) {

            if (f.boundaryQ()) {

                // Compute the lifting matrix
                mesh->lift(f, &L, nullptr, nullptr, nullptr, nullptr);
                double n = f.normal[f.dim];

                // Evaluate the boundary condition at the quadrature points
                const auto& bcfun = bcs.bcmap.at(f.boundary()).f;
                for (RangeIterator<N-1,Q> it; it != Range<N-1,Q>::end(); ++it) {
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
                        Jg[f.dim].template segment<npl>(npl*f.left) += n * L * bc;
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

        // Dirichlet contribution to the RHS
        for (const auto& e : mesh->elements) {
            int elem = e.lid;
            for (int d=0; d<N; ++d) {
                for (int i : ops_->G[d].rowsInCol(elem)) {
                    rhs.vec(elem) -= ops_->G[d].getBlock(i, elem).transpose() * Jg[d].template segment<npl>(npl*i);
                }
            }
        }

        Timer::toc("Add source terms");
    }

    /** @brief Add the forcing function to the RHS
     *
     *  @param[in] f : The forcing function
     */
    template<int N, int P>
    Function<N,P> LDGPoisson<N,P>::computeRHS(Function<N,P>& f)
    {
        Function<N,P> ff(*mesh);
        for (const auto& e : mesh->elements) {
            int elem = e.lid;
            ff.vec(elem) = rhs.vec(elem) + e.mass()*f.vec(elem);
        }
        return ff;
    }
}

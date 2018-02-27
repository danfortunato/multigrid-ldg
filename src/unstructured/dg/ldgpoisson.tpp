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

        for (const auto& e : mesh->elements) {
            int k = e.lid;
            Mat<N> Jinv = e.simplex.jacobian_mat().inverse();
            for (int i=0; i<N; ++i) {
                for (int j=0; j<N; ++j) {
                    ops_->G[i].addToBlock(k, k, Jinv(j,i)*Master<N,P>::diff[j]);
                }
            }
        }

        Timer::toc("Construct broken gradient");
    }

    /** @brief Construct the lifting operator and penalty terms */
    template<int N, int P>
    void LDGPoisson<N,P>::construct_lifting_terms()
    {
        Timer::tic();

        SimplexMat<N,P> LL, RR, RL, MinvR, MinvL;

        for (const auto& f : mesh->faces) {

            // Compute the lifting matrices
            mesh->lift(f, nullptr, nullptr, &LL, &RR, &RL);

            if (f.interiorQ()) {
                MinvR = mesh->elements[f.right].invmass();
                // Lifting operator
                for (int i=0; i<N; ++i) {
                    double n = f.normal[i];
                    ops_->G[i].addToBlock(f.right, f.right, n * MinvR * RR);
                    ops_->G[i].addToBlock(f.right, f.left, -n * MinvR * RL);
                }
                // Penalty terms
                ops_->T.addToBlock(f.right, f.right,  tau0 * RR);
                ops_->T.addToBlock(f.left,  f.left,   tau0 * LL);
                ops_->T.addToBlock(f.right, f.left,  -tau0 * RL);
                ops_->T.addToBlock(f.left,  f.right, -tau0 * RL.transpose());
            } else if (bcs.bcmap.at(f.boundary()).type == kDirichlet) {
                MinvL = mesh->elements[f.left].invmass();
                // Lifting operator
                for (int i=0; i<N; ++i) {
                    double n = f.normal[i];
                    ops_->G[i].addToBlock(f.left, f.left, -n * MinvL * LL);
                }
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

        SimplexFaceQuadMat<N,P,Q> L;
        Vec<Quadrature<N-1,Q>::size> bc;

        for (const auto& f : mesh->faces) {

            if (f.boundaryQ()) {

                // Compute the lifting matrix
                mesh->lift(f, &L, nullptr, nullptr, nullptr, nullptr);

                // Evaluate the boundary condition at the quadrature points
                const auto& bcfun = bcs.bcmap.at(f.boundary()).f;
                for (int i=0; i<Quadrature<N-1,Q>::size; ++i) {
                    Tuple<double,N> q = f.simplex.p[0].matrix() + f.simplex.jacobian_mat()*Quadrature<N-1,Q>::nodes[i].matrix();
                    bc[i] = bcfun(q);
                }

                switch (bcs.bcmap.at(f.boundary()).type) {
                    case kDirichlet:
                        // Dirichlet contribution to RHS
                        for (int i=0; i<N; ++i) {
                            double n = f.normal[i];
                            Jg[i].template segment<npl>(npl*f.left) += n * L * bc;
                        }
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

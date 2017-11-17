#ifndef LDG_POISSON_H
#define LDG_POISSON_H

#include <fstream>
#include <limits>
#include <memory>
#include "common.h"
#include "mesh.h"
#include "boundaryconditions.h"
#include "function.h"
#include "sparseblockmatrix.h"

namespace DG
{
    template<int P, int N>
    struct LDGOperators
    {
        /** @brief The number of nodal points per element */
        static const int npl = Master<P,N>::npl;
        /** @brief The type of the matrix blocks */
        typedef typename SparseBlockMatrix<npl>::Block Block;
        /** @brief Mass matrix */
        SparseBlockMatrix<npl> M;
        /** @brief The Cholesky decomposition for mass matrix */
        std::vector<Eigen::LDLT<Block>> Minv;
        /** @brief Discrete gradient */
        std::array<SparseBlockMatrix<npl>,N> G;
        /** @brief Penalty parameters */
        SparseBlockMatrix<npl> T;
        /** @brief Discrete Laplacian */
        SparseBlockMatrix<npl> A;

        /** @brief Construct the discrete Laplacian operator: A = G^T M G + T */
        void construct_laplacian()
        {
            Timer::tic();
            A = T;
            KronMat<P,N> GT_M, GT_M_G;
            for (int d = 0; d < N; ++d) {
                for (int k = 0; k < M.blockRows(); ++k) {
                    const auto& cols = G[d].colsInRow(k);
                    for (int i : cols) {
                        GT_M = (G[d].getBlock(k, i).transpose() * M.getBlock(k, k)).eval();
                        for (int j : cols) {
                            GT_M_G = GT_M * G[d].getBlock(k, j);
                            A.addToBlock(i, j, GT_M_G);
                        }
                    }
                }
            }
            Timer::toc("Construct Laplacian");
        }
    };

    template<int P, int N>
    class LDGPoisson
    {
        public:
            /** @brief Constructor from mesh, boundary conditions, and penalty parameters */
            LDGPoisson(const Mesh<P,N>& mesh_, BoundaryConditions<P,N> bcs_, double tau0_ = 1, double tauD_ = 1) :
                mesh(&mesh_),
                bcs(bcs_),
                tau0(tau0_),
                tauD(tauD_),
                ops_(std::make_shared<LDGOperators<P,N>>()),
                rhs(mesh_, 0)
            {
                // Reset the matrices to be the correct size
                ops_->M.reset(mesh->ne, mesh->ne);
                ops_->T.reset(mesh->ne, mesh->ne);
                for (int d=0; d<N; ++d) {
                    ops_->G[d].reset(mesh->ne, mesh->ne);
                    Jg[d].setZero(mesh->ne * npl);
                }

                discretize();
            }

            /** @brief Discretize the Poisson operator on the mesh using LDG
             *  
             *  The discretization will construct matrices for the gradient,
             *  lifting, and penalty terms and assemble the global linear system
             *  to be solved.
             */
            void discretize();

            /** @brief Add the forcing function to the RHS
             *
             *  @param[in] f : The forcing function
             */
            Function<P,N> computeRHS(Function<P,N>& f);

            /** @brief Get the operators */
            std::shared_ptr<LDGOperators<P,N>> ops()
            {
                return ops_;
            }

            /** @brief Dump the matrices */
            void dump()
            {
                ops_->A.write("data/A.mtx");
                ops_->M.write("data/M.mtx");
                ops_->T.write("data/T.mtx");
                for (int i=0; i<N; ++i) {
                    ops_->G[i].write("data/G" + std::to_string(i) + ".mtx");
                }
                rhs.write("data/rhs.fun");
            }

        private:
            void construct_mass_matrix();
            void construct_broken_gradient();
            void construct_lifting_terms();
            void construct_laplacian();
            void add_source_terms();

            /** @brief The mesh on which to solve */
            const Mesh<P,N>* mesh;
            /** @brief The boundary conditions */
            BoundaryConditions<P,N> bcs;
            /** @brief The penalty parameters */
            double tau0, tauD;
            /** @brief The order of the polynomial on each element */
            static const int p = P-1;
            /** @brief The number of nodal points per element */
            static const int npl = Master<P,N>::npl;
            /** @brief The LDG operators */
            std::shared_ptr<LDGOperators<P,N>> ops_;
            /** @brief Dirichlet contribution to RHS */
            std::array<Vector,N> Jg;
            /** @brief Right-hand side */
            Function<P,N> rhs;
    };
}

#include "ldgpoisson.tpp"

#endif

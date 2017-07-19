#ifndef LDG_POISSON_H
#define LDG_POISSON_H

#include "common.h"
#include "mesh.h"
#include "boundaryconditions.h"
#include "function.h"
#include "sparseblockmatrix.h"

namespace DG
{
    template<int P, int N>
    class LDGPoisson
    {
        public:
            /** @brief Constructor from mesh, boundary conditions, and penalty parameters */
            LDGPoisson(const Mesh<P,N>& mesh_, BoundaryConditions<P,N> bcs_, double tau0_ = 1, double tauD_ = 1) :
                mesh(&mesh_),
                bcs(bcs_),
                tau0(tau0_),
                tauD(tauD_)
            {
                // Reset the builders to be the correct size
                M_builder.reset(mesh->ne, mesh->ne);
                MM_builder.reset(N * mesh->ne, N * mesh->ne);
                G_builder.reset(N * mesh->ne, mesh->ne);
                T_builder.reset(mesh->ne, mesh->ne);

                discretize();
            }

            /** @brief Discretize the Poisson operator on the mesh using LDG
             *  
             *  The discretization will construct matrices for the gradient,
             *  lifting, and penalty terms and assemble the global linear system
             *  to be solved.
             */
            void discretize();

            /** @brief Solve the LDG linear system
             *
             *  @note discretize() must be called before solve().
             */
            Function<P,N> solve(Function<P,N>& f);

            /** @brief Dump the matrices */
            void dump()
            {
                M_builder.write("M.mtx");
                MM_builder.write("MM.mtx");
                G_builder.write("G.mtx");
                T_builder.write("T.mtx");
            }

        private:
            void construct_mass_matrix();
            void construct_broken_gradient();
            void construct_lifting_terms();
            void assemble_system();

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
            /** @brief Mass matrix */
            SparseBlockMatrix<npl> M;
            /** @brief Mass matrix (for vectors) */
            SparseBlockMatrix<npl> MM;
            /** @brief Discrete gradient */
            SparseBlockMatrix<npl> G;
            /** @brief Penalty parameters */
            SparseBlockMatrix<npl> T;
            /** @brief Discrete Laplacian */
            SparseBlockMatrix<npl> A;
            /** @brief Helper for building the sparse block matrices */
            SparseBlockMatrixBuilder<npl> M_builder, MM_builder, G_builder, T_builder;
    };
}

#include "ldgpoisson.tpp"

#endif

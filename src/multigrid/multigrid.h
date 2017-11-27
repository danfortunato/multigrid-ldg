#ifndef MULTIGRID_H
#define MULTIGRID_H

#include <memory>
#include <stdexcept>
#include "common.h"
#include "ldgpoisson.h"
#include "sparseblockmatrix.h"
#include "interpolationhierarchy.h"

namespace DG
{
    // Forward declarations
    template<int N, int P>
    struct Level;
    template<int N, int P, int... Ps>
    class Multigrid;

    /** @brief The relaxation method */
    enum Relaxation
    {
        kJacobi,
        kGaussSeidel
    };

    /** @brief The solver for the coarse problem */
    enum Solver
    {
        kCholesky,
        kRelaxation
    };

    /** @brief The parameters to use in multigrid */
    struct Parameters
    {
        /** @brief The relaxation method */
        Relaxation relaxation = kGaussSeidel;
        /** @brief The relaxation damping factor */
        double omega = 0.8;
        /** @brief The solver for the coarse problem */
        Solver solver = kRelaxation;
        /** @brief The number of pre-smooths */
        int npre = 3;
        /** @brief The number of post-smooths */
        int npost = 3;
    };

    template<int N, int P>
    using OperatorsPtr = std::shared_ptr<LDGOperators<N,P>>;
    template<int N, int P>
    using LevelPtr = std::shared_ptr<Level<N,P>>;

    /** @brief A level in the multigrid hierarchy */
    template<int N, int P>
    struct Level
    {
        /** @brief The number of nodal points per element */
        static const int npl = Master<N,P>::npl;

        typedef typename SparseBlockMatrix<npl>::Block Block;

        /** @brief Construct a level directly from a set of operators
         *
         *  This is useful for constructing the fine level, where no computation
         *  is needed.
         *
         *  @param[in] fineOps : The set of operators for this level
         *  @param[in] params_ : The parameters
         *  @param[in] num_    : The level number
         */
        Level(const OperatorsPtr<N,P>& fineOps, const Parameters& params_ = Parameters(), int num_ = 0) :
            params(params_),
            num(num_)
        {
            ops = fineOps;
            x.resize(ops->A.cols());
            b.resize(ops->A.cols());
            r.resize(ops->A.cols());

            // Compute the Cholesky decomposition of the block diagonal of A
            for (int i=0; i<ops->A.blockRows(); ++i) {
                Dinv.emplace_back(ops->A.getBlock(i, i));
            }
        }

        /** @brief Construct a level by coarsening a set of operators
         *
         *  @param[in] fineOps : The set of operators to coarsen
         *  @param[in] T       : The interpolation operator
         *  @param[in] params_ : The parameters
         *  @param[in] num_    : The level number
         */
        template<int P1>
        Level(const OperatorsPtr<N,P1>& fineOps, const InterpolationOperator<N,P1,P>& T, const Parameters& params_, int num_) :
            params(params_),
            num(num_)
        {
            ops = std::make_shared<LDGOperators<N,P>>();
            x.resize(T.cols());
            b.resize(T.cols());
            r.resize(T.cols());

            // Coarse mass matrix: M_c = T^T M_f T
            ops->M.reset(T.blockCols(), T.blockCols());
            KronMat<N,P,P1> TT_M;
            KronMat<N,P> TT_M_T;
            for (int k = 0; k < fineOps->M.blockRows(); ++k) {
                const auto& cols = T.colsInRow(k);
                for (int i : cols) {
                    TT_M = (T.getBlock(k, i).transpose() * fineOps->M.getBlock(k, k)).eval();
                    for (int j : cols) {
                        TT_M_T = TT_M * T.getBlock(k, j);
                        ops->M.addToBlock(i, j, TT_M_T);
                    }
                }
            }

            // Compute the Cholesky decomposition of M_c
            for (int i = 0; i < ops->M.blockRows(); ++i) {
                ops->Minv.emplace_back(ops->M.getBlock(i, i));
            }

            // Coarse gradient: G_c = M_c^{-1} T^T M_f G_f T
            SparseBlockMatrix<Master<N,P1>::npl,npl> temp1, temp2;
            for (int d=0; d<N; ++d) {
                multiply_mm(fineOps->G[d], T, temp1);  // P1 x P2
                multiply_mm(fineOps->M, temp1, temp2); // P1 x P2
                multiply_mm_t(T, temp2, ops->G[d]);    // P2 x P2
                for (int i=0; i<ops->M.blockRows(); ++i) {
                    const auto& cols = ops->G[d].colsInRow(i);
                    for (int j : cols) {
                        ops->G[d].setBlock(i, j, ops->Minv[i].solve(ops->G[d].getBlock(i, j)));
                    }
                }
            }

            // Coarse penalty parameters: T_c = T^T T_f T
            multiply_mm(fineOps->T, T, temp1); // P1 x P2
            multiply_mm_t(T, temp1, ops->T);   // P2 x P2

            // Coarse Laplacian: A = G_c^T M_c G_c + T_c
            ops->construct_laplacian();

            // Compute the Cholesky decomposition of the block diagonal of A
            for (int i=0; i < ops->A.blockRows(); ++i) {
                Dinv.emplace_back(ops->A.getBlock(i, i));
            }
        }

        /** @brief Compute the residual: r = b - A x */
        void computeResidual()
        {
            r = b;
            multiply_add_mv(-1.0, ops->A, x, 1.0, r);
        }

        /** @brief Solve the system */
        void solve()
        {
            switch (params.solver) {
                case kRelaxation:
                    relax(params.npre);
                    relax(params.npost, true);
                    break;
                case kCholesky:
                    assert(ops->A.blockRows() == 1); // Ensure there is only one element
                    x = Dinv[0].solve(b);
                    break;
                default:
                    throw std::invalid_argument("Unknown coarse solver.");
            }
        }

        /** @brief Perform a number of relaxations
         *
         *  @param[in] n       : The number of relaxations to do
         *  @param[in] reverse : Flag to process the elements in reverse order
         */
        void relax(int n = 1, bool reverse = false)
        {
            for (int i=0; i<n; ++i) {
                switch (params.relaxation) {
                    case kJacobi:
                        jacobi(params.omega);
                        break;
                    case kGaussSeidel:
                        gauss_seidel(params.omega, reverse);
                        break;
                    default:
                        throw std::invalid_argument("Unknown relaxation method.");
                }
            }
        }

        /** @brief Perform a (weighted) Jacobi iteration:
         *         x_{n+1} = x_{n} + w D^{-1} (b - A x_{n})
         *
         *  @param[in] omega : The weight
         */
        void jacobi(double omega = 1.0)
        {
            computeResidual();
            for (int i=0; i < ops->A.blockRows(); ++i) {
                x.segment<npl>(npl*i) += omega * Dinv[i].solve(r.segment<npl>(npl*i));
            }
        }

        /** @brief Perform a (weighted) Gauss-Seidel iteration (i.e. SSOR)
         *
         *  @param[in] omega   : The weight
         *  @param[in] reverse : Flag to process the elements in reverse order
         */
        void gauss_seidel(double omega = 1.0, bool reverse = false)
        {
            Vec<npl> temp;
            if (!reverse) {
                for (int i=0; i<ops->A.blockRows(); ++i) {
                    temp.setZero();
                    const auto& cols = ops->A.colsInRow(i);
                    for (int j : cols) {
                        if (j!=i) {
                            temp += ops->A.getBlock(i, j) * x.segment<npl>(npl*j);
                        }
                    }
                    x.segment<npl>(npl*i) += omega * (Dinv[i].solve(b.segment<npl>(npl*i) - temp) - x.segment<npl>(npl*i));
                }
            } else {
                for (int i=ops->A.blockRows()-1; i>=0; --i) {
                    temp.setZero();
                    const auto& cols = ops->A.colsInRow(i);
                    for (int j : cols) {
                        if (j!=i) {
                            temp += ops->A.getBlock(i, j) * x.segment<npl>(npl*j);
                        }
                    }
                    x.segment<npl>(npl*i) += omega * (Dinv[i].solve(b.segment<npl>(npl*i) - temp) - x.segment<npl>(npl*i));
                }
            }
        }

        /** @brief The operators for this level */
        std::shared_ptr<LDGOperators<N,P>> ops;
        /** @brief The Cholesky decomposition for the block diagonal */
        std::vector<Eigen::LDLT<Block>> Dinv;
        /** @brief The solution */
        Vector x;
        /** @brief The right-hand side */
        Vector b;
        /** @brief The residual */
        Vector r;
        /** @brief The multigrid parameters */
        Parameters params;
        /** @brief The level number */
        int num;
    };

    /** @brief An h-multigrid solver */
    template<int N, int P>
    class Multigrid<N,P>
    {
        public:
            /** @brief Constructor
             *
             *  @param[in] ops       : The LDG discretization on the fine level
             *  @param[in] hierarchy : The hierarchy of h-interpolation operators
             *  @param[in] params    : The multigrid parameters
             */
            Multigrid(const OperatorsPtr<N,P>& ops, const InterpolationHierarchy<N,P>& hierarchy, const Parameters& params = Parameters()) :
                hierarchy_(&hierarchy),
                params_(params)
            {
                // Create the fine level
                auto level = std::make_shared<Level<N,P>>(ops, params_, 0);
                levels_.push_back(level);

                // Set up the rest
                setup();
            }

            /** @brief Constructor
             *
             *  @param[in] level     : The fine level
             *  @param[in] hierarchy : The hierarchy of h-interpolation operators
             *  @param[in] params    : The multigrid parameters
             */
            Multigrid(const LevelPtr<N,P>& level, const InterpolationHierarchy<N,P>& hierarchy, const Parameters& params = Parameters()) :
                hierarchy_(&hierarchy),
                params_(params)
            {
                // Add the fine level
                levels_.push_back(level);

                // Set up the rest
                setup();
            }

            /** @brief The solution */
            const Vector& solution() const
            {
                return levels_[0]->x;
            }

            /** @brief The solution */
            Vector& solution()
            {
                return levels_[0]->x;
            }

            /** @brief The right-hand side */
            const Vector& rhs() const
            {
                return levels_[0]->b;
            }

            /** @brief The right-hand side */
            Vector& rhs()
            {
                return levels_[0]->b;
            }

            /** @brief Compute and return the residual */
            Vector residual()
            {
                levels_[0]->computeResidual();
                return levels_[0]->r;
            }

            /** @brief The i-th level in the hierarchy */
            LevelPtr<N,P> level(int i = 0)
            {
                return levels_[i];
            }

            /** @brief Perform a V-cycle starting at a specific level
             *
             *  @param[in] i : The level from which to start the V-cycle
             */
            void vcycle(int i = 0)
            {
                assert(i>=0 && i<(int)levels_.size());

                if (i == (int)levels_.size()-1) {
                    // Coarse solve
                    levels_[i]->b.array() -= levels_[i]->b.mean();
                    levels_[i]->solve();
                } else {
                    // Pre-smooth
                    levels_[i]->relax(params_.npre);

                    // Coarse-grid correction
                    levels_[i]->computeResidual();
                    restrictResidual(i);
                    levels_[i+1]->x.setZero();
                    vcycle(i+1);
                    correctSolution(i);

                    // Post-smooth
                    levels_[i]->relax(params_.npost, true);
                }
            }

        private:
            /** @brief Setup up the h-multigrid hierarchy */
            void setup()
            {
                int start = levels_[0]->num;
                for (int i=0; i < hierarchy_->size(); ++i) {
                    auto level = std::make_shared<Level<N,P>>(levels_[i]->ops, hierarchy_->T[i], params_, start+i+1);
                    levels_.push_back(level);
                }
            }

            /** @brief Restrict the residual from level i to level i+1 */
            void restrictResidual(int i = 0)
            {
                assert(i>=0 && i<hierarchy_->size());
                multiply_mv_t(hierarchy_->T[i], levels_[i]->r, levels_[i+1]->b);
            }

            /** @brief Correct the solution on level i with an update from level i+1 */
            void correctSolution(int i = 0)
            {
                assert(i>=0 && i<(int)levels_.size()-1);
                multiply_add_mv(1.0, hierarchy_->T[i], levels_[i+1]->x, 1.0, levels_[i]->x);
            }

            /** @brief The sequence of interpolation operators that define the multigrid hierarchy */
            const InterpolationHierarchy<N,P>* hierarchy_;
            /** @brief The levels in the multigrid hierarchy */
            std::vector<LevelPtr<N,P>> levels_;
            /** @brief The multigrid parameters */
            Parameters params_;
    };

    /** @brief An hp- or p-multigrid solver */
    template<int N, int P1, int P2, int... Ps>
    class Multigrid<N,P1,P2,Ps...>
    {
        public:
            /* @brief Constructor
             *
             *  @param[in] ops       : The LDG discretization on the fine level
             *  @param[in] hierarchy : The hierarchy of hp-interpolation operators
             *  @param[in] params    : The multigrid parameters
             */
            Multigrid(const OperatorsPtr<N,P1>& ops, const InterpolationHierarchy<N,P1,P2,Ps...>& hierarchy, const Parameters& params = Parameters()) :
                hierarchy_(&hierarchy),
                params_(params),
                below_(std::make_shared<Level<N,P2>>(ops, hierarchy.T, params, 1), hierarchy.below, params)
            {
                // Create the fine level
                level_ = std::make_shared<Level<N,P1>>(ops, params_, 0);
            }

            /* @brief Constructor
             *
             *  @param[in] level     : The fine level
             *  @param[in] hierarchy : The hierarchy of hp-interpolation operators
             *  @param[in] params    : The multigrid parameters
             */
            Multigrid(const LevelPtr<N,P1>& level, const InterpolationHierarchy<N,P1,P2,Ps...>& hierarchy, const Parameters& params = Parameters()) :
                hierarchy_(&hierarchy),
                params_(params),
                below_(std::make_shared<Level<N,P2>>(level->ops, hierarchy.T, params, level->num+1), hierarchy.below, params),
                level_(level)
            {}

            /** @brief The solution */
            const Vector& solution() const
            {
                return level_->x;
            }

            /** @brief The solution */
            Vector& solution()
            {
                return level_->x;
            }

            /** @brief The right-hand side */
            const Vector& rhs() const
            {
                return level_->b;
            }

            /** @brief The right-hand side */
            Vector& rhs()
            {
                return level_->b;
            }

            /** @brief Compute and return the residual */
            Vector residual()
            {
                level_->computeResidual();
                return level_->r;
            }

            /** @brief The current level in the hierarchy */
            LevelPtr<N,P1> level()
            {
                return level_;
            }

            /** @brief Perform a V-cycle starting at a specific level */
            void vcycle()
            {
                // Pre-smooth
                level_->relax(params_.npre);

                // Coarse-grid correction
                level_->computeResidual();
                restrictResidual();
                below_.level()->x.setZero();
                below_.vcycle();
                correctSolution();

                // Post-smooth
                level_->relax(params_.npost, true);
            }

        private:
            /** @brief Restrict the residual from level P1 to level P2 */
            void restrictResidual()
            {
                multiply_mv_t(hierarchy_->T, level_->r, below_.level()->b);
            }

            /** @brief Correct the solution on level P1 with an update from level P2 */
            void correctSolution()
            {
                multiply_add_mv(1.0, hierarchy_->T, below_.level()->x, 1.0, level_->x);
            }

            /** @brief The sequence of hp- or p-interpolation operators that define the multigrid hierarchy */
            const InterpolationHierarchy<N,P1,P2,Ps...>* hierarchy_;
            /** @brief The multigrid parameters */
            Parameters params_;
            /** @brief The rest of the multigrid hierarchy */
            Multigrid<N,P2,Ps...> below_;
            /** @brief The current level in the hierarchy */
            LevelPtr<N,P1> level_;
    };

    /** @brief A halted, p-multigrid-only solver */
    template<int N, int P>
    class Multigrid<N,P,StopCoarsening>
    {
        public:
            Multigrid(const LevelPtr<N,P>& level, const InterpolationHierarchy<N,P,StopCoarsening>& hierarchy, const Parameters& params = Parameters()) :
                hierarchy_(&hierarchy),
                params_(params),
                level_(level)
            {}

            // The coarsest V-cycle is just a coarse solve
            void vcycle()
            {
                level_->b.array() -= level_->b.mean();
                level_->solve();
            }

            LevelPtr<N,P> level() { return level_; }

        private:
            const InterpolationHierarchy<N,P,StopCoarsening>* hierarchy_;
            Parameters params_;
            LevelPtr<N,P> level_;
    };
}

#endif

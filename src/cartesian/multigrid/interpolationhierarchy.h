#ifndef INTERPOLATION_HIERARCHY_H
#define INTERPOLATION_HIERARCHY_H

#include <unordered_map>
#include "common.h"
#include "quadtree.h"
#include "master.h"

namespace DG
{
    /** @brief A flag to halt the hierarchy to be p-only */
    static const int StopCoarsening = -1;

    template<int N, int P, int Q = P>
    using InterpolationOperator = SparseBlockMatrix<ipow(P,N),ipow(Q,N)>;

    // Forward declaration
    template<int N, int P, int... Ps>
    class InterpolationHierarchy;

    /** @brief A hierarchy of h-interpolation operators */
    template<int N, int P>
    class InterpolationHierarchy<N,P>
    {
        public:
            /** @brief Constructor
             *
             *  @param[in] qt : The quadtree defining the geometric hierarchy
             */
            InterpolationHierarchy(const Quadtree<N>& qt) :
                qt_(&qt)
            {
                assert(qt_->numLevels() > 1);

                // Loop from fine to coarse to construct the h-interpolation
                // operator from level l+1 to l
                for (int l=0; l<qt_->numLevels()-1; ++l) {
                    // Get the fine and coarse levels in the tree
                    std::vector<int> finelevel   = qt_->layer(l);
                    std::vector<int> coarselevel = qt_->layer(l+1);

                    std::unordered_map<int,int> coarseG2L;
                    int coarselid = 0;
                    for (auto& coarsegid : coarselevel) {
                        coarseG2L[coarsegid] = coarselid++;
                    }

                    // Construct a sparse block matrix with fine-level number of
                    // rows and coarse-level number of columns
                    T.emplace_back(finelevel.size(), coarselevel.size());

                    // Loop over the fine-level bounding boxes
                    int finelid = 0;
                    for (int finegid : finelevel) {
                        // Is this element also on the coarse level?
                        if (qt_->isLeaf((*qt_)[finegid],l+1)) {
                            T[l].setBlock(finelid, coarseG2L[finegid], KronMat<N,P>::Identity());
                            finelid++;
                        } else {
                            // Get the coarse-level parent of this fine-level
                            // bounding box
                            int coarsegid = (*qt_)[finegid].parent;

                            // Compute the fine-level bounding box when the
                            // parent bounding box is mapped to [0,1]^N
                            Cell<N> relcell;
                            relcell.lower = ((*qt_)[finegid].cell.lower - (*qt_)[coarsegid].cell.lower) / (*qt_)[coarsegid].cell.width();
                            relcell.upper = ((*qt_)[finegid].cell.upper - (*qt_)[coarsegid].cell.lower) / (*qt_)[coarsegid].cell.width();

                            // Construct a matrix that evaluates the parent
                            // polynomial at the fine-level nodes
                            KronMat<N,P> E;
                            for (RangeIterator<N,P> it; it != Range<N,P>::end(); ++it) {
                                Tuple<double,N> finenode = Master<N,P>::dgnodes(it.index(), relcell);
                                int i = it.linearIndex();
                                E.row(i) = LagrangePoly<P>::eval(finenode);
                            }
                            T[l].setBlock(finelid++, coarseG2L[coarsegid], E);
                        }
                    }
                }
            }

            /** @brief The number of levels in the hierarchy */
            int size() const
            {
                return T.size();
            }

            /** @brief The h-interpolation operators */
            std::vector<InterpolationOperator<N,P>> T;

        private:
            /** @brief The quadtree */
            const Quadtree<N>* qt_;
    };

    /** @brief A hierarchy of hp- or p-interpolation operators */
    template<int N, int P1, int P2, int... Ps>
    class InterpolationHierarchy<N,P1,P2,Ps...>
    {
        static_assert(P1 > P2, "The p-multigrid hierarchy is ill-formed.");

        public:
            /** @brief Constructor
             *
             *  @param[in] qt : The quadtree defining the geometry
             */
            InterpolationHierarchy(const Quadtree<N>& qt) :
                below(qt),
                qt_(&qt)
            {
                std::vector<int> finelevel = qt_->layer(0);
                T.reset(finelevel.size(), finelevel.size());

                // Construct a matrix that evaluates the degree-P2 polynomial
                // at the degree-P1 nodes
                KronMat<N,P1,P2> E;
                for (RangeIterator<N,P1> it; it != Range<N,P1>::end(); ++it) {
                    Tuple<double,N> finenode = Master<N,P1>::dgnodes(it.index());
                    int i = it.linearIndex();
                    E.row(i) = LagrangePoly<P2>::eval(finenode);
                }

                // Loop over the fine-level bounding boxes
                for (int i=0; i<(int)finelevel.size(); ++i) {
                    T.setBlock(i, i, E);
                }
            }

            /** @brief The number of levels in the hierarchy */
            int size() const
            {
                return 1 + below.size();
            }

            /** @brief The p-interpolation operator */
            InterpolationOperator<N,P1,P2> T;
            /** @brief The rest of the hierarchy */
            InterpolationHierarchy<N,P2,Ps...> below;

        private:
            /** @brief The quadtree */
            const Quadtree<N>* qt_;
    };

    /** @brief A halted, p-only interpolation hierarchy */
    template<int N, int P>
    class InterpolationHierarchy<N,P,StopCoarsening>
    {
        public:
            InterpolationHierarchy(const Quadtree<N>& qt) {}
            int size() const { return 1; }
    };
}

#endif

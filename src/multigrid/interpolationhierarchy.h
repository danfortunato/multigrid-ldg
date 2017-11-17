#ifndef INTERPOLATION_HIERARCHY_H
#define INTERPOLATION_HIERARCHY_H

#include <unordered_map>
#include "common.h"
#include "quadtree.h"
#include "master.h"

namespace DG
{
    /** @brief A hierarchy of interpolation operators */
    template<int N, int P>
    class InterpolationHierarchy
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

                // Loop from fine to coarse to construct the interpolation
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

            /** @brief The number of levels in the hierarchy */
            int size() const
            {
                return T.size();
            }

            /** @brief The interpolation operators */
            std::vector<SparseBlockMatrix<Master<N,P>::npl>> T;

        private:
            /** @brief The quadtree */
            const Quadtree<N>* qt_;
    };
}

#endif

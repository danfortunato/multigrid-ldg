#ifndef INTERPOLATION_HIERARCHY_H
#define INTERPOLATION_HIERARCHY_H

#include "common.h"
#include "sparseblockmatrix.h"
#include "range.h"
#include "geometry.h"
#include "agglomeration.h"
#include "master.h"
#include "triangletree.h"

namespace DG
{
    template<int nplf, int nplc = nplf>
    using InterpolationOperator = SparseBlockMatrix<nplf,nplc>;

    /** @brief A hierarchy of h-interpolation operators */
    template<int N, int P>
    class InterpolationHierarchy
    {
        public:
            /** @brief Construct from a nested mesh hierarchy
             *
             *  @param[in] tt : The triangle tree defining the geometric hierarchy
             */
            InterpolationHierarchy(const TriangleTree& tt) :
                tt_(&tt)
            {
                assert(N == 2 && "Mesh refinement only works in 2D.");
                assert(tt_->numLevels() > 1);

                // Loop from fine to coarse to construct the h-interpolation
                // operator from level l+1 to l
                for (int l=0; l<tt_->numLevels()-1; ++l) {
                    // Get the fine and coarse levels in the tree
                    std::vector<int> finelevel   = tt_->layer(l);
                    std::vector<int> coarselevel = tt_->layer(l+1);

                    std::unordered_map<int,int> coarseG2L;
                    int coarselid = 0;
                    for (auto& coarsegid : coarselevel) {
                        coarseG2L[coarsegid] = coarselid++;
                    }

                    // Construct a sparse block matrix with fine-level number of
                    // rows and coarse-level number of columns
                    if (l==0) {
                        T0.reset(finelevel.size(), coarselevel.size());
                    } else {
                        T.emplace_back(finelevel.size(), coarselevel.size());
                    }

                    // Loop over the fine-level triangles
                    int finelid = 0;
                    for (int finegid : finelevel) {
                        // Get the coarse-level parent of this fine-level
                        // triangle
                        int coarsegid = (*tt_)[finegid].parent;

                        // Compute the fine-level triangle when the coarse
                        // triangle is mapped to the master element
                        Simplex<N> relsimplex;
                        Mat<N> Jinv = (*tt_)[coarsegid].simplex.jacobian_mat().inverse();
                        for (int i=0; i<N+1; ++i) {
                            relsimplex.p[i] = Jinv * ((*tt_)[finegid].simplex.p[i] - (*tt_)[coarsegid].simplex.p[0]).matrix();
                        }

                        SimplexMat<N,P> E;
                        for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                            int i = it.linearIndex();
                            Tuple<double,N> finenode = Master<N,P>::dgnodes(i, relsimplex);
                            E.row(i) = Phi<N,P>::eval(finenode);
                        }

                        if (l==0) {
                            T0.setBlock(finelid++, coarseG2L[coarsegid], E);
                        } else {
                            T[l-1].setBlock(finelid++, coarseG2L[coarsegid], E);
                        }
                    }
                }
            }

            /** @brief Construct from an agglomeration
             *
             *  @param[in] agg : The agglomeration
             */
            InterpolationHierarchy(const Agglomeration<N,P>& agg) :
                agg_(&agg),
                mesh_(agg_->mesh)
            {
                // Build the operators
                assert(agg_->numAgglom() > 0);

                // Construct the finest h-interpolation operator
                T0.reset(mesh_->ne, agg_->aggloms[0].size());

                // Loop over the coarse agglomerated elements
                for (const AgglomeratedElement<N,P>& coarse_elem : agg_->aggloms[0].elements) {
                    // Loop over the fine elements that make up this coarse
                    // element
                    for (int finelid : coarse_elem.orig_elems) {
                        const Element<N,P>& fine_elem = mesh_->elements[finelid];
#ifdef BOUNDING_SIMPLEX
                        // Compute the fine-level simplex when the coarse
                        // simplex is mapped to the master element
                        Simplex<N> relsimplex;
                        Mat<N> Jinv = coarse_elem.bounding_simplex.jacobian_mat().inverse();
                        for (int i=0; i<N+1; ++i) {
                            relsimplex.p[i] = Jinv * (fine_elem.simplex.p[i] - coarse_elem.bounding_simplex.p[0]).matrix();
                        }

                        SimplexMat<N,P> E;
                        for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                            int i = it.linearIndex();
                            Tuple<double,N> finenode = Master<N,P>::dgnodes(i, relsimplex);
                            E.row(i) = Phi<N,P>::eval(finenode);
                        }
#else
                        // Compute the fine-level simplex when the coarse
                        // bounding box is mapped to [0,1]^N
                        Simplex<N> relsimplex;
                        for (int i=0; i<N+1; ++i) {
                            relsimplex.p[i] = (fine_elem.simplex.p[i] - coarse_elem.bounding_box.lower) / coarse_elem.bounding_box.width();
                        }

                        // Construct a matrix that evaluates the coarse
                        // polynomial at the fine-level simplex nodes
                        KronToSimplexMat<N,P> E;
                        for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                            int i = it.linearIndex();
                            Tuple<double,N> finenode = Master<N,P>::dgnodes(i, relsimplex);
                            E.row(i) = Cartesian::LagrangePoly<P>::eval(finenode);
                        }
#endif
                        
                        T0.setBlock(finelid, coarse_elem.lid, E);
                    }
                }

                // Loop from fine to coarse to construct the rest of the
                // h-interpolation operators from level l+1 to l
                for (int l=0; l<agg_->numAgglom()-1; ++l) {
                    // Get the fine and coarse levels in the tree
                    const AgglomeratedMesh<N,P>& finelevel   = agg_->aggloms[l];
                    const AgglomeratedMesh<N,P>& coarselevel = agg_->aggloms[l+1];

                    // Construct a sparse block matrix with fine-level number of
                    // rows and coarse-level number of columns
                    T.emplace_back(finelevel.size(), coarselevel.size());

                    // Loop over the coarse agglomerated elements
                    for (const AgglomeratedElement<N,P>& coarse_elem : coarselevel.elements) {
                        // Does the coarse element equal the fine element?
                        if (coarse_elem.finer_elems.size() == 1) {
                            int finelid = coarse_elem.finer_elems[0];
#ifdef BOUNDING_SIMPLEX
                            T[l].setBlock(finelid, coarse_elem.lid, SimplexMat<N,P>::Identity());
#else
                            T[l].setBlock(finelid, coarse_elem.lid, KronMat<N,P>::Identity());
#endif
                        } else {
                            // Loop over the fine elements that make up this coarse
                            // element
                            for (int finelid : coarse_elem.finer_elems) {
                                const AgglomeratedElement<N,P>& fine_elem = finelevel.elements[finelid];
#ifdef BOUNDING_SIMPLEX
                                // Compute the fine-level simplex when the coarse
                                // simplex is mapped to the master element
                                Simplex<N> relsimplex;
                                Mat<N> Jinv = coarse_elem.bounding_simplex.jacobian_mat().inverse();
                                for (int i=0; i<N+1; ++i) {
                                    relsimplex.p[i] = Jinv * (fine_elem.bounding_simplex.p[i] - coarse_elem.bounding_simplex.p[0]).matrix();
                                }

                                SimplexMat<N,P> E;
                                for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                                    int i = it.linearIndex();
                                    Tuple<double,N> finenode = Master<N,P>::dgnodes(i, relsimplex);
                                    E.row(i) = Phi<N,P>::eval(finenode);
                                }
#else
                                // Compute the fine-level bounding box when the
                                // coarse bounding box is mapped to [0,1]^N
                                Cell<N> relcell;
                                relcell.lower = (fine_elem.bounding_box.lower - coarse_elem.bounding_box.lower) / coarse_elem.bounding_box.width();
                                relcell.upper = (fine_elem.bounding_box.upper - coarse_elem.bounding_box.lower) / coarse_elem.bounding_box.width();

                                // Construct a matrix that evaluates the coarse
                                // polynomial at the fine-level nodes
                                KronMat<N,P> E;
                                for (RangeIterator<N,P> it; it != Range<N,P>::end(); ++it) {
                                    Tuple<double,N> finenode = Cartesian::Master<N,P>::dgnodes(it.index(), relcell);
                                    int i = it.linearIndex();
                                    E.row(i) = Cartesian::LagrangePoly<P>::eval(finenode);
                                }
#endif
                                
                                T[l].setBlock(finelid, coarse_elem.lid, E);
                            }
                        }
                    }
                }
            }

            /** @brief The number of levels in the hierarchy */
            int size() const
            {
                return T.size()+1;
            }

            /** @brief The h-interpolation operators */
#ifdef BOUNDING_SIMPLEX
            InterpolationOperator<Master<N,P>::npl> T0;
            std::vector<InterpolationOperator<Master<N,P>::npl>> T;
#else
            InterpolationOperator<Cartesian::Master<N,P>::npl,Master<N,P>::npl> T0;
            std::vector<InterpolationOperator<Cartesian::Master<N,P>::npl>> T;
#endif

        private:
            /** @brief The agglomeration */
            const Agglomeration<N,P>* agg_;
            /** @brief The mesh */
            const Mesh<N,P>* mesh_;
            /** @brief The triangle tree */
            const TriangleTree* tt_;
    };
}

#endif

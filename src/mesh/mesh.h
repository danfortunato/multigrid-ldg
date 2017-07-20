#ifndef MESH_H
#define MESH_H

#include <vector>
#include <functional>
#include <unordered_map>
#include "common.h"
#include "function.h"
#include "quadtree.h"
#include "master.h"
#include "ndarray.h"
#include "range.h"

namespace DG
{
    /** @brief A face in the mesh */
    template<int P, int N>
    struct Face
    {
        Face(const Mesh<P,N>& mesh_, int left_, int right_, int dim_, Vec<N> normal_, Cell<N> cell_) :
            mesh(&mesh_),
            left(left_),
            right(right_),
            dim(dim_),
            normal(normal_),
            cell(cell_)
        {
            // Make sure this face is actually codimension one in the specified dimension
            bool codimension_one = ((cell.lower == cell.upper).count() == 1 &&
                                    (cell.lower[dim] == cell.upper[dim]));
            assert(codimension_one);

            // Boundary faces are canonical
            canonical = true;
            if (interiorQ()) {
                Cell<N> lcell = mesh->elements[left].cell;
                Cell<N> rcell = mesh->elements[right].cell;
                lcell.lower[dim] = lcell.upper[dim];
                rcell.upper[dim] = rcell.lower[dim];

                // Determine whether this is a canonical face
                canonical = lcell == rcell;

                if (!canonical) {
                    // Compute and store the quadrature points for this face with
                    // respect to the local coordinates of the left and right elements

                    // Let f be the mapping from the element's coordinate system to
                    // to [0,1]^N
                    Tuple<double,N> al = cell.lower - lcell.lower;
                    Tuple<double,N> bl = cell.upper - lcell.lower;
                    Tuple<double,N> ar = cell.lower - rcell.lower;
                    Tuple<double,N> br = cell.upper - rcell.lower;
                    for (int i=0; i<N; ++i) {
                        if (i!=dim) {
                            al[i] /= lcell.width(i);
                            bl[i] /= lcell.width(i);
                            ar[i] /= rcell.width(i);
                            br[i] /= rcell.width(i);
                        }
                    }
                    // Create the coordinate tensor product
                    for (RangeIterator<Q,N-1> it; it != Range<Q,N-1>::end(); ++it) {
                        Tuple<double,N> tl, tr;
                        for (int k=0; k<N-1; ++k) {
                            double x = Quadrature<Q>::nodes[it(k)];
                            int kk = k + (k>=dim);
                            tl[kk] = al[kk] + (bl[kk]-al[kk])*x;
                            tr[kk] = ar[kk] + (br[kk]-ar[kk])*x;
                        }
                        // Make degenerate coordinate match reference element
                        tl[dim] = 1;
                        tr[dim] = 0;
                        xl(it.index()) = tl;
                        xr(it.index()) = tr;
                    }
                }
            }
        }

        /** The area of the face */
        double area() const
        {
            double a = 1;
            for (int i=0; i<N; ++i) {
                if (i!=dim) {
                    a *= cell.width(i);
                }
            }
            return a;
        }

        /** The mass matrix for this face */
        KronMat<P,N-1> mass() const
        {
            double jac = area();
            return Master<P,N-1>::mass * jac;
        }

        bool interiorQ() const
        {
            return right >= 0;
        }

        bool boundaryQ() const
        {
            return right < 0;
        }

        int boundary() const
        {
            return boundaryQ() ? right : 0;
        }

        /** A pointer to the mesh */
        const Mesh<P,N>* mesh;
        /** Index of element on the "left"  (anti-normal direction) */
        int left;
        /** Index of element on the "right" (normal direction) */
        int right;
        /** The dimension this face lives in */
        int dim;
        /** Is this face canonical? */
        bool canonical;
        /** The normal vector */
        Vec<N> normal;
        /** The cell defining the face.
         *  Note: This should be codimension one! */
        Cell<N> cell;
        /** Local quadrature points with respect to the left element */
        NDArray<Tuple<double,N>,Q,N-1> xl;
        /** Local quadrature points with respect to the right element */
        NDArray<Tuple<double,N>,Q,N-1> xr;
    };

    /** @brief An element in the mesh */
    template<int P, int N>
    struct Element
    {
        Element(const Mesh<P,N>& mesh_, int lid_, int gid_, Cell<N> cell_) :
            mesh(&mesh_),
            lid(lid_),
            gid(gid_),
            cell(cell_)
        {}

        /** The volume of the element */
        double volume() const
        {
            return cell.volume();
        }

        /** The mass matrix for the element */
        KronMat<P,N> mass() const
        {
            double jac = volume();
            return Master<P,N>::mass * jac;
        }

        /** The inverse mass matrix for the element */
        KronMat<P,N> invmass() const
        {
            double jac = 1/volume();
            return Master<P,N>::invmass * jac;
        }

        /** The i-th node in the element */
        Tuple<double,N> dgnodes(Tuple<int,N> index) const
        {
            assert((0 <= index).all() && (index < P).all());
            Tuple<double,N> node;
            for (int i=0; i<N; ++i) {
                node[i] = GaussLobatto<P>::nodes[index[i]];
            }
            return cell.lower + cell.width() * node;
        }

        /** A pointer to the mesh */
        const Mesh<P,N>* mesh;
        /** The local ID of the element */
        int lid;
        /** The global ID of the element */
        int gid;
        /** The bounding box of the element */
        Cell<N> cell;
    };

    /** @brief A mesh */
    template<int P, int N>
    struct Mesh
    {
        // Can only construct from quadtrees
        Mesh() = delete;

        /** @brief Construct a mesh from the finest level of a quadtree */
        Mesh(const Quadtree<N>& qt, int coarsening = 0)
        {
            // Initialize geometric boundaries
            for (int i=0; i<N; ++i) {
                int index = 2*i+1;
                boundaryIndices.push_back(-index);
                boundaryIndices.push_back(-(index+1));
            }

            // Get the global IDs of the elements in the specified layer
            std::vector<int> ids = qt.layer(coarsening);

            // Construct the elements and the local/global ID maps
            L2G.resize(ids.size());
            for (auto& gid : ids) {
                int lid = elements.size();
                G2L[gid] = lid;
                L2G[lid] = gid;
                Element<P,N> elem(*this, lid, gid, qt[gid].cell);
                elements.push_back(elem);
            }

            nb = 0;
            // Construct the faces from the neighbors of each element
            for (auto& elem : elements) {
                for (int i=0; i<N; ++i) {
                    // The normal vector is the coordinate vector in dimension N
                    Vec<N> normal = Vec<N>::Zero();
                    normal[i] = 1;
                    // Look for left and right neighbors in each dimension
                    for (auto& dir : {kLeft, kRight}) {
                        std::vector<int> neighbors = qt.neighbors(elem.gid, i, dir, coarsening);
                        if (neighbors.empty()) {
                            // Add boundary face with outward pointing normal vector
                            int outward = (dir == kLeft) ? -1 : 1;
                            // Compute the bounding box of the face
                            Tuple<double,N> lower = elem.cell.lower;
                            Tuple<double,N> upper = elem.cell.upper;
                            if (dir == kLeft) upper[i] = lower[i]; else lower[i] = upper[i];
                            Cell<N> fcell(lower, upper);
                            // Compute the geometric boundary index
                            int bnd = findBoundary(fcell, dir, i);
                            // Create the face
                            Face<P,N> face(*this, elem.lid, bnd, i, outward * normal, fcell);
                            faces.push_back(face);
                            nb++;
                        } else {
                            // Add faces between me and my neighbors
                            for (auto& ngid : neighbors) {
                                // Get the local ID of my neighbor
                                int nlid = G2L[ngid];
                                // Make sure not to double count faces
                                if (elem.lid < nlid) {
                                    // Determine who is left and who is right
                                    int left  = (dir == kLeft)  ? nlid : elem.lid;
                                    int right = (dir == kRight) ? nlid : elem.lid;
                                    // Compute the bounding box of the face
                                    Tuple<double,N> lower = elements[left].cell.lower.max(elements[right].cell.lower);
                                    Tuple<double,N> upper = elements[left].cell.upper.min(elements[right].cell.upper);
                                    assert(elements[left].cell.upper[i] == elements[right].cell.lower[i]);
                                    lower[i] = elements[right].cell.lower[i]; // = These should be equal
                                    upper[i] = elements[left].cell.upper[i];  // =
                                    Cell<N> fcell(lower, upper);
                                    // Create the face
                                    Face<P,N> face(*this, left, right, i, normal, fcell);
                                    faces.push_back(face);
                                }
                            }
                        }
                    }
                }
            }

            ne = elements.size();
            nf = faces.size();
        }

        /** @brief Compute the geometric boundary index for a face */
        int findBoundary(Cell<N> fcell, Direction dir, int dim)
        {
            return boundaryIndices[2*dim + (dir==kRight)];
        }

        /** @brief The polynomial order */
        static const int p = P-1;
        /** @brief The number of nodes per element */
        static const int npl = Master<P,N>::npl;
        /** @brief The number of faces */
        int nf;
        /** @brief The number of elements */
        int ne;
        /** @brief The number of boundary faces */
        int nb;
        /** @brief An enumeration of the elements in the mesh */
        std::vector<Element<P,N>> elements;
        /** @brief An enumeration of the faces in the mesh */
        std::vector<Face<P,N>> faces;
        /** @brief Geometric boundary indices */
        std::vector<int> boundaryIndices;
        /** @brief Global-to-local element ID map */
        std::unordered_map<int,int> G2L;
        /** @brief Local-to-global element ID map */
        std::vector<int> L2G;
    };
}

#endif

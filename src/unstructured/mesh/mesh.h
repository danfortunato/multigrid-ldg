#ifndef MESH_H
#define MESH_H

#include <vector>
#include <array>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "common.h"
#include "wireframe.h"
#include "master.h"
#include "timer.h"
#include "function.h"

namespace DG
{
    // Forward declaration
    template<int N, int P>
    class Mesh;

    /** @brief A face in the mesh */
    template<int N, int P>
    struct Face
    {
        Face(const Mesh<N,P>& mesh, int left_, int right_, Vec<N> normal_, Simplex<N-1,N> simplex_) :
            left(left_),
            right(right_),
            normal(normal_),
            simplex(simplex_)
        {
            if (interiorQ()) {
                // Compute slicing matrices for left and right elements
                slice_l = SimplexSliceMat<N,P>::Zero();
                slice_r = SimplexSliceMat<N,P>::Zero();
                Simplex<N> simplex_l = mesh.elements[left].simplex;
                Simplex<N> simplex_r = mesh.elements[right].simplex;
                Mat<N> Jinv_l = simplex_l.jacobian_mat().inverse();
                Mat<N> Jinv_r = simplex_r.jacobian_mat().inverse();
                // Loop through face nodes
                for (SimplexRangeIterator<N-1,P> it; it != SimplexRange<N-1,P>::end(); ++it) {
                    int i = it.linearIndex();
                    Tuple<double,N-1> faceref = Master<N-1,P>::nodes(i);
                    Tuple<double,N> facephys = simplex.p[0].matrix() + simplex.jacobian_mat()*faceref.matrix();
                    Tuple<double,N> nl = Jinv_l * (facephys - simplex_l.p[0]).matrix();
                    Tuple<double,N> nr = Jinv_r * (facephys - simplex_r.p[0]).matrix();
                    // Loop through volume nodes to find a match
                    bool found_l = false, found_r = false;
                    for (SimplexRangeIterator<N,P> jt; jt != SimplexRange<N,P>::end(); ++jt) {
                        int j = jt.linearIndex();
                        if (!found_l && ((nl - Master<N,P>::nodes(j)).abs() < 1e-6).all()) {
                            slice_l(i,j) = 1;
                            found_l = true;
                        }
                        if (!found_r && ((nr - Master<N,P>::nodes(j)).abs() < 1e-6).all()) {
                            slice_r(i,j) = 1;
                            found_r = true;
                        }
                        if (found_l && found_r) break;
                    }
                }
            } else {
                // Compute quadrature nodes with respect to left element
                Simplex<N> simplex_l = mesh.elements[left].simplex;
                Mat<N> Jinv_l = simplex_l.jacobian_mat().inverse();
                for (int i=0; i<Quadrature<N-1,Q>::size; ++i) {
                    Tuple<double,N-1> faceref = Quadrature<N-1,Q>::nodes[i];
                    Tuple<double,N> facephys = simplex.p[0].matrix() + simplex.jacobian_mat()*faceref.matrix();
                    xl[i] = Jinv_l * (facephys - simplex_l.p[0]).matrix();
                }
            }
        }

        /** The area of the face */
        double area() const
        {
            return simplex.volume();
        }

        /** The mass matrix for this face */
        SimplexMat<N-1,P> mass() const
        {
            double jac = area();
            return Master<N-1,P>::mass * jac;
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
            return boundaryQ() ? -right : 0;
        }

        /** Index of element on the "left"  (anti-normal direction) */
        int left;
        /** Index of element on the "right" (normal direction) */
        int right;
        /** The normal vector */
        Vec<N> normal;
        /** The simplex defining the face */
        Simplex<N-1,N> simplex;
        /** Slicing vector of indices */
        SimplexSliceMat<N,P> slice_l, slice_r;
        /** Local quadrature points with respect to the left element */
        std::array<Tuple<double,N>,Quadrature<N-1,Q>::size> xl;
        /** Local quadrature points with respect to the right element */
        std::array<Tuple<double,N>,Quadrature<N-1,Q>::size> xr;
    };

    /** @brief An element in the mesh */
    template<int N, int P>
    struct Element
    {
        Element(int lid_, Simplex<N> simplex_) :
            lid(lid_),
            simplex(simplex_)
        {}

        /** The volume of the element */
        double volume() const
        {
            return simplex.volume();
        }

        /** The mass matrix for the element */
        SimplexMat<N,P> mass() const
        {
            double jac = volume();
            return Master<N,P>::mass * jac;
        }

        /** The inverse mass matrix for the element */
        SimplexMat<N,P> invmass() const
        {
            double jac = 1.0/volume();
            return Master<N,P>::invmass * jac;
        }

        /** The linearIndex-th node in the element */
        Tuple<double,N> dgnodes(const int linearIndex) const
        {
            return Master<N,P>::dgnodes(linearIndex, simplex);
        }

        /** The local ID of the element */
        int lid;
        /** The geometry of the element */
        Simplex<N> simplex;
    };

    /** @brief A hash function for an array */
    template<typename T, int N>
    struct ArrayHasher
    {
        inline void hash_combine(std::size_t& seed, const T& v) const
        {
            std::hash<T> hasher;
            seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }

        std::size_t operator()(const std::array<T,N>& x) const
        {
            std::size_t hash = 0;
            for (int i=0; i<N; ++i) {
                hash_combine(hash, x[i]);
            }
            return hash;
        }
    };

    /** @brief A mesh */
    template<int N, int P>
    struct Mesh
    {
        // Can only construct from wireframes
        Mesh() = delete;

        /** @brief Construct a mesh from a wireframe */
        Mesh(const Wireframe<N>& wireframe, const std::vector<std::function<bool(Tuple<double,N>)>>& bndfuns = {})
        {
            Timer::tic();
            // Construct the elements
            for (int i=0; i<wireframe.nt; ++i) {
                Tuple<int,N+1> t = wireframe.t[i];
                std::array<Tuple<double,N>,N+1> p;
                for (int j=0; j<N+1; ++j) {
                    p[j] = wireframe.p[t[j]];
                }
                Simplex<N> simplex(p);
                Element<N,P> elem(i, simplex);
                elements.push_back(elem);
            }
            Timer::toc("Construct elements");

            Timer::tic();
            // Construct the face-to-simplex lookup table
            std::unordered_map<std::array<int,N>, std::array<int,2>, ArrayHasher<int,N>> f2t;
            for (int i=0; i<wireframe.nt; ++i) {
                Tuple<int,N+1> t = wireframe.t[i];
                // Look on each face of the simplex
                for (int j=0; j<N+1; ++j) {
                    std::array<int,N> f;
                    for (int k=0; k<N; ++k) {
                        f[k] = t[(k+j)%(N+1)];
                    }
                    std::sort(f.begin(), f.end());
                    if (f2t.count(f)==0) {
                        f2t[f] = {i,-1};
                    } else {
                        f2t[f] = {f2t[f][0],i};
                    }
                }
            }

            nb = 0;
            ngb = bndfuns.size();
            for (auto it = f2t.begin(); it != f2t.end(); ++it) {
                // Get the left and right simplices
                int id1 = it->second[0];
                int id2 = it->second[1];
                int min = std::min(id1,id2);
                int max = std::max(id1,id2);
                int left  = (min>=0) ? min : max;
                int right = (min>=0) ? max : min;
                if (right < 0) nb++;

                // Construct the facial simplex
                std::array<int,N> f = it->first;
                std::array<Tuple<double,N>,N> p;
                for (int j=0; j<N; ++j) {
                    p[j] = wireframe.p[f[j]];
                }
                Simplex<N-1,N> fsimplex(p);

                // Compute the normal vector
                Vec<N> normal;
                Mat<N-1,N> cross = fsimplex.jacobian_mat().transpose();
                int sign = 1;
                for (int i=0; i<N; ++i) {
                    Mat<N-1,N-1> minor;
                    for (int j=0, d=0; j<N; ++j) {
                        if (j!=i) {
                            minor.col(d) = cross.col(j);
                            d++;
                        }
                    }
                    normal[i] = sign * minor.determinant();
                    sign *= -1;
                }
                normal /= normal.norm();

                // Flip the normal to point outward
                int a;
                for (int i=0; i<N+1; ++i) {
                    if (std::find(f.begin(), f.end(), wireframe.t[left][i]) == f.end()) {
                        a = wireframe.t[left][i];
                    }
                }
                Vec<N> inward = wireframe.p[a] - p[0];
                if (normal.dot(inward) > 0) normal *= -1;

                // Add the boundary indicator
                if (!bndfuns.empty() && right < 0) {
                    bool found = false;
                    int ind = 1;
                    for (const auto& bnd : bndfuns) {
                        if (std::all_of(fsimplex.p.begin(), fsimplex.p.end(), bnd)) {
                            found = true;
                            break;
                        }
                        ind++;
                    }
                    if (!found) throw std::invalid_argument("Boundary face does not satisfy any boundary function.");
                    right = -ind;
                }

                // Add the face
                Face<N,P> face(*this, left, right, normal, fsimplex);
                faces.push_back(face);
            }
            Timer::toc("Construct faces");

            ne = elements.size();
            nf = faces.size();
        }

        /** @brief Perform an L^2 projection of a function */
        Function<N,P> l2_project(const std::function<double(Tuple<double,N>)>& f)
        {
            Function<N,P> f_proj(*this);
            Vec<Quadrature<N,Q>::size> w(Quadrature<N,Q>::weights.data());
            for (int i=0; i<ne; ++i) {
                const Element<N,P>& e = elements[i];
                // Evaluate f at the quadrature points
                Vec<Quadrature<N,Q>::size> f_eval;
                for (int j=0; j<Quadrature<N,Q>::size; ++j) {
                    Tuple<double,N> q = e.simplex.p[0].matrix() + e.simplex.jacobian_mat()*Quadrature<N,Q>::nodes[j].matrix();
                    f_eval[j] = f(q);
                }
                f_proj.vec(i) = e.invmass() * Phi<N,P,Q>::phi * (e.volume * Master<N,P>::volume * w.asDiagonal()) * f_eval;
            }
            return f_proj;
        }

        /** @brief Compute the lifting matrices for a given face
         *
         *  @param[in]  f  : The face to lift from
         *  @param[out] L  : Test from the left
         *  @param[out] R  : Test from the right
         *  @param[out] LL : Test from the left, trial from the left
         *  @param[out] RR : Test from the right, trial from the right
         *  @param[out] RL : Test from the right, trial from the left
         */
        void lift(const Face<N,P>& f, SimplexFaceQuadMat<N,P,Q>* L, SimplexFaceQuadMat<N,P,Q>* R, SimplexMat<N,P>* LL, SimplexMat<N,P>* RR, SimplexMat<N,P>* RL) const
        {
            // Can we use the precomputed mass matrix?
            if (f.interiorQ()) {
                if (LL) *LL = f.slice_l.transpose() * f.mass() * f.slice_l;
                if (RR) *RR = f.slice_r.transpose() * f.mass() * f.slice_r;
                if (RL) *RL = f.slice_r.transpose() * f.mass() * f.slice_l;
            } else {
                // Compute matrices to evaluate at quadrature points
                SimplexSliceEvalMat<N,P,Q> phi_l;
                for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                    phi_l.col(it.linearIndex()) = koornwinder<N,Quadrature<N-1,Q>::size>(f.xl, it.index());
                }
                phi_l *= Master<N,P>::invvandermonde;

                // Scale the quadrature weights according to the area of the face
                Vec<Quadrature<N-1,Q>::size> w(Quadrature<N-1,Q>::weights.data());
                w *= f.area() * Master<N-1,P>::volume;

                if (L) *L = phi_l.transpose() * w.asDiagonal();
                if (LL) *LL = (L ? *L : phi_l.transpose() * w.asDiagonal()) * phi_l;
            }
        }

        /** @brief The polynomial order */
        static const int p = P-1;
        /** @brief The number of nodes per element */
        static const int npl = Master<N,P>::npl;
        /** @brief The number of faces */
        int nf;
        /** @brief The number of elements */
        int ne;
        /** @brief The number of boundary faces */
        int nb;
        /** @brief The number of geometric boundaries */
        int ngb;
        /** @brief An enumeration of the elements in the mesh */
        std::vector<Element<N,P>> elements;
        /** @brief An enumeration of the faces in the mesh */
        std::vector<Face<N,P>> faces;
    };
}

#endif

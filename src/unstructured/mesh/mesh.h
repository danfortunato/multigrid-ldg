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
        {}

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
    };

    /** @brief An element in the mesh */
    template<int N, int P>
    struct Element
    {
        Element(int id_, Simplex<N> simplex_) :
            id(id_),
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
            double jac = 1/volume();
            return Master<N,P>::invmass * jac;
        }

        /** The index-th node in the element */
        Tuple<double,N> dgnodes(const Tuple<int,N>& index) const
        {
            return Master<N,P>::dgnodes(index, simplex);
        }

        /** The global ID of the element */
        int id;
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

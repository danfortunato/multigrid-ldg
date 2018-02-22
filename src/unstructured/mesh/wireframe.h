#ifndef WIREFRAME_H
#define WIREFRAME_H

#include <vector>
#include <array>
#include <fstream>
#include <sstream>

namespace DG
{
    /** @brief An N-dimensional simplex (embedded in dimension D) */
    template<int N, int D = N>
    struct Simplex
    {
        static_assert(N <= D);

        /** @brief Construct a unit simplex */
        Simplex()
        {
            p[0] = Tuple<double,D>::Zero();
            for (int i=0; i<N; ++i) {
                p[i+1] = Tuple<double,D>::Zero();
                p[i+1][i] = 1;
            }
        }

        /** @brief Construct a simplex from the given points */
        Simplex(const std::array<Tuple<double,D>,N+1>& p_) :
            p(p_)
        {}

        /** @brief The Jacobian matrix */
        Mat<D,N> jacobian_mat() const
        {
            Mat<D,N> J;
            for (int i=0; i<N; ++i) {
                J.col(i) = p[i+1]-p[0];
            }
            return J;
        }

        /** @brief The Jacobian determinant */
        double jacobian_det() const
        {
            return jacobian_mat().determinant();
        }

        /** @brief The Gramian matrix */
        Mat<N> gramian_mat() const
        {
            Mat<N> G;
            Mat<D,N> J = jacobian_mat();
            for (int i=0; i<N; ++i) {
                for (int j=0; j<N; ++j) {
                    G(i,j) = J.col(i).dot(J.col(j));
                }
            }
            return G;
        }

        /** @brief The Gramian determinant */
        double gramian_det() const
        {
            return gramian_mat().determinant();
        }

        /** @brief The volume of the simplex
         *
         *  @note If the simplex is embedded, the volume is computed with
         *        respect to the intrinsic dimension N, not the embedded
         *        dimension D.
         */
        double volume() const
        {
            return std::sqrt(gramian_det()) / ifac(N);
        }

        /** @brief The points defining the simplex */
        std::array<Tuple<double,D>,N+1> p;
    };

    /** @brief A wireframe for an unstructured mesh */
    template<int N>
    struct Wireframe
    {
        /** @brief Construct a wireframe from a set of points and simplices */
        Wireframe(const std::vector<Tuple<double,N>>& p_, const std::vector<Tuple<int,N+1>>& t_) :
            np(p_.size()),
            nt(t_.size()),
            p(p_),
            t(t_)
        {}

        /** @brief Read in a wireframe from a file */
        Wireframe(const std::string& file)
        {
            std::ifstream ifs(file);
            std::stringstream ss;
            std::string line;

            if (ifs.is_open()) {

                // Read headers
                std::getline(ifs, line);
                ss.clear();
                ss.str(line);
                ss >> np >> nt;
                p.reserve(np);
                t.reserve(nt);

                // Read p
                std::getline(ifs, line);
                while (std::getline(ifs, line) && !line.empty()) {
                    ss.clear();
                    ss.str(line);
                    Tuple<double,N> point;
                    double val;
                    int i=0;
                    while (ss >> val) point[i++] = val;
                    assert(i == N);
                    p.push_back(point);
                }

                // Read t
                while (std::getline(ifs, line) && !line.empty()) {
                    ss.clear();
                    ss.str(line);
                    Tuple<int,N+1> tri;
                    int val;
                    int i=0;
                    while (ss >> val) tri[i++] = val;
                    assert(i == N+1);
                    t.push_back(tri-1); // Shift 1-index to 0-index
                }

                ifs.close();
            }
        }

        /** @brief The number of points */
        int np;
        /** @brief The number of simplices */
        int nt;
        /** @brief The points (np x N) */
        std::vector<Tuple<double,N>> p;
        /** @brief The simplices (nt x N+1) */
        std::vector<Tuple<int,N+1>> t;
    };
}

#endif

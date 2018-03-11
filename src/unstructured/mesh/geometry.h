#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <array>
#include "common.h"

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

    /** @brief An N-dimensional cell. The bounding box of the cell is specified
     *         by the coordinates of its lower left and upper right corners. */
    template<int N>
    struct Cell
    {
        /** @brief Construct a unit cell */
        Cell() :
            lower(Tuple<double,N>(0)),
            upper(Tuple<double,N>(1))
        {}

        /** @brief Construct a cell from bounding box coordinates */
        Cell(Tuple<double,N> lower_, Tuple<double,N> upper_) :
            lower(lower_),
            upper(upper_)
        {}

        Tuple<double,N> width() const {
            return upper-lower;
        }

        /** @brief Compute the width of the cell in the dimension d */
        double width(int d) const {
            if (d < 0 || d > N) {
                throw std::out_of_range("Requested dimension does not exist.");
            }
            return upper[d]-lower[d];
        }

        double maxWidth() const {
            double max = width(0);
            for (int i=1; i<N; ++i) {
                if (width(i) > max) {
                    max = width(i);
                }
            }
            return max;
        }

        double minWidth() const {
            double min = width(0);
            for (int i=1; i<N; ++i) {
                if (width(i) < min) {
                    min = width(i);
                }
            }
            return min;
        }

        /** @brief Compute the volume of the cell */
        double volume() const {
            double v = 1;
            for (int i=0; i<N; ++i) { v *= width(i); }
            return v;
        }

        /** @brief The lower left and upper right coordinates of the cell */
        Tuple<double,N> lower, upper;
    };

    template<int N>
    bool operator==(const Cell<N>& cell1, const Cell<N>& cell2)
    {
        return (cell1.lower == cell2.lower).all() &&
               (cell1.upper == cell2.upper).all();
    }
}

#endif

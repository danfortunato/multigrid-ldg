#ifndef COMMON_H
#define COMMON_H

#define EIGEN_NO_DEBUG
#define EIGEN_USE_MKL_ALL
#define MKL_ALIGN 64

#include <Eigen/Dense>   // Eigen
#include <mkl.h>         // MKL
#include <mkl_spblas.h>  // MKL routines
#include <array>         // std::array

namespace DG
{
    template<typename T>
    constexpr T ipow(T x, int n)
    {
        return ((size_t)n >= sizeof(int)*8) ? 0 : n == 0 ? 1 : x * ipow(x, n-1);
    }

    template<int P, int N>
    using Mat = Eigen::Matrix<double,ipow(P,N),ipow(P,N),Eigen::RowMajor>;

    template<int P, int N>
    using Vec = Eigen::Matrix<double,ipow(P,N),1,Eigen::RowMajor>;

    /** @brief An n-dimensional coordinate */
    template<int N>
    struct Coordinate
    {
        /** @brief Constructor from value */
        Coordinate(double v = 0)
        {
            for (int i = 0; i < N; ++i) {
                x[i] = v;
            }
        }

        /** @brief Constructor from values */
        Coordinate(std::array<double,N> v)
        {
            for (int i = 0; i < N; ++i) {
                x[i] = v[i];
            }
        }

        /** @brief Access operator */
        inline double& operator[] (int i) { return x[i]; }
        inline const double& operator[] (int i) const { return x[i]; }

        /** @brief Array of coordinate components */
        std::array<double,N> x;
    };

    /** @brief Equals operator */
    template<int N>
    bool operator==(const Coordinate<N>& p, const Coordinate<N>& q) {
        for (int i = 0; i < N; i++) {
            if (p[i] != q[i]) {
                return false;
            }
        }
        return true;
    }

    /** @brief Unequals operator */
    template<int N>
    bool operator!=(const Coordinate<N>& p, const Coordinate<N>& q) {
        return !(p == q);
    }

    /** @brief An n-dimensional cell. The bounding box of the cell is specified
     *         by the coordinates of its lower left and upper right corners.
     */
    template<int N>
    struct Cell
    {
        /** @brief Construct a unit cell */
        Cell() :
            lower(Coordinate<N>(0)),
            upper(Coordinate<N>(1))
        {}

        /** @brief Construct a cell from bounding box coordinates */
        Cell(Coordinate<N> lower_, Coordinate<N> upper_) :
            lower(lower_), upper(upper_)
        {}

        /** @brief Compute the width of the cell in the dimension d */
        double width(int d) {
            if (d >= N) {
                throw std::invalid_argument("Requested dimension too large.");
            }
            return upper[d]-lower[d];
        }

        /** @brief Compute the volume of the cell */
        double volume() {
            double v = 1;
            for (int i=0; i<N; ++i) { v *= width(i); }
            return v;
        }

        /** @brief The lower left and upper right coordinates of the cell */
        Coordinate<N> lower, upper;
    };
}

#endif
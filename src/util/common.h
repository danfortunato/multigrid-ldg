#ifndef COMMON
#define COMMON

#define EIGEN_NO_DEBUG
#define EIGEN_USE_MKL_ALL
#define MKL_ALIGN 64

#include <Eigen/Dense>   // Eigen
#include <mkl.h>         // MKL
#include <mkl_spblas.h>  // MKL routines

namespace DG
{
    template<typename T>
    constexpr T ipow(T x, int n)
    {
        return (n >= sizeof(int)*8) ? 0 : n == 0 ? 1 : x * ipow(x, n-1);
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
            Coordinate(double v = 0) { for (int i = 0; i < N; ++i) { x[i] = v; } }
            /** @brief Access operator */
            inline double& operator[] (int i) { return x[i]; }
            inline const double& operator[] (int i) const { return x[i]; }
            /** @brief Array of coordinate components */
            double x[N];
    };

    template<>
    struct Coordinate<2>
    {
        Coordinate(double v = 0) : x(v), y(v) {}
        Coordinate(double x_, double y_) : x(x_), y(y_) {}
        inline double& operator[] (int i) {
            if (i < 1 || i > 2) {
                throw std::out_of_range("Requested component does not exist.");
            }
            return i==1 ? x : y;
        }
        inline const double& operator[] (int i) const {
            if (i < 1 || i > 2) {
                throw std::out_of_range("Requested component does not exist.");
            }
            return i==1 ? x : y;
        }
        double x, y;
    };

    template<>
    struct Coordinate<3>
    {
        Coordinate(double v = 0) : x(v), y(v), z(v) {}
        Coordinate(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
        inline double& operator[] (int i) {
            if (i < 1 || i > 3) {
                throw std::out_of_range("Requested component does not exist.");
            }
            return i==1 ? x : (i==2 ? y : z);
        }
        inline const double& operator[] (int i) const {
            if (i < 1 || i > 3) {
                throw std::out_of_range("Requested component does not exist.");
            }
            return i==1 ? x : (i==2 ? y : z);
        }
        double x, y, z;
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
}

#endif
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
}

#endif
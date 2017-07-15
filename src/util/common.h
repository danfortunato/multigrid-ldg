#ifndef COMMON_H
#define COMMON_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_STACK_ALLOCATION_LIMIT 1000000
#define MKL_ALIGN 64

#include <Eigen/Dense>   // Eigen
#include <mkl.h>         // MKL
#include <mkl_spblas.h>  // MKL routines

namespace DG
{
    /** @brief The number of quadrature points to use */
    const int Q = 10;

    /** @brief Compile-time power function */
    template<typename T>
    inline constexpr T ipow(const T base, unsigned const exponent)
    {
        return (exponent == 0) ? 1 : (base * ipow(base, exponent-1));
    }

    /** Compile-time-sized matrix */
    template<int M, int N>
    using Mat = Eigen::Matrix<double,M,N,Eigen::RowMajor>;

    /** Compile-time-sized vector */
    template<int N>
    using Vec = Eigen::Matrix<double,N,1>;

    /** Compile-time-sized tuple */
    template<typename T, int N>
    using Tuple = Eigen::Array<T,N,1>;

    /** Compile-time-sized diagonal matrix */
    template<int N>
    using Diag = Eigen::DiagonalMatrix<double,N>;

    /** Compile-time-sized tensor product matrix */
    template<int P, int N>
    using KronMat = Eigen::Matrix<double,ipow(P,N),ipow(P,N),Eigen::RowMajor>;

    /** Compile-time-sized tensor product vector */
    template<int P, int N>
    using KronVec = Eigen::Matrix<double,ipow(P,N),1>;

    /** Compile-time-sized tensor product diagonal matrix */
    template<int P, int N>
    using KronDiag = Eigen::DiagonalMatrix<double,ipow(P,N)>;

    /** Compile-time-sized slicing matrix */
    template<int P, int N>
    using SliceMat = Eigen::Matrix<double,ipow(P,N-1),ipow(P,N),Eigen::RowMajor>;

    /** Compile-time-sized evaluation matrix */
    template<int P, int Q, int N>
    using EvalMat = Eigen::Matrix<double,ipow(Q,N-1),ipow(P,N),Eigen::RowMajor>;

    /** Wrapper to convert raw pointer to Eigen */
    template<typename T>
    using Map = Eigen::Map<T>;

    /** IO formatting for Tuple */
    const Eigen::IOFormat TupleFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "(", ")");
}

#endif

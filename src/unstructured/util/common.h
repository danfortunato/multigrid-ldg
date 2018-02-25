#ifndef COMMON_H
#define COMMON_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_STACK_ALLOCATION_LIMIT 1000000
#define MKL_ALIGN 64

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <mkl.h>

namespace DG
{
    template<int N, int P>
    struct Quadrature;

    /** @brief The number of quadrature points to use */
    static const int Q = 10;

    /** @brief Compile-time power function */
    template<typename T>
    inline constexpr T ipow(const T base, unsigned const exponent)
    {
        return (exponent == 0) ? 1 : (base * ipow(base, exponent-1));
    }

    /** @brief Compile-time binomial function */
    template<typename T>
    inline constexpr T ichoose(const T n, const T k)
    {
        return (k==0 || k==n) ? 1 : (ichoose(n-1,k-1) + ichoose(n-1,k));
    }

    /** @brief Compile-time factorial function */
    template<typename T>
    inline constexpr T ifac(const T n)
    {
        return (n==0) ? 1 : n*ifac(n-1);
    }

    /** The storage order to use */
    constexpr auto StorageOrder = Eigen::RowMajor;
    // Eigen disallows row-major column vectors
    template<int M, int N>
    constexpr auto ChooseStorageOrder = (M>1 && N==1) ? Eigen::ColMajor : StorageOrder;

    /** Compile-time-sized matrix */
    template<int M, int N = M>
    using Mat = Eigen::Matrix<double,M,N,ChooseStorageOrder<M,N>>;

    /** Runtime-sized matrix */
    using Matrix = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,StorageOrder>;

    /** Compile-time-sized column vector */
    template<int N>
    using Vec = Eigen::Matrix<double,N,1>;

    /** Runtime-sized column vector */
    using Vector = Eigen::Matrix<double,Eigen::Dynamic,1>;

    /** Compile-time-sized tuple */
    template<typename T, int N>
    using Tuple = Eigen::Array<T,N,1>;

    /** Compile-time-sized diagonal matrix */
    template<int N>
    using Diag = Eigen::DiagonalMatrix<double,N>;

    /** Compile-time-sized tensor product matrix */
    template<int N, int P, int Q = P>
    using KronMat = Mat<ipow(P,N),ipow(Q,N)>;
    /** Compile-time-sized simplex matrix */
    template<int N, int P, int Q = P>
    using SimplexMat = Mat<ichoose(P+N-1,N),ichoose(Q+N-1,N)>;

    /** Compile-time-sized tensor product vector */
    template<int N, int P>
    using KronVec = Vec<ipow(P,N)>;
    /** Compile-time-sized simplex vector */
    template<int N, int P>
    using SimplexVec = Vec<ichoose(P+N-1,N)>;

    /** Compile-time-sized tensor product diagonal matrix */
    template<int N, int P>
    using KronDiag = Diag<ipow(P,N)>;
    /** Compile-time-sized simplex diagonal matrix */
    template<int N, int P>
    using SimplexDiag = Diag<ichoose(P+N-1,N)>;

    /** Compile-time-sized tensor product slicing matrix */
    template<int N, int P>
    using SliceMat = Mat<ipow(P,N-1),ipow(P,N)>;
    /** Compile-time-sized simplex slicing matrix */
    template<int N, int P>
    using SimplexSliceMat = Mat<ichoose(P+N-2,N-1),ichoose(P+N-1,N)>;

    /** Compile-time-sized tensor product evaluation matrix */
    template<int N, int P, int Q>
    using EvalMat = KronMat<N,Q,P>;
    /** Compile-time-sized simplex evaluation matrix */
    template<int N, int P, int Q>
    using SimplexEvalMat = Mat<Quadrature<N,Q>::size,ichoose(P+N-1,N)>;

    /** Compile-time-sized tensor product slice-evaluation matrix */
    template<int N, int P, int Q>
    using SliceEvalMat = Mat<ipow(Q,N-1),ipow(P,N)>;
    /** Compile-time-sized simplex slice-evaluation matrix */
    template<int N, int P, int Q>
    using SimplexSliceEvalMat = Mat<Quadrature<N-1,Q>::size,ichoose(P+N-1,N)>;

    /** Compile-time-sized tensor product element integration matrix */
    template<int N, int P, int Q>
    using ElemQuadMat = KronMat<N,P,Q>;
    /** Compile-time-sized simplex element integration matrix */
    template<int N, int P, int Q>
    using SimplexElemQuadMat = Mat<ichoose(P+N-1,N),Quadrature<N,Q>::size>;

    /** Compile-time-sized tensor product face integration matrix */
    template<int N, int P, int Q>
    using FaceQuadMat = Mat<ipow(P,N),ipow(Q,N-1)>;
    /** Compile-time-sized simplex face integration matrix */
    template<int N, int P, int Q>
    using SimplexFaceQuadMat = Mat<ichoose(P+N-1,N),Quadrature<N-1,Q>::size>;

    /** Wrapper to convert raw pointer to Eigen */
    template<typename T>
    using Map = Eigen::Map<T>;

    /** IO formatting for Tuple */
    const Eigen::IOFormat TupleFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "(", ")");
}

#endif

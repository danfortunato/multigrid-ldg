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
    template<int N, int P, int Q = P>
    using SimplexKronMat = Mat<ichoose(P+N-1,N),ichoose(Q+N-1,N)>;

    /** Compile-time-sized tensor product vector */
    template<int N, int P>
    using KronVec = Vec<ipow(P,N)>;
    template<int N, int P>
    using SimplexKronVec = Vec<ichoose(P+N-1,N)>;

    /** Compile-time-sized tensor product diagonal matrix */
    template<int N, int P>
    using KronDiag = Diag<ipow(P,N)>;
    template<int N, int P>
    using SimplexKronDiag = Diag<ichoose(P+N-1,N)>;

    /** Compile-time-sized slicing matrix */
    template<int N, int P>
    using SliceMat = Mat<ipow(P,N-1),ipow(P,N)>;
    template<int N, int P>
    using SimplexSliceMat = Mat<ichoose(P+N-2,N-1),ichoose(P+N-1,N)>;

    /** Compile-time-sized evaluation matrix */
    template<int N, int P, int Q>
    using EvalMat = KronMat<N,Q,P>;
    template<int N, int P, int Q>
    using SimplexEvalMat = SimplexKronMat<N,Q,P>;

    /** Compile-time-sized slice evaluation matrix */
    template<int N, int P, int Q>
    using SliceEvalMat = Mat<ipow(Q,N-1),ipow(P,N)>;
    template<int N, int P, int Q>
    using SimplexSliceEvalMat = Mat<ichoose(Q+N-2,N-1),ichoose(P+N-1,N)>;

    /** Compile-time-sized element integration matrix */
    template<int N, int P, int Q>
    using ElemQuadMat = KronMat<N,P,Q>;
    template<int N, int P, int Q>
    using SimplexElemQuadMat = SimplexKronMat<N,P,Q>;

    /** Compile-time-sized face integration matrix */
    template<int N, int P, int Q>
    using FaceQuadMat = Mat<ipow(P,N),ipow(Q,N-1)>;
    template<int N, int P, int Q>
    using SimplexFaceQuadMat = Mat<ichoose(P+N-1,N),ichoose(Q+N-2,N-1)>;

    /** Wrapper to convert raw pointer to Eigen */
    template<typename T>
    using Map = Eigen::Map<T>;

    /** IO formatting for Tuple */
    const Eigen::IOFormat TupleFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "(", ")");
}

#endif

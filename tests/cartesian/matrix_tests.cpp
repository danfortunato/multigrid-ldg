#include "common.h"
#include "sparseblockmatrix.h"
#include <iostream>

int main()
{
    const unsigned int N = 2;
    DG::SparseBlockMatrix<N> T(2,2);
    typedef DG::SparseBlockMatrix<N>::Block Mat;
    Mat R = Mat::Random();

    DG::Vector x(4), y(4);
    x.setOnes();

    // // Build a matrix
    // T.setBlock(0,0,R);
    // T.setBlock(1,1,R*R);
    // DG::SparseBlockMatrix<N> A = T.build();

    // // Build a second matrix
    // T.reset(2,3);
    // T.setBlock(0,0,R);
    // T.setBlock(1,0,R);
    // T.setBlock(1,2,R*R);
    // DG::SparseBlockMatrix<N> B = T.build();

    // DG::SparseBlockMatrix<N> C;
    // bool success = DG::multiply_mm(A, B, C);
    // std::cout << success << std::endl;

    T.setBlock(0,0,(Mat() << 1,2,3,4).finished());
    T.setBlock(1,0,(Mat() << 4,6,7,8).finished());
    T.setBlock(0,1,(Mat() << 9,10,11,12).finished());
    T.setBlock(1,1,(Mat() << 13,14,15,8).finished());
    DG::multiply_mv(T, x, y);
    // DG::SparseBlockMatrix<N> A = T.build();

    // DG::Vec<4> x, y;
    // x << 1, 1, 1, 1;
    // matrix_descr descr;
    // descr.type = SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR;
    // descr.mode = SPARSE_FILL_MODE_UPPER;
    // descr.diag = SPARSE_DIAG_NON_UNIT;
    // sparse_status_t status = mkl_sparse_d_trsv(
    //     SPARSE_OPERATION_NON_TRANSPOSE,
    //     1.0,
    //     A.getMKL(),
    //     descr,
    //     x.data(),
    //     y.data()
    // );
    std::cout << y << std::endl;

    return 0;
}
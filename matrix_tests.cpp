#include "SparseBlockMatrix.h"
#include <iostream>

int main()
{
	const unsigned int N = 4;
	LDG::SparseBlockMatrixBuilder<N> T(2,2);
	typedef LDG::SparseBlockMatrix<N>::Block Mat;
	Mat R = Mat::Random();

	// Build a matrix
	T.setBlock(0,0,R);
	T.setBlock(1,1,R*R);
	LDG::SparseBlockMatrix<N> A = T.build();

	// Build a second matrix
	T.reset(2,3);
	T.setBlock(0,0,R);
	T.setBlock(1,0,R);
	T.setBlock(1,2,R*R);
	LDG::SparseBlockMatrix<N> B = T.build();

	LDG::SparseBlockMatrix<N> C;
	bool success = LDG::multiply_mm(A, B, C);
	std::cout << success << std::endl;

	return 0;
}
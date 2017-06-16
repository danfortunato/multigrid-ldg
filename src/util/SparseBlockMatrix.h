#ifndef SPARSE_BLOCK_MATRIX
#define SPARSE_BLOCK_MATRIX

#include <unordered_map> // std::unordered_map
#include <stdexcept>     // std::out_of_range
#include <mkl.h>         // MKL
#include <mkl_spblas.h>  // MKL routines
#include <Eigen/Dense>

#define MKL_ALIGN 64

namespace LDG
{
	/** @brief A sparse block matrix class
	 *
	 *  This class encapsulates the basic MKL block sparse row (BSR) data
	 *  format. Blocks must be square and all blocks must be the same size.
	 *  Data is stored contiguously by block and each block is dense. For easy
	 *  creation of the BSR sparse data format the class
	 *  SparseBlockMatrixBuilder is provided, which converts from a more
	 *  intuitive block map representation to the internal MKL format. All
	 *  arithmetic is currently done by MKL.
	 */
	template<unsigned int P>
	class SparseBlockMatrix
	{
		public:

			/** The type of the blocks */
			typedef Eigen::Matrix<double,P,P,Eigen::RowMajor> Block;

			/** Constructor */
			SparseBlockMatrix();
			SparseBlockMatrix(unsigned int m, unsigned int n);
			SparseBlockMatrix(sparse_matrix_t mkl);
			SparseBlockMatrix(unsigned int m, unsigned int n, double* values, int* columns, int* rowIndex);

			/** Destructor */
			~SparseBlockMatrix();

			/** Copy constructor */
			SparseBlockMatrix(const SparseBlockMatrix<P>& other);

			/** Copy assignment operator */
			SparseBlockMatrix<P>& operator=(SparseBlockMatrix<P> other);

			/** The total number of rows in the matrix */
			unsigned int rows() const { return m_*P; }

			/** The total number of columns in the matrix */
			unsigned int cols() const { return n_*P; }

			/** The total size of the matrix */
			unsigned int size() const { return rows()*cols(); }

			/** The number of block rows in the matrix */
			unsigned int blockRows() const { return m_; }

			/** The number of block columns in the matrix */
			unsigned int blockCols() const { return n_; }

			/** The number of rows (or columns) per block */
			static unsigned int blockDim() { return P; }

			/** The number of elements per block */
			static unsigned int blockSize() { return P*P; }

			/** The number of nonzero blocks in the matrix */
			unsigned int nnzb() const { return nnzb_; }

			/** The number of explicitly stored elements in the matrix */
			unsigned int nnz() const { return blockSize()*nnzb(); }

			/** Get a handle to the MKL representation */
			sparse_matrix_t getMKL() const { return mkl_; };

			/** Scale the matrix */
			void scale(double alpha);

		private:

			/** The number of block rows */
			unsigned int m_;

			/** The number of block columns */
			unsigned int n_;

			/** The number of nonzero blocks */
			unsigned int nnzb_;

			/** MKL data */
			double* values_;
			int* columns_;
			int* rowIndex_;
			mutable sparse_matrix_t mkl_;
	};

	/** @brief An object for indexing into a sparse matrix */
	struct Index
	{
		Index() = default;
		Index(unsigned int i_, unsigned int j_) : i(i_), j(j_) {}
		unsigned int i, j;

		bool operator<(const Index &other) const
		{
			return i < other.i || (i == other.i && j < other.j);
		}

		bool operator==(const Index &other) const
		{
			return i == other.i && j == other.j;
		}
	};

	/** @brief A hash function for an Index */
	struct IndexHasher
	{
		// Matthew Szudzik's elegant pairing function.
		// Note that this is more compact than Cantor's pairing function!
		std::size_t operator()(const Index& index) const
		{
			unsigned int i = index.i;
			unsigned int j = index.j;
			return i>=j ? i*i+i+j : i+j*j;
		}
	};

	/** @brief A class to assist in creating a SparseBlockMatrix
	 *
	 *  This class allows easy creation of a SparseBlockMatrix without requiring
	 *  the user to define the BSR data. The user instead constructs a block map
	 *  representation of the sparse block matrix; no arithmetic can be done at
	 *  this stage, only matrix assembly. Once matrix assembly is complete, an
	 *  optimized BSR MKL representation is created and arithmetic can be done.
	 */
	template<unsigned int P>
	class SparseBlockMatrixBuilder
	{
		public:

			/** The type of the blocks */
			typedef Eigen::Matrix<double,P,P,Eigen::RowMajor> Block;
			typedef std::unordered_map<Index,Block,IndexHasher> BlockMap;

			/** Constructor */
			SparseBlockMatrixBuilder();
			SparseBlockMatrixBuilder(unsigned int m, unsigned int n);

			/** The total number of rows in the matrix */
			unsigned int rows() const { return m_*P; }

			/** The total number of columns in the matrix */
			unsigned int cols() const { return n_*P; }

			/** The total size of the matrix */
			unsigned int size() const { return rows()*cols(); }

			/** The number of block rows in the matrix */
			unsigned int blockRows() const { return m_; }

			/** The number of block columns in the matrix */
			unsigned int blockCols() const { return n_; }

			/** The number of rows (or columns) per block */
			static unsigned int blockDim() { return P; }

			/** The number of elements per block */
			static unsigned int blockSize() { return P*P; }

			/** The number of nonzero blocks in the matrix */
			unsigned int nnzb() const { return blockMap_.size(); }

			/** The number of explicitly stored elements in the matrix */
			unsigned int nnz() const { return blockSize()*nnzb(); }

			/** Get a block */
			const Block& getBlock(unsigned int i, unsigned int j);

			/** Set a block */
			void setBlock(unsigned int i, unsigned int j, const Block& block);

			/** Check if a block exists */
			bool blockExists(unsigned int i, unsigned int j);

			/** Convert to MKL representation */
			SparseBlockMatrix<P> build();

			/** Reset */
			void reset(unsigned int m, unsigned int n);

		private:

			/** The number of block rows */
			unsigned int m_;

			/** The number of block columns */
			unsigned int n_;

			/** A map of the blocks in the sparse matrix.
			 *  This provides an easy way for the user to construct the matrix. */
			BlockMap blockMap_;

			/** Zero block */
			const Block zero_;
	};
}

#include "SparseBlockMatrix.tpp"

#endif

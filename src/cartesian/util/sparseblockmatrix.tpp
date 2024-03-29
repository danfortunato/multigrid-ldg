#include <vector>
#include <fstream>
#include <limits>

namespace DG
{
    /*************************
     *** SparseBlockMatrix ***
     *************************/

    /** @brief Empty constructor */
    template<int P, int Q>
    SparseBlockMatrix<P,Q>::SparseBlockMatrix() :
        m_(0),
        n_(0)
    {}

    /** @brief Constructor
     *
     *  @param[in] m : The number of block rows
     *  @param[in] n : The number of block columns
     */
    template<int P, int Q>
    SparseBlockMatrix<P,Q>::SparseBlockMatrix(int m, int n) :
        m_(m),
        n_(n)
    {}

    /** @brief Copy constructor
     *
     *  @param[in] other : The sparse block matrix to be copied
     */
    template<int P, int Q>
    SparseBlockMatrix<P,Q>::SparseBlockMatrix(const SparseBlockMatrix<P,Q>& other) :
        m_(other.m_),
        n_(other.n_),
        blockMap_(other.blockMap_),
        colsInRow_(other.colsInRow_),
        rowsInCol_(other.rowsInCol_)
    {}

    /** @brief Copy assignment operator
     *
     *  @param[in] other : The sparse block matrix to be copied
     */
    template<int P, int Q>
    SparseBlockMatrix<P,Q>& SparseBlockMatrix<P,Q>::operator=(SparseBlockMatrix<P,Q> other)
    {
        m_ = other.m_;
        n_ = other.n_;
        blockMap_ = std::move(other.blockMap_);
        colsInRow_ = std::move(other.colsInRow_);
        rowsInCol_ = std::move(other.rowsInCol_);
        return *this;
    }

    /** @brief Get a block
     *
     *  @param[in] i : The row index
     *  @param[in] j : The column index
     *
     *  @note If it is not present the zero matrix is returned.
     */
    template<int P, int Q>
    const typename SparseBlockMatrix<P,Q>::Block&
    SparseBlockMatrix<P,Q>::getBlock(int i, int j) const
    {
        assert(blockExists(i,j));
        return blockMap_.at(Index(i,j));
    }

    /** @brief Set a block
     *
     *  @param[in] i : The row index
     *  @param[in] j : The column index
     *  @param[in] block : The block to insert
     */
    template<int P, int Q>
    void SparseBlockMatrix<P,Q>::setBlock(int i, int j, const Block& block)
    {
        blockMap_[Index(i,j)] = block;
        colsInRow_[i].insert(j);
        rowsInCol_[j].insert(i);
    }

    /** @brief Add to a block
     *
     *  @param[in] i : The row index
     *  @param[in] j : The column index
     *  @param[in] block : The block to add
     */
    template<int P, int Q>
    void SparseBlockMatrix<P,Q>::addToBlock(int i, int j, const Block& block)
    {
        if (blockExists(i,j)) {
            blockMap_[Index(i,j)] += block;
        } else {
            setBlock(i, j, block);
        }
    }

    /** @brief Check for a nonzero block at a specified position
     *
     *  @param[in] i : The row index
     *  @param[in] j : The column index
     */
    template<int P, int Q>
    bool SparseBlockMatrix<P,Q>::blockExists(int i, int j) const
    {
        assert(i >= 0 || i < m_ || j >= 0 || j < n_);
        return blockMap_.count(Index(i,j)) > 0;
    }

    /** @brief Resize and clear the matrix
     *
     *  @param[in] m : The number of block rows
     *  @param[in] n : The number of block columns
     */
    template<int P, int Q>
    void SparseBlockMatrix<P,Q>::reset(int m, int n)
    {
        m_ = m;
        n_ = n;
        blockMap_.clear();
        colsInRow_.clear();
        rowsInCol_.clear();
    }

    /** @brief Scale the matrix
     *
     *  @param[in] alpha : Scaling factor
     */
    template<int P, int Q>
    void SparseBlockMatrix<P,Q>::scale(double alpha)
    {
        for (auto& kv : blockMap_) {
            kv.second *= alpha;
        }
    }

    /** @brief Write the matrix to a file (in Matrix Market format)
     *
     *  @param[in] file : The filename
     */
    template<int P, int Q>
    void SparseBlockMatrix<P,Q>::write(const std::string& file) const
    {
        std::ofstream ofs(file);
        ofs.precision(std::numeric_limits<double>::max_digits10);

        ofs << "%%MatrixMarket matrix coordinate real general" << std::endl;
        ofs << rows() << " " << cols() << " " << nnz() << std::endl;
        for (auto it = blockMap_.begin(); it != blockMap_.end(); ++it) {
            int bi = it->first.i * P;
            int bj = it->first.j * Q;
            for (int i=0; i<P; ++i) {
                for (int j=0; j<Q; ++j) {
                    ofs << bi+i+1 << " " << bj+j+1 << " " << it->second(i,j) << std::endl;
                }
            }
        }

        ofs.close();
    }

    /***************************
     *** Arithmetic routines ***
     ***************************/

    /** @brief Matrix addition
     *
     *  @param[in]  alpha : Scaling factor for A
     *  @param[in]  A     : Sparse block matrix
     *  @param[in]  B     : Sparse block matrix
     *  @param[out] C     : C := alpha*A + B
     */
    template<int P, int Q>
    bool add_mm(double alpha, const SparseBlockMatrix<P,Q>& A, const SparseBlockMatrix<P,Q>& B, SparseBlockMatrix<P,Q>& C)
    {
        // Check that the matrix dimensions match
        if (A.blockRows() != B.blockRows() ||
            A.blockCols() != B.blockCols()) {
            return false;
        }

        C.reset(A.blockRows(), A.blockCols());

        for (const auto& kv : A.blockMap()) {
            C.setBlock(kv.first.i, kv.first.j, alpha*kv.second);
        }

        for (const auto& kv : B.blockMap()) {
            C.addToBlock(kv.first.i, kv.first.j, kv.second);
        }

        return true;
    }

    /** @brief Matrix addition (transposed)
     *
     *  @param[in]  alpha : Scaling factor for A
     *  @param[in]  A     : Sparse block matrix
     *  @param[in]  B     : Sparse block matrix
     *  @param[out] C     : C := alpha*A^T + B
     */
    template<int P, int Q>
    bool add_mm_t(double alpha, const SparseBlockMatrix<P,Q>& A, const SparseBlockMatrix<Q,P>& B, SparseBlockMatrix<Q,P>& C)
    {
        // Check that the matrix dimensions match
        if (A.blockCols() != B.blockRows() ||
            A.blockRows() != B.blockCols()) {
            return false;
        }

        C.reset(B.blockRows(), B.blockCols());

        for (const auto& kv : A.blockMap()) {
            C.setBlock(kv.first.j, kv.first.i, alpha*kv.second.transpose());
        }

        for (const auto& kv : B.blockMap()) {
            C.addToBlock(kv.first.i, kv.first.j, kv.second);
        }

        return true;
    }

    /** @brief Matrix multiplication
     *
     *  @param[in]  A  : Sparse block matrix
     *  @param[in]  B  : Sparse block matrix
     *  @param[out] C  : C := A * B
     */
    template<int P, int Q, int R>
    bool multiply_mm(const SparseBlockMatrix<P,Q>& A, const SparseBlockMatrix<Q,R>& B, SparseBlockMatrix<P,R>& C)
    {
        // Check that the matrix dimensions match
        if (A.blockCols() != B.blockRows()) {
            return false;
        }

        C.reset(A.blockRows(), B.blockCols());

        for (int i = 0; i < A.blockRows(); ++i) {
            for (int k : A.colsInRow(i)) {
                for (int j : B.colsInRow(k)) {
                    C.addToBlock(i, j, A.getBlock(i, k) * B.getBlock(k, j));
                }
            }
        }

        return true;
    }

    /** @brief Matrix multiplication (transposed)
     *
     *  @param[in]  A  : Sparse block matrix
     *  @param[in]  B  : Sparse block matrix
     *  @param[out] C  : C := A^T * B
     */
    template<int P, int Q, int R>
    bool multiply_mm_t(const SparseBlockMatrix<P,Q>& A, const SparseBlockMatrix<P,R>& B, SparseBlockMatrix<Q,R>& C)
    {
        // Check that the matrix dimensions match
        if (A.blockRows() != B.blockRows()) {
            return false;
        }

        C.reset(A.blockCols(), B.blockCols());

        for (int i = 0; i < A.blockCols(); ++i) {
            for (int k : A.rowsInCol(i)) {
                for (int j : B.colsInRow(k)) {
                    C.addToBlock(i, j, A.getBlock(k, i).transpose() * B.getBlock(k, j));
                }
            }
        }

        return true;
    }

    /** @brief Matrix-vector multiplication
     *
     *  @param[in]  A : Sparse block matrix
     *  @param[in]  x : Dense array
     *  @param[out] y : y := A*x
     */
    template<int P, int Q>
    bool multiply_mv(const SparseBlockMatrix<P,Q>& A, const double* x, double* y)
    {
        Map<const Vector> xvec(x, A.cols(), 1);
        Map<Vector> yvec(y, A.rows(), 1);
        for (int i = 0; i < A.blockRows(); ++i) {
            yvec.segment<P>(P*i).setZero();
            for (int j : A.colsInRow(i)) {
                yvec.segment<P>(P*i) += A.getBlock(i, j) * xvec.segment<Q>(Q*j);
            }
        }

        return true;
    }

    /** @brief Matrix-vector multiplication
     *
     *  @param[in]  A : Sparse block matrix
     *  @param[in]  x : Dense vector
     *  @param[out] y : y := A*x
     */
    template<int P, int Q>
    bool multiply_mv(const SparseBlockMatrix<P,Q>& A, const Vector& x, Vector& y)
    {
        y.resize(A.rows());
        return multiply_mv(A, x.data(), y.data());
    }

    /** @brief Matrix-vector multiplication (transposed)
     *
     *  @param[in]  A : Sparse block matrix
     *  @param[in]  x : Dense array
     *  @param[out] y : y := A^T*x
     */
    template<int P, int Q>
    bool multiply_mv_t(const SparseBlockMatrix<P,Q>& A, const double* x, double* y)
    {
        Map<const Vector> xvec(x, A.rows(), 1);
        Map<Vector> yvec(y, A.cols(), 1);
        for (int i = 0; i < A.blockCols(); ++i) {
            yvec.segment<Q>(Q*i).setZero();
            for (int j : A.rowsInCol(i)) {
                yvec.segment<Q>(Q*i) += A.getBlock(j, i).transpose() * xvec.segment<P>(P*j);
            }
        }

        return true;
    }

    /** @brief Matrix-vector multiplication (transposed)
     *
     *  @param[in]  A : Sparse block matrix
     *  @param[in]  x : Dense vector
     *  @param[out] y : y := A^T*x
     */
    template<int P, int Q>
    bool multiply_mv_t(const SparseBlockMatrix<P,Q>& A, const Vector& x, Vector& y)
    {
        y.resize(A.cols());
        return multiply_mv_t(A, x.data(), y.data());
    }

    /** @brief Matrix-vector multiplication
     *
     *  @param[in]     alpha : Scaling factor for A
     *  @param[in]     A     : Sparse block matrix
     *  @param[in]     x     : Dense array
     *  @param[in]     beta  : Scaling factor for y
     *  @param[in,out] y     : y := alpha*A*x + beta*y
     */
    template<int P, int Q>
    bool multiply_add_mv(double alpha, const SparseBlockMatrix<P,Q>& A, const double* x, double beta, double* y)
    {
        Map<const Vector> xvec(x, A.cols(), 1);
        Map<Vector> yvec(y, A.rows(), 1);
        for (int i = 0; i < A.blockRows(); ++i) {
            yvec.segment<P>(P*i) *= beta;
            for (int j : A.colsInRow(i)) {
                yvec.segment<P>(P*i) += alpha * A.getBlock(i, j) * xvec.segment<Q>(Q*j);
            }
        }

        return true;
    }

    /** @brief Matrix-vector multiplication
     *
     *  @param[in]     alpha : Scaling factor for A
     *  @param[in]     A     : Sparse block matrix
     *  @param[in]     x     : Dense vector
     *  @param[in]     beta  : Scaling factor for y
     *  @param[in,out] y     : y := alpha*A*x + beta*y
     */
    template<int P, int Q>
    bool multiply_add_mv(double alpha, const SparseBlockMatrix<P,Q>& A, const Vector& x, double beta, Vector& y)
    {
        return multiply_add_mv(alpha, A, x.data(), beta, y.data());
    }

    /** @brief Matrix-vector multiplication (transposed)
     *
     *  @param[in]     alpha : Scaling factor for A
     *  @param[in]     A     : Sparse block matrix
     *  @param[in]     x     : Dense array
     *  @param[in]     beta  : Scaling factor for y
     *  @param[in,out] y     : y := alpha*A^T*x + beta*y
     */
    template<int P, int Q>
    bool multiply_add_mv_t(double alpha, const SparseBlockMatrix<P,Q>& A, const double* x, double beta, double* y)
    {
        Map<const Vector> xvec(x, A.rows(), 1);
        Map<Vector> yvec(y, A.cols(), 1);
        for (int i = 0; i < A.blockCols(); ++i) {
            yvec.segment<Q>(Q*i) *= beta;
            for (int j : A.rowsInCol(i)) {
                yvec.segment<Q>(Q*i) += alpha * A.getBlock(j, i).transpose() * xvec.segment<P>(P*j);
            }
        }

        return true;
    }

    /** @brief Matrix-vector multiplication (transposed)
     *
     *  @param[in]     alpha : Scaling factor for A
     *  @param[in]     A     : Sparse block matrix
     *  @param[in]     x     : Dense array
     *  @param[in]     beta  : Scaling factor for y
     *  @param[in,out] y     : y := alpha*A^T*x + beta*y
     */
    template<int P, int Q>
    bool multiply_add_mv_t(double alpha, const SparseBlockMatrix<P,Q>& A, const Vector& x, double beta, Vector& y)
    {
        return multiply_add_mv_t(alpha, A, x.data(), beta, y.data());
    }
}

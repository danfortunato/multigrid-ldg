#include <vector>
#include <fstream>
#include <limits>

namespace DG
{
    /*************************
     *** SparseBlockMatrix ***
     *************************/

    /** @brief Empty constructor */
    template<int P>
    SparseBlockMatrix<P>::SparseBlockMatrix() :
        m_(0),
        n_(0),
        nnzb_(0),
        values_(nullptr),
        columns_(nullptr),
        rowIndex_(nullptr),
        mkl_(nullptr),
        fromMKL_(false)
    {}

    /** @brief Constructor
     *
     *  @param[in] m : The number of block rows
     *  @param[in] n : The number of block columns
     */
    template<int P>
    SparseBlockMatrix<P>::SparseBlockMatrix(int m, int n) :
        m_(m),
        n_(n),
        nnzb_(0),
        values_(nullptr),
        columns_(nullptr),
        rowIndex_(nullptr),
        mkl_(nullptr),
        fromMKL_(false)
    {}

    /** @brief Constructor
     *
     *  @param[in] mkl : Handle to MKL representation of matrix
     */
    template<int P>
    SparseBlockMatrix<P>::SparseBlockMatrix(sparse_matrix_t mkl) :
        mkl_(mkl),
        fromMKL_(true)
    {
        sparse_index_base_t indexing;
        sparse_layout_t layout;
        int size, *rowsEnd;
        MKL_INT m, n;

        sparse_status_t status = mkl_sparse_d_export_bsr(
            mkl,
            &indexing,
            &layout,
            &m,
            &n,
            &size,
            &rowIndex_,
            &rowsEnd,
            &columns_,
            &values_
        );

        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::exception();
        }

        m_ = (int) m;
        n_ = (int) n;
        nnzb_ = rowIndex_[m_];
    }

    /** @brief Constructor
     *
     *  @param[in] m        : The number of block rows
     *  @param[in] n        : The number of block columns
     *  @param[in] values   : BSR format for values
     *  @param[in] columns  : BSR format for block columns
     *  @param[in] rowIndex : BSR format for row indices
     */
    template<int P>
    SparseBlockMatrix<P>::SparseBlockMatrix(int m, int n, double* values, int* columns, int* rowIndex) :
        m_(m),
        n_(n),
        nnzb_(rowIndex[m]),
        values_(values),
        columns_(columns),
        rowIndex_(rowIndex),
        fromMKL_(false)
    {
        // Create the MKL representation
        if (mkl_sparse_d_create_bsr(
                &mkl_,
                SPARSE_INDEX_BASE_ZERO,
                SPARSE_LAYOUT_ROW_MAJOR,
                m_,
                n_,
                P,
                (int *)rowIndex_,
                (int *)rowIndex_ + 1,
                (int *)columns_,
                (double *)values_
            ) != SPARSE_STATUS_SUCCESS) {
            throw std::exception();
        }

        // Make sure MKL does not reallocate the arrays
        if (mkl_sparse_set_memory_hint(mkl_, SPARSE_MEMORY_NONE) != SPARSE_STATUS_SUCCESS) {
            mkl_sparse_destroy(mkl_);
            mkl_ = nullptr;
            throw std::exception();
        }

        // Optimize the MKL representation
        if (mkl_sparse_optimize(mkl_) != SPARSE_STATUS_SUCCESS) {
            mkl_sparse_destroy(mkl_);
            mkl_ = nullptr;
            throw std::exception();
        }
    }

    /** Destructor */
    template<int P>
    SparseBlockMatrix<P>::~SparseBlockMatrix()
    {
        if (!fromMKL_) {
            if (values_)   mkl_free(values_);
            if (columns_)  mkl_free(columns_);
            if (rowIndex_) mkl_free(rowIndex_);
        }
        if (mkl_) {
            mkl_sparse_destroy(mkl_);
            mkl_ = nullptr;
        }
    }

    /** @brief Copy constructor
     *
     *  @param[in] other : The sparse block matrix to be copied
     */
    template<int P>
    SparseBlockMatrix<P>::SparseBlockMatrix(const SparseBlockMatrix<P>& other) :
        m_(other.m_),
        n_(other.n_),
        nnzb_(other.nnzb_),
        fromMKL_(false)
    {
        values_   = (double *) mkl_malloc(nnz()  * sizeof(double), MKL_ALIGN);
        columns_  = (int *)    mkl_malloc(nnzb() * sizeof(int),    MKL_ALIGN);
        rowIndex_ = (int *)    mkl_malloc((m_+1) * sizeof(int),    MKL_ALIGN);

        std::copy(values_,   values_ + nnz(),   other.values_);
        std::copy(columns_,  columns_ + nnzb(), other.columns_);
        std::copy(rowIndex_, rowIndex_ + m_+1,  other.rowIndex_);

        // Create the MKL representation
        if (mkl_sparse_d_create_bsr(
                &mkl_,
                SPARSE_INDEX_BASE_ZERO,
                SPARSE_LAYOUT_ROW_MAJOR,
                m_,
                n_,
                P,
                (int *)rowIndex_,
                (int *)rowIndex_ + 1,
                (int *)columns_,
                (double *)values_
            ) != SPARSE_STATUS_SUCCESS) {
            throw std::exception();
        }

        // Make sure MKL does not reallocate the arrays
        if (mkl_sparse_set_memory_hint(mkl_, SPARSE_MEMORY_NONE) != SPARSE_STATUS_SUCCESS) {
            mkl_sparse_destroy(mkl_);
            mkl_ = nullptr;
            throw std::exception();
        }

        // Optimize the MKL representation
        if (mkl_sparse_optimize(mkl_) != SPARSE_STATUS_SUCCESS) {
            mkl_sparse_destroy(mkl_);
            mkl_ = nullptr;
            throw std::exception();
        }
    }

    /** @brief Copy assignment operator
     *
     *  @param[in] other : The sparse block matrix to be copied
     */
    template<int P>
    SparseBlockMatrix<P>& SparseBlockMatrix<P>::operator=(SparseBlockMatrix<P> other)
    {
        m_ = other.m_;
        n_ = other.n_;
        nnzb_ = other.nnzb_;

        values_ = std::move(other.values_);
        other.values_ = 0;

        columns_ = std::move(other.columns_);
        other.columns_ = 0;

        rowIndex_ = std::move(other.rowIndex_);
        other.rowIndex_ = 0;

        mkl_ = std::move(other.mkl_);
        other.mkl_ = 0;

        fromMKL_ = other.fromMKL_;

        return *this;
    }

    /** @brief Scale the matrix
     *
     *  @param[in] alpha : Scaling factor
     */
    template<int P>
    void SparseBlockMatrix<P>::scale(double alpha)
    {
        for (int i = 0; i < nnz(); ++i) {
            values_[i] *= alpha;
        }
    }

    /** @brief Write the matrix to a file (in Matrix Market format)
     *
     *  @param[in] file : The filename
     */
    template<int P>
    void SparseBlockMatrix<P>::write(const std::string& file) const
    {
        std::ofstream ofs(file);
        ofs.precision(std::numeric_limits<double>::max_digits10);

        ofs << "%%MatrixMarket matrix coordinate real general" << std::endl;
        ofs << rows() << " " << cols() << " " << nnz() << std::endl;
        for (int bi = 0; bi < blockRows(); ++bi) {
            for (int k = rowIndex_[bi]; k < rowIndex_[bi+1]; ++k) {
                int bj = columns_[k];
                for (int i=0; i<P; ++i) {
                    for (int j=0; j<P; ++j) {
                        double val = values_[blockSize()*k + P*i + j];
                        ofs << P*bi+i+1 << " " << P*bj+j+1 << " " << val << std::endl;
                    }
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
    template<int P>
    bool add_mm(double alpha, const SparseBlockMatrix<P>& A, const SparseBlockMatrix<P>& B, SparseBlockMatrix<P>& C)
    {
        // Check that the matrix dimensions match
        if (A.blockRows() != B.blockRows() ||
            A.blockCols() != B.blockCols()) {
            return false;
        }

        // Let MKL to the arithmetic
        sparse_matrix_t result;
        sparse_status_t status = mkl_sparse_d_add(
            SPARSE_OPERATION_NON_TRANSPOSE,
            A.getMKL(),
            alpha,
            B.getMKL(),
            &result
        );

        if (status != SPARSE_STATUS_SUCCESS) {
            return false;
        }

        C = SparseBlockMatrix<P>(result);
        return true;
    }

    /** @brief Matrix addition (transposed)
     *
     *  @param[in]  alpha : Scaling factor for A
     *  @param[in]  A     : Sparse block matrix
     *  @param[in]  B     : Sparse block matrix
     *  @param[out] C     : C := alpha*A^T + B
     */
    template<int P>
    bool add_mm_t(double alpha, const SparseBlockMatrix<P>& A, const SparseBlockMatrix<P>& B, SparseBlockMatrix<P>& C)
    {
        // Check that the matrix dimensions match
        if (A.blockCols() != B.blockRows() ||
            A.blockRows() != B.blockCols()) {
            return false;
        }

        // Let MKL to the arithmetic
        sparse_matrix_t result;
        sparse_status_t status = mkl_sparse_d_add(
            SPARSE_OPERATION_TRANSPOSE,
            A.getMKL(),
            alpha,
            B.getMKL(),
            &result
        );

        if (status != SPARSE_STATUS_SUCCESS) {
            return false;
        }

        C = SparseBlockMatrix<P>(result);
        return true;
    }

    /** @brief Matrix multiplication
     *
     *  @param[in]  A  : Sparse block matrix
     *  @param[in]  B  : Sparse block matrix
     *  @param[out] C  : C := A * B
     */
    template<int P>
    bool multiply_mm(const SparseBlockMatrix<P>& A, const SparseBlockMatrix<P>& B, SparseBlockMatrix<P>& C)
    {
        // Check that the matrix dimensions match
        if (A.blockCols() != B.blockRows()) {
            return false;
        }

        // Let MKL do the arithmetic
        sparse_matrix_t result;
        sparse_status_t status = mkl_sparse_spmm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            A.getMKL(),
            B.getMKL(),
            &result
        );

        if (status != SPARSE_STATUS_SUCCESS) {
            return false;
        }

        C = SparseBlockMatrix<P>(result);
        return true;
    }

    /** @brief Matrix multiplication (transposed)
     *
     *  @param[in]  A  : Sparse block matrix
     *  @param[in]  B  : Sparse block matrix
     *  @param[out] C  : C := A^T * B
     */
    template<int P>
    bool multiply_mm_t(const SparseBlockMatrix<P>& A, const SparseBlockMatrix<P>& B, SparseBlockMatrix<P>& C)
    {
        // Check that the matrix dimensions match
        if (A.blockRows() != B.blockRows()) {
            return false;
        }

        // Let MKL do the arithmetic
        sparse_matrix_t result;
        sparse_status_t status = mkl_sparse_spmm(
            SPARSE_OPERATION_TRANSPOSE,
            A.getMKL(),
            B.getMKL(),
            &result
        );

        if (status != SPARSE_STATUS_SUCCESS) {
            return false;
        }

        C = SparseBlockMatrix<P>(result);
        return true;
    }

    /** @brief Matrix-vector multiplication
     *
     *  @param[in]  A : Sparse block matrix
     *  @param[in]  x : Dense array
     *  @param[out] y : y := A*x
     */
    template<int P>
    bool multiply_mv(const SparseBlockMatrix<P>& A, const double* x, double* y)
    {
        return multiply_add_mv(1.0, A, x, 0.0, y);
    }

    /** @brief Matrix-vector multiplication
     *
     *  @param[in]  A : Sparse block matrix
     *  @param[in]  x : Dense vector
     *  @param[out] y : y := A*x
     */
    template<int P>
    bool multiply_mv(const SparseBlockMatrix<P>& A, const Vector& x, Vector& y)
    {
        return multiply_add_mv(1.0, A, x, 0.0, y);
    }

    /** @brief Matrix-vector multiplication (transposed)
     *
     *  @param[in]  A : Sparse block matrix
     *  @param[in]  x : Dense array
     *  @param[out] y : y := A^T*x
     */
    template<int P>
    bool multiply_mv_t(const SparseBlockMatrix<P>& A, const double* x, double* y)
    {
        return multiply_add_mv_t(1.0, A, x, 0.0, y);
    }

    /** @brief Matrix-vector multiplication (transposed)
     *
     *  @param[in]  A : Sparse block matrix
     *  @param[in]  x : Dense vector
     *  @param[out] y : y := A^T*x
     */
    template<int P>
    bool multiply_mv_t(const SparseBlockMatrix<P>& A, const Vector& x, Vector& y)
    {
        return multiply_add_mv_t(1.0, A, x, 0.0, y);
    }

    /** @brief Matrix-vector multiplication
     *
     *  @param[in]     alpha : Scaling factor for A
     *  @param[in]     A     : Sparse block matrix
     *  @param[in]     x     : Dense array
     *  @param[in]     beta  : Scaling factor for y
     *  @param[in,out] y     : y := alpha*A*x + beta*y
     */
    template<int P>
    bool multiply_add_mv(double alpha, const SparseBlockMatrix<P>& A, const double* x, double beta, double* y)
    {
        // Let MKL do the arithmetic
        matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_status_t status = mkl_sparse_d_mv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            A.getMKL(),
            descr,
            x,
            beta,
            y
        );

        return status == SPARSE_STATUS_SUCCESS;
    }

    /** @brief Matrix-vector multiplication
     *
     *  @param[in]     alpha : Scaling factor for A
     *  @param[in]     A     : Sparse block matrix
     *  @param[in]     x     : Dense vector
     *  @param[in]     beta  : Scaling factor for y
     *  @param[in,out] y     : y := alpha*A*x + beta*y
     */
    template<int P>
    bool multiply_add_mv(double alpha, const SparseBlockMatrix<P>& A, const Vector& x, double beta, Vector& y)
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
    template<int P>
    bool multiply_add_mv_t(double alpha, const SparseBlockMatrix<P>& A, const double* x, double beta, double* y)
    {
        // Let MKL do the arithmetic
        matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_status_t status = mkl_sparse_d_mv(
            SPARSE_OPERATION_TRANSPOSE,
            alpha,
            A.getMKL(),
            descr,
            x,
            beta,
            y
        );

        return status == SPARSE_STATUS_SUCCESS;
    }

    /** @brief Matrix-vector multiplication (transposed)
     *
     *  @param[in]     alpha : Scaling factor for A
     *  @param[in]     A     : Sparse block matrix
     *  @param[in]     x     : Dense array
     *  @param[in]     beta  : Scaling factor for y
     *  @param[in,out] y     : y := alpha*A^T*x + beta*y
     */
    template<int P>
    bool multiply_add_mv_t(double alpha, const SparseBlockMatrix<P>& A, const Vector& x, double beta, Vector& y)
    {
        return multiply_add_mv_t(alpha, A, x.data(), beta, y.data());
    }

    /********************************
     *** SparseBlockMatrixBuilder ***
     ********************************/

    /** @brief Empty constructor */
    template<int P>
    SparseBlockMatrixBuilder<P>::SparseBlockMatrixBuilder() :
        m_(0),
        n_(0),
        zero_(SparseBlockMatrixBuilder<P>::Block::Zero())
    {}

    /** @brief Constructor
     *
     *  @param[in] m : The number of block rows
     *  @param[in] n : The number of block columns
     */
    template<int P>
    SparseBlockMatrixBuilder<P>::SparseBlockMatrixBuilder(int m, int n) :
        m_(m),
        n_(n),
        zero_(SparseBlockMatrixBuilder<P>::Block::Zero())
    {}

    /** @brief Get a block
     *
     *  @param[in] i : The row index
     *  @param[in] j : The column index
     *
     *  @note If it is not present the zero matrix is returned.
     */
    template<int P>
    const typename SparseBlockMatrixBuilder<P>::Block&
    SparseBlockMatrixBuilder<P>::getBlock(int i, int j)
    {
        if (blockExists(i,j)) {
            Index index(i,j);
            return blockMap_.at(index);
        } else {
            return zero_;
        }
    }

    /** @brief Set a block
     *
     *  @param[in] i : The row index
     *  @param[in] j : The column index
     *  @param[in] block : The block to insert
     */
    template<int P>
    void SparseBlockMatrixBuilder<P>::setBlock(int i, int j, const Block& block)
    {
        Index index(i,j);
        blockMap_[index] = block;
    }

    /** @brief Add to a block
     *
     *  @param[in] i : The row index
     *  @param[in] j : The column index
     *  @param[in] block : The block to add
     */
    template<int P>
    void SparseBlockMatrixBuilder<P>::addToBlock(int i, int j, const Block& block)
    {
        Index index(i,j);
        if (blockExists(i,j)) {
            blockMap_[index] += block;
        } else {
            blockMap_[index] = block;
        }
    }

    /** @brief Check for a nonzero block at a specified position
     *
     *  @param[in] i : The row index
     *  @param[in] j : The column index
     */
    template<int P>
    bool SparseBlockMatrixBuilder<P>::blockExists(int i, int j)
    {
        if (i < 0 || i >= m_ || j < 0 || j >= n_) {
            throw std::out_of_range("Block index is out of range");
        }
        Index index(i,j);
        return blockMap_.count(index) > 0;
    }

    /** @brief Convert to MKL representation */
    template<int P>
    SparseBlockMatrix<P> SparseBlockMatrixBuilder<P>::build()
    {
        // Get the indices of all the blocks and order them (in row-major order)
        std::vector<Index> keys;
        keys.reserve(nnzb());
        for (auto& kv : blockMap_) {
            keys.push_back(kv.first);
        }
        std::sort(keys.begin(), keys.end());

        // Construct the BSR arrays
        double* values = (double *) mkl_malloc(nnz()  * sizeof(double), MKL_ALIGN);
        int* columns   = (int *)    mkl_malloc(nnzb() * sizeof(int),    MKL_ALIGN);
        int* rowIndex  = (int *)    mkl_malloc((m_+1) * sizeof(int),    MKL_ALIGN);

        int r = 0;
        int rowStart = -1;
        for (int k = 0; k < nnzb(); ++k) {
            const Block& block = blockMap_.at(keys[k]);
            std::copy(block.data(), block.data() + block.size(), &values[k*blockSize()]);
            columns[k] = keys[k].j;
            if (keys[k].i != rowStart) {
                rowIndex[r] = k;
                rowStart = keys[k].i;
                ++r;
            }
        }
        rowIndex[m_] = nnzb();

        return SparseBlockMatrix<P>(m_, n_, values, columns, rowIndex);
    }

    /** @brief Resize and clear the builder
     *
     *  @param[in] m : The number of block rows
     *  @param[in] n : The number of block columns
     */
    template<int P>
    void SparseBlockMatrixBuilder<P>::reset(int m, int n)
    {
        m_ = m;
        n_ = n;
        blockMap_.clear();
    }

    /** @brief Write the matrix to a file (in Matrix Market format)
     *
     *  @param[in] file : The filename
     */
    template<int P>
    void SparseBlockMatrixBuilder<P>::write(const std::string& file) const
    {
        std::ofstream ofs(file);
        ofs.precision(std::numeric_limits<double>::max_digits10);

        ofs << "%%MatrixMarket matrix coordinate real general" << std::endl;
        ofs << rows() << " " << cols() << " " << nnz() << std::endl;
        for (auto it = blockMap_.begin(); it != blockMap_.end(); ++it) {
            int bi = it->first.i * P;
            int bj = it->first.j * P;
            for (int i=0; i<P; ++i) {
                for (int j=0; j<P; ++j) {
                    ofs << bi+i+1 << " " << bj+j+1 << " " << it->second(i,j) << std::endl;
                }
            }
        }

        ofs.close();
    }
}

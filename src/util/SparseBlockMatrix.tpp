#include <vector>

namespace LDG
{
    /*************************
     *** SparseBlockMatrix ***
     *************************/

    /** @brief Empty constructor */
    template<unsigned int P>
    SparseBlockMatrix<P>::SparseBlockMatrix() :
        m_(0),
        n_(0),
        nnzb_(0),
        values_(nullptr),
        columns_(nullptr),
        rowIndex_(nullptr),
        mkl_(nullptr)
    {}

    /** @brief Constructor
     *
     *  @param[in] m : The number of block rows
     *  @param[in] n : The number of block columns
     */
    template<unsigned int P>
    SparseBlockMatrix<P>::SparseBlockMatrix(unsigned int m, unsigned int n) :
        m_(m),
        n_(n),
        nnzb_(0),
        values_(nullptr),
        columns_(nullptr),
        rowIndex_(nullptr),
        mkl_(nullptr)
    {}

    /** @brief Constructor
     *
     *  @param[in] mkl : Handle to MKL representation of matrix
     */
    template<unsigned int P>
    SparseBlockMatrix<P>::SparseBlockMatrix(sparse_matrix_t mkl) :
        mkl_(mkl)
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

        m_ = (unsigned int) m;
        n_ = (unsigned int) n;
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
    template<unsigned int P>
    SparseBlockMatrix<P>::SparseBlockMatrix(unsigned int m, unsigned int n, double* values, int* columns, int* rowIndex) :
        m_(m),
        n_(n),
        nnzb_(rowIndex[m]),
        values_(values),
        columns_(columns),
        rowIndex_(rowIndex)
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
    template<unsigned int P>
    SparseBlockMatrix<P>::~SparseBlockMatrix()
    {
        if (values_)   mkl_free(values_);
        if (columns_)  mkl_free(columns_);
        if (rowIndex_) mkl_free(rowIndex_);
        if (mkl_) {
            mkl_sparse_destroy(mkl_);
            mkl_ = nullptr;
        }
    }

    /** @brief Copy constructor
     *
     *  @param[in] other : The sparse block matrix to be copied
     */
    template<unsigned int P>
    SparseBlockMatrix<P>::SparseBlockMatrix(const SparseBlockMatrix<P>& other) :
        m_(other.m_),
        n_(other.n_),
        nnzb_(other.nnzb_)
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
    template<unsigned int P>
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

        return *this;
    }

    /** @brief Scale the matrix
     *
     *  @param[in] alpha : Scaling factor
     */
    template<unsigned int P>
    void SparseBlockMatrix<P>::scale(double alpha)
    {
        for (unsigned int i = 0; i < nnz(); ++i) {
            values_[i] *= alpha;
        }
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
    template<unsigned int P>
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
    template<unsigned int P>
    bool add_mm_t(double alpha, const SparseBlockMatrix<P>& A, const SparseBlockMatrix<P>& B, SparseBlockMatrix<P>& C)
    {
        // Check that the matrix dimensions match
        if (A.blockRows() != B.blockRows() ||
            A.blockCols() != B.blockCols()) {
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
    template<unsigned int P>
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
    template<unsigned int P>
    bool multiply_mm_t(const SparseBlockMatrix<P>& A, const SparseBlockMatrix<P>& B, SparseBlockMatrix<P>& C)
    {
        // Check that the matrix dimensions match
        if (A.blockCols() != B.blockRows()) {
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
    template<unsigned int P>
    bool multiply_mv(const SparseBlockMatrix<P>& A, const double* x, double* y)
    {
        return multiply_add_mv(1.0, A, x, 0.0, y);
    }

    /** @brief Matrix-vector multiplication (transposed)
     *
     *  @param[in]  A : Sparse block matrix
     *  @param[in]  x : Dense array
     *  @param[out] y : y := A^T*x
     */
    template<unsigned int P>
    bool multiply_mv_t(const SparseBlockMatrix<P>& A, const double* x, double* y)
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
    template<unsigned int P>
    bool multiply_add_mv(double alpha, const SparseBlockMatrix<P>& A, const double* x, double beta, double* y)
    {
        // Let MKL do the arithmetic
        sparse_status_t status = mkl_sparse_d_mv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            A.getMKL(),
            SPARSE_MATRIX_TYPE_GENERAL,
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
    template<unsigned int P>
    bool multiply_add_mv_t(double alpha, const SparseBlockMatrix<P>& A, const double* x, double beta, double* y)
    {
        // Let MKL do the arithmetic
        sparse_status_t status = mkl_sparse_d_mv(
            SPARSE_OPERATION_TRANSPOSE,
            alpha,
            A.getMKL(),
            SPARSE_MATRIX_TYPE_GENERAL,
            x,
            beta,
            y
        );

        return status == SPARSE_STATUS_SUCCESS;
    }

    /********************************
     *** SparseBlockMatrixBuilder ***
     ********************************/

    /** @brief Empty constructor */
    template<unsigned int P>
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
    template<unsigned int P>
    SparseBlockMatrixBuilder<P>::SparseBlockMatrixBuilder(unsigned int m, unsigned int n) :
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
    template<unsigned int P>
    const typename SparseBlockMatrixBuilder<P>::Block&
    SparseBlockMatrixBuilder<P>::getBlock(unsigned int i, unsigned int j)
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
    template<unsigned int P>
    void SparseBlockMatrixBuilder<P>::setBlock(unsigned int i, unsigned int j, const Block& block)
    {
        Index index(i,j);
        blockMap_[index] = block;
    }

    /** @brief Check for a nonzero block at a specified position
     *
     *  @param[in] i : The row index
     *  @param[in] j : The column index
     */
    template<unsigned int P>
    bool SparseBlockMatrixBuilder<P>::blockExists(unsigned int i, unsigned int j)
    {
        if (i >= m_ || j >= n_) {
            throw std::out_of_range("Block index is out of range");
        }
        Index index(i,j);
        return blockMap_.count(index) > 0;
    }

    /** @brief Convert to MKL representation */
    template<unsigned int P>
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

        unsigned int r = 0;
        unsigned int rowStart = -1;
        for (unsigned int k = 0; k < nnzb(); ++k) {
            const Block& block = blockMap_.at(keys[k]);
            std::copy(block.data(), block.data() + block.size(), &values[k*blockSize()]);
            columns[k] = keys[k].j;
            if (keys[k].i != rowStart) {
                rowIndex[r] = k;
                rowStart = keys[k].i;
                ++r;
            }
        }
        rowIndex[m_] = nnzb() + 1;

        return SparseBlockMatrix<P>(m_, n_, values, columns, rowIndex);
    }

    /** @brief Resize and clear the builder
     *
     *  @param[in] m : The number of block rows
     *  @param[in] n : The number of block columns
     */
    template<unsigned int P>
    void SparseBlockMatrixBuilder<P>::reset(unsigned int m, unsigned int n)
    {
        m_ = m;
        n_ = n;
        blockMap_.clear();
    }
}

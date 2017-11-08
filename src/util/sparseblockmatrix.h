#ifndef SPARSE_BLOCK_MATRIX_H
#define SPARSE_BLOCK_MATRIX_H

#include <unordered_map> // std::unordered_map
#include <unordered_set> // std::unordered_set
#include <functional>    // std::hash
#include <stdexcept>     // std::out_of_range
#include <string>        // std::string
#include "common.h"

namespace DG
{
    /** @brief An object for indexing into a sparse matrix */
    struct Index
    {
        Index() = default;
        Index(int i_, int j_) : i(i_), j(j_) {}
        int i, j;

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
        template<typename T>
        inline void hash_combine(std::size_t& seed, const T& v) const
        {
            std::hash<T> hasher;
            seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }

        // Matthew Szudzik's elegant pairing function.
        // Note that this is more compact than Cantor's pairing function!
        std::size_t operator()(const Index& index) const
        {
            int i = index.i;
            int j = index.j;
            return i>=j ? i*i+i+j : i+j*j;

            // std::size_t hash = 0;
            // hash_combine(hash, index.i);
            // hash_combine(hash, index.j);
            // return hash;
        }
    };

    /** @brief A sparse block matrix
     *
     *  This class represents a sparse block matrix as a block map. This allows
     *  for easy construction and modification of the matrix. Arithmetic is done
     *  in block map form.
     */
    template<int P>
    class SparseBlockMatrix
    {
        public:
            /** The type of the blocks */
            typedef Mat<P,P> Block;
            typedef std::unordered_map<Index,Block,IndexHasher> BlockMap;

            /** @brief Empty constructor */
            SparseBlockMatrix();

            /** @brief Constructor */
            SparseBlockMatrix(int m, int n);

            /** @brief Copy constructor */
            SparseBlockMatrix(const SparseBlockMatrix<P>& other);

            /** @brief Copy assignment operator */
            SparseBlockMatrix<P>& operator=(SparseBlockMatrix<P> other);

            /** @brief The total number of rows in the matrix */
            int rows() const { return m_*P; }

            /** @brief The total number of columns in the matrix */
            int cols() const { return n_*P; }

            /** @brief The total size of the matrix */
            int size() const { return rows()*cols(); }

            /** @brief The number of block rows in the matrix */
            int blockRows() const { return m_; }

            /** @brief The number of block columns in the matrix */
            int blockCols() const { return n_; }

            /** @brief The number of rows (or columns) per block */
            static int blockDim() { return P; }

            /** @brief The number of elements per block */
            static int blockSize() { return P*P; }

            /** @brief The number of nonzero blocks in the matrix */
            int nnzb() const { return blockMap_.size(); }

            /** @brief The number of explicitly stored elements in the matrix */
            int nnz() const { return blockSize()*nnzb(); }

            /** @brief Get a block */
            const Block& getBlock(int i, int j) const;

            /** @brief Set a block */
            void setBlock(int i, int j, const Block& block);

            /** @brief Add to a block */
            void addToBlock(int i, int j, const Block& block);

            /** @brief Check if a block exists */
            bool blockExists(int i, int j) const;

            /** @brief Reset */
            void reset(int m, int n);

            /** @brief Scale the matrix */
            void scale(double alpha);

            /** @brief Write the matrix to a file (in Matrix Market format) */
            void write(const std::string& file) const;

            /** @brief Get the block map */
            const BlockMap& blockMap() const { return blockMap_; }

            /** @brief Get the nonzero columns in row i */
            const std::unordered_set<int>& colsInRow(int i) const {
                if (colsInRow_.count(i) > 0) {
                    return colsInRow_.at(i);
                } else {
                    return emptySet_;
                }
            }

            /** @brief Get the nonzero rows in column j */
            const std::unordered_set<int>& rowsInCol(int j) const {
                if (rowsInCol_.count(j) > 0) {
                    return rowsInCol_.at(j);
                } else {
                    return emptySet_;
                }
            }

        private:
            /** The number of block rows */
            int m_;
            /** The number of block columns */
            int n_;
            /** A map of the blocks in the sparse matrix. */
            BlockMap blockMap_;
            /** Indices of the nonzero columns/rows in each row/column. */
            std::unordered_map<int, std::unordered_set<int>> colsInRow_, rowsInCol_;
            /** The empty set of integers */
            std::unordered_set<int> emptySet_;
    };
}

#include "sparseblockmatrix.tpp"

#endif

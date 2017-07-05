#ifndef NDARRAY_H
#define NDARRAY_H

#include <array>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "common.h"
#include "range.h"

namespace DG
{
    // Forward declaration of iterators
    template<typename T, int... Args>
    class NDArrayIterator;

    // Variadic template declaration
    template<typename T, int... Args>
    class NDArray;

    /** @brief An N-dimensional array with extent P */
    template<typename T, int P, int N>
    class NDArray<T,P,N>
    {
        public:
            NDArray() = default;

            NDArray(T value)
            {
                data_.fill(value);
            }

            T& operator()(Tuple<int,N> index)
            {
                return data_[flatten(index)];
            }

            const T& operator()(Tuple<int,N> index) const
            {
                return data_[flatten(index)];
            }

            static int size()
            {
                return ipow(P,N);
            }

            static int size(int)
            {
                return P;
            }

            // Iterators
            friend class NDArrayIterator<T,P,N>;
            typedef NDArrayIterator<T,P,N> iterator;
            iterator begin() { return iterator(this); }
            iterator end() { return iterator(this,0); }

        private:
            int flatten(Tuple<int,N> index)
            {
                int lin = index[0];
                for (int i=1; i<N; ++i) {
                    lin = P*lin + index[i];
                }
                return lin;
            }

            std::array<T,ipow(P,N)> data_;
    };

    /** @brief An N-dimensional array */
    template<typename T, int N>
    class NDArray<T,N>
    {
        public:
            NDArray() = delete;

            NDArray(int size)
            {
                sizes_.fill(size);
                length_ = std::pow(size, N);
                data_.resize(length_);
            }

            NDArray(int size, T value)
            {
                sizes_.fill(size);
                length_ = std::pow(size, N);
                data_.resize(length_, value);
            }

            NDArray(Tuple<int,N> sizes) :
                sizes_(sizes)
            {
                length_ = 1;
                for (int i=0; i<N; ++i) {
                    length_ *= sizes_[i];
                }
                data_.resize(length_);
            }

            NDArray(Tuple<int,N> sizes, T value) :
                sizes_(sizes)
            {
                length_ = 1;
                for (int i=0; i<N; ++i) {
                    length_ *= sizes_[i];
                }
                data_.resize(length_, value);
            }

            T& operator()(Tuple<int,N> index)
            {
                return data_[flatten(index)];
            }

            const T& operator()(Tuple<int,N> index) const
            {
                return data_[flatten(index)];
            }

            int size()
            {
                return length_;
            }

            int size(int i)
            {
                return sizes_[i];
            }

            // Iterators
            friend class NDArrayIterator<T,N>;
            typedef NDArrayIterator<T,N> iterator;
            iterator begin() { return iterator(this); }
            iterator end() { return iterator(this,0); }

        private:
            int flatten(Tuple<int,N> index)
            {
                int lin = index[0];
                for (int i=1; i<N; ++i) {
                    lin = sizes_[i]*lin + index[i];
                }
                return lin;
            }

            int length_;
            Tuple<int,N> sizes_;
            std::vector<int> data_;
    };

    /*****************
     *** Iterators ***
     *****************/

    /** @brief An iterator for an n-dimensional array with extent P */
    template<typename T, int P, int N>
    class NDArrayIterator<T,P,N> : public RangeIterator<P,N>
    {
        public:
            NDArrayIterator(NDArray<T,P,N>* array) :
                array_(array)
            {}

            NDArrayIterator(NDArray<T,P,N>* array, const int) :
                RangeIterator<P,N>(0),
                array_(array)
            {}

            T& operator*() const
            {
                return array_->data_[RangeIterator<P,N>::linearIndex()];
            }

            T* operator->() const
            {
                return array_->data_.data();
            }

        private:
            NDArray<T,P,N>* array_;
    };

    /** @brief An iterator for an N-dimensional array */
    template<typename T, int N>
    class NDArrayIterator<T,N> : public RangeIterator<N>
    {
        public:
            NDArrayIterator(NDArray<T,N>* array) :
                RangeIterator<N>(Range<N>(array->sizes_)),
                array_(array)
            {}

            NDArrayIterator(NDArray<T,N>* array, const int) :
                RangeIterator<N>(Range<N>(array->sizes_),0),
                array_(array)
            {}

            T& operator*() const
            {
                return array_->data_[RangeIterator<N>::linearIndex()];
            }

            T* operator->() const
            {
                return array_->data_.data();
            }

        private:
            NDArray<T,N>* array_;
    };
}

#endif

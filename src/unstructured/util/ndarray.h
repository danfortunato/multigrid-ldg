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
    template<typename T, int... Args>
    class NDArrayConstIterator;
    template<typename T, int... Args>
    class SimplexArrayIterator;
    template<typename T, int... Args>
    class SimplexArrayConstIterator;

    // Variadic template declaration
    template<typename T, int... Args>
    class NDArray;
    template<typename T, int... Args>
    class SimplexArray;

    /** @brief An N-dimensional array with extent P */
    template<typename T, int N, int P>
    class NDArray<T,N,P>
    {
        public:
            NDArray() = default;

            NDArray(T value)
            {
                data_.fill(value);
            }

            NDArray(const std::array<T,ipow(P,N)>& data) :
                data_(data)
            {}

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

            T* data()
            {
                return data_.data();
            }

            const T* data() const
            {
                return data_.data();
            }

            // Iterators
            friend class NDArrayIterator<T,N,P>;
            friend class NDArrayConstIterator<T,N,P>;
            typedef NDArrayIterator<T,N,P> iterator;
            typedef NDArrayConstIterator<T,N,P> const_iterator;
            iterator begin() { return iterator(this); }
            iterator end() { return iterator(this,0); }
            const_iterator begin() const { return const_iterator(this); }
            const_iterator end() const { return const_iterator(this,0); }

        private:
            int flatten(Tuple<int,N> index) const
            {
                int lin = index[0];
                for (int i=1; i<N; ++i) {
                    lin = P*lin + index[i];
                }
                return lin;
            }

            std::array<T,ipow(P,N)> data_;
    };

    /** @brief A simplex array with extent P */
    template<typename T, int N, int P>
    class SimplexArray<T,N,P>
    {
        public:
            SimplexArray() = default;

            SimplexArray(T value)
            {
                data_.fill(value);
            }

            SimplexArray(const std::array<T,ichoose(P+N-1,N)>& data) :
                data_(data)
            {}

            T& operator()(int linearIndex)
            {
                return data_[linearIndex];
            }

            const T& operator()(int linearIndex) const
            {
                return data_[linearIndex];
            }

            static int size()
            {
                return ichoose(P+N-1,N);
            }

            static int size(int)
            {
                return P;
            }

            T* data()
            {
                return data_.data();
            }

            const T* data() const
            {
                return data_.data();
            }

            // Iterators
            friend class SimplexArrayIterator<T,N,P>;
            friend class SimplexArrayConstIterator<T,N,P>;
            typedef SimplexArrayIterator<T,N,P> iterator;
            typedef SimplexArrayConstIterator<T,N,P> const_iterator;
            iterator begin() { return iterator(this); }
            iterator end() { return iterator(this,0); }
            const_iterator begin() const { return const_iterator(this); }
            const_iterator end() const { return const_iterator(this,0); }

        private:
            std::array<T,ichoose(P+N-1,N)> data_;
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

            T* data()
            {
                return data_.data();
            }

            const T* data() const
            {
                return data_.data();
            }

            // Iterators
            friend class NDArrayIterator<T,N>;
            friend class NDArrayConstIterator<T,N>;
            typedef NDArrayIterator<T,N> iterator;
            typedef NDArrayConstIterator<T,N> const_iterator;
            iterator begin() { return iterator(this); }
            iterator end() { return iterator(this,0); }
            const_iterator begin() const { return const_iterator(this); }
            const_iterator end() const { return const_iterator(this,0); }

        private:
            int flatten(Tuple<int,N> index) const
            {
                int lin = index[0];
                for (int i=1; i<N; ++i) {
                    lin = sizes_[i]*lin + index[i];
                }
                return lin;
            }

            int length_;
            Tuple<int,N> sizes_;
            std::vector<T> data_;
    };

    /** @brief A simplex array */
    template<typename T, int N>
    class SimplexArray<T,N>
    {
        public:
            SimplexArray() = delete;

            SimplexArray(int size) :
                P_(size)
            {
                length_ = ichoose(P_+N-1, N);
                data_.resize(length_);
            }

            SimplexArray(int size, T value) :
                P_(size)
            {
                length_ = ichoose(P_+N-1, N);
                data_.resize(length_, value);
            }

            T& operator()(int linearIndex)
            {
                return data_[linearIndex];
            }

            const T& operator()(int linearIndex) const
            {
                return data_[linearIndex];
            }

            int size()
            {
                return length_;
            }

            T* data()
            {
                return data_.data();
            }

            const T* data() const
            {
                return data_.data();
            }

            // Iterators
            friend class SimplexArrayIterator<T,N>;
            friend class SimplexArrayConstIterator<T,N>;
            typedef SimplexArrayIterator<T,N> iterator;
            typedef SimplexArrayConstIterator<T,N> const_iterator;
            iterator begin() { return iterator(this); }
            iterator end() { return iterator(this,0); }
            const_iterator begin() const { return const_iterator(this); }
            const_iterator end() const { return const_iterator(this,0); }

        private:
            int P_, length_;
            std::vector<T> data_;
    };

    /*****************
     *** Iterators ***
     *****************/

    /** @brief An iterator for an n-dimensional array with extent P */
    template<typename T, int N, int P>
    class NDArrayIterator<T,N,P> : public RangeIterator<N,P>
    {
        public:
            NDArrayIterator(NDArray<T,N,P>* array) :
                array_(array)
            {}

            NDArrayIterator(NDArray<T,N,P>* array, const int) :
                RangeIterator<N,P>(0),
                array_(array)
            {}

            T& operator*() const
            {
                return array_->data_[RangeIterator<N,P>::linearIndex()];
            }

            T* operator->() const
            {
                return array_->data_.data();
            }

        private:
            NDArray<T,N,P>* array_;
    };

    /** @brief A const iterator for an n-dimensional array with extent P */
    template<typename T, int N, int P>
    class NDArrayConstIterator<T,N,P> : public RangeIterator<N,P>
    {
        public:
            NDArrayConstIterator(const NDArray<T,N,P>* array) :
                array_(array)
            {}

            NDArrayConstIterator(const NDArray<T,N,P>* array, const int) :
                RangeIterator<N,P>(0),
                array_(array)
            {}

            const T& operator*() const
            {
                return array_->data_[RangeIterator<N,P>::linearIndex()];
            }

            const T* operator->() const
            {
                return array_->data_.data();
            }

        private:
            const NDArray<T,N,P>* array_;
    };

    /** @brief An iterator for a simplex array with extent P */
    template<typename T, int N, int P>
    class SimplexArrayIterator<T,N,P> : public SimplexRangeIterator<N,P>
    {
        public:
            SimplexArrayIterator(SimplexArray<T,N,P>* array) :
                array_(array)
            {}

            SimplexArrayIterator(SimplexArray<T,N,P>* array, const int) :
                SimplexRangeIterator<N,P>(0),
                array_(array)
            {}

            T& operator*() const
            {
                return array_->data_[SimplexRangeIterator<N,P>::linearIndex()];
            }

            T* operator->() const
            {
                return array_->data_.data();
            }

        private:
            SimplexArray<T,N,P>* array_;
    };

    /** @brief A const iterator for a simplex array with extent P */
    template<typename T, int N, int P>
    class SimplexArrayConstIterator<T,N,P> : public SimplexRangeIterator<N,P>
    {
        public:
            SimplexArrayConstIterator(const SimplexArray<T,N,P>* array) :
                array_(array)
            {}

            SimplexArrayConstIterator(const SimplexArray<T,N,P>* array, const int) :
                SimplexRangeIterator<N,P>(0),
                array_(array)
            {}

            const T& operator*() const
            {
                return array_->data_[SimplexRangeIterator<N,P>::linearIndex()];
            }

            const T* operator->() const
            {
                return array_->data_.data();
            }

        private:
            const SimplexArray<T,N,P>* array_;
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

    /** @brief A const iterator for an N-dimensional array */
    template<typename T, int N>
    class NDArrayConstIterator<T,N> : public RangeIterator<N>
    {
        public:
            NDArrayConstIterator(const NDArray<T,N>* array) :
                RangeIterator<N>(Range<N>(array->sizes_)),
                array_(array)
            {}

            NDArrayConstIterator(const NDArray<T,N>* array, const int) :
                RangeIterator<N>(Range<N>(array->sizes_),0),
                array_(array)
            {}

            const T& operator*() const
            {
                return array_->data_[RangeIterator<N>::linearIndex()];
            }

            const T* operator->() const
            {
                return array_->data_.data();
            }

        private:
            const NDArray<T,N>* array_;
    };

    /** @brief An iterator for a simplex array */
    template<typename T, int N>
    class SimplexArrayIterator<T,N> : public SimplexRangeIterator<N>
    {
        public:
            SimplexArrayIterator(SimplexArray<T,N>* array) :
                SimplexRangeIterator<N>(SimplexRange<N>(array->P_)),
                array_(array)
            {}

            SimplexArrayIterator(SimplexArray<T,N>* array, const int) :
                SimplexRangeIterator<N>(SimplexRange<N>(array->P_),0),
                array_(array)
            {}

            T& operator*() const
            {
                return array_->data_[SimplexRangeIterator<N>::linearIndex()];
            }

            T* operator->() const
            {
                return array_->data_.data();
            }

        private:
            SimplexArray<T,N>* array_;
    };

    /** @brief A const iterator for a simplex array */
    template<typename T, int N>
    class SimplexArrayConstIterator<T,N> : public SimplexRangeIterator<N>
    {
        public:
            SimplexArrayConstIterator(const SimplexArray<T,N>* array) :
                SimplexRangeIterator<N>(SimplexRange<N>(array->P_)),
                array_(array)
            {}

            SimplexArrayConstIterator(const SimplexArray<T,N>* array, const int) :
                SimplexRangeIterator<N>(SimplexRange<N>(array->P_),0),
                array_(array)
            {}

            const T& operator*() const
            {
                return array_->data_[SimplexRangeIterator<N>::linearIndex()];
            }

            const T* operator->() const
            {
                return array_->data_.data();
            }

        private:
            const SimplexArray<T,N>* array_;
    };
}

#endif

#ifndef RANGE_H
#define RANGE_H

#include <array>
#include <stdexcept>

namespace DG
{
    // Forward declaration of iterators
    template<int... Args>
    class RangeIterator;

    // Variadic template declaration
    template<int... Args>
    class Range;

    /** @brief An N-dimensional range with extent P */
    template<int P, int N>
    class Range<P,N>
    {
        public:
            // Iterators
            typedef RangeIterator<P,N> iterator;
            static iterator begin() { return iterator(); }
            static iterator end() { return iterator(0); }
    };

    /** @brief An N-dimensional range */
    template<int N>
    class Range<N>
    {
        public:
            Range() = delete;

            Range(int max)
            {
                min_.fill(0);
                max_.fill(max);
                validate();
            }

            Range(int min, int max)
            {
                min_.fill(min);
                max_.fill(max);
                validate();
            }

            Range(std::array<int,N> max) :
                max_(max)
            {
                min_.fill(0);
                validate();
            }

            Range(std::array<int,N> min, std::array<int,N> max) :
                min_(min),
                max_(max)
            {
                validate();
            }

            bool isEmpty() const
            {
                return isEmpty_;
            }

            // Iterators
            friend class RangeIterator<N>;
            typedef RangeIterator<N> iterator;
            iterator begin() { return iterator(*this); }
            iterator end() { return iterator(*this,0); }

        private:
            void validate()
            {
                isEmpty_ = true;
                for (int i=0; i<N; ++i) {
                    if (min_[i] >= max_[i]) {
                        max_[i] = min_[i]+1;
                        squash_[i] = 0;
                    } else {
                        squash_[i] = 1;
                        isEmpty_ = false;
                    }
                }
            }

            bool isEmpty_;
            std::array<int,N> min_, max_, squash_;
    };

    /*****************
     *** Iterators ***
     *****************/

    /** @brief An iterator for an N-dimensional range with extent P */
    template<int P, int N>
    class RangeIterator<P,N>
    {
        public:
            RangeIterator()
            {
                index_.fill(0);
            }

            RangeIterator(const int)
            {
                index_.fill(0);
                index_[0] = P;
            }

            bool operator==(const RangeIterator<P,N>& x) const
            {
                for (int i=0; i<N; ++i) {
                    if (index_[i] != x.index_[i]) {
                        return false;
                    }
                }
                return true;
            }

            bool operator!=(const RangeIterator<P,N>& x) const
            {
                return !(*this == x);
            }

            RangeIterator<P,N>& operator++()
            {
                for (int i=0; i<N; ++i) {
                    int s = N-i-1;
                    if (++index_[s] != P) {
                        return *this;
                    }
                    index_[s] = 0;
                }
                index_[0] = P;
                return *this;
            }

            RangeIterator<P,N> operator++(int)
            {
                RangeIterator<P,N> tmp = *this;
                ++(*this); 
                return tmp;
            }

            RangeIterator<P,N>& operator--()
            {
                for (int i=0; i<N; ++i) {
                    int s = N-i-1;
                    if (index_[s]-- != 0) {
                        return *this;
                    }
                    index_[s] = P-1;
                }
                return *this;
            }

            RangeIterator<P,N> operator--(int)
            {
                RangeIterator<P,N> tmp = *this;
                --(*this);
                return tmp;
            }

            const std::array<int,N>& index() const
            {
                return index_;
            }

            int operator()(int i) const
            {
                if (i < 0 || i >= N) {
                    throw std::out_of_range("Index is out of range.");
                }
                return index_[i];
            }

            int linearIndex() const
            {
                int lin = index_[0];
                for (int i=1; i<N; ++i) {
                    lin = P*lin + index_[i];
                }
                return lin;
            }

        protected:
            std::array<int,N> index_;
    };

    /** @brief An iterator for an N-dimensional range */
    template<int N>
    class RangeIterator<N>
    {
        public:
            RangeIterator() {}

            RangeIterator(const Range<N>& range) :
                min_(range.min_),
                max_(range.max_),
                squash_(range.squash_)
            {
                index_ = min_;
                if (range.isEmpty()) index_[0] = max_[0];
            }

            RangeIterator(const Range<N>& range, const int) :
                min_(range.min_),
                max_(range.max_),
                squash_(range.squash_)
            {
                index_ = min_;
                index_[0] = max_[0];
            }

            bool operator==(const RangeIterator<N>& x) const
            {
                for (int i=0; i<N; ++i) {
                    if (index_[i] != x.index_[i]) {
                        return false;
                    }
                }
                return true;
            }

            bool operator!=(const RangeIterator<N>& x) const
            {
                return !(*this == x);
            }

            RangeIterator<N>& operator++()
            {
                for (int i=0; i<N; ++i) {
                    int s = N-i-1;
                    if (++index_[s] != max_[s]) {
                        return *this;
                    }
                    index_[s] = min_[s];
                }
                index_[0] = max_[0];
                return *this;
            }

            RangeIterator<N> operator++(int)
            {
                RangeIterator<N> tmp = *this;
                ++(*this); 
                return tmp;
            }

            RangeIterator<N>& operator--()
            {
                for (int i=0; i<N; ++i) {
                    int s = N-i-1;
                    if (index_[s]-- != min_[s]) {
                        return *this;
                    }
                    index_[s] = max_[s]-1;
                }
                return *this;
            }

            RangeIterator<N> operator--(int)
            {
                RangeIterator<N> tmp = *this;
                --(*this);
                return tmp;
            }

            const std::array<int,N>& index() const
            {
                return index_;
            }

            int operator()(int i) const
            {
                if (i < 0 || i >= N) {
                    throw std::out_of_range("Index is out of range");
                }
                return index_[i];
            }

            int linearIndex() const
            {
                int lin = index_[0]-min_[0];
                for (int i=1; i<N; ++i) {
                    lin = (max_[i]-min_[i])*lin + squash_[i]*(index_[i]-min_[i]);
                }
                return lin;
            }

        protected:
            std::array<int,N> index_, min_, max_, squash_;
    };
}

#endif


#ifndef RANGE_H
#define RANGE_H

#include <stdexcept>

namespace DG
{
    // Forward declaration of iterators
    template<int... Args>
    class RangeIterator;
    template<int... Args>
    class SimplexRangeIterator;

    // Variadic template declaration
    template<int... Args>
    class Range;
    template<int... Args>
    class SimplexRange;

    /** @brief An N-dimensional range with extent P */
    template<int N, int P>
    class Range<N,P>
    {
        public:
            // Iterators
            typedef RangeIterator<N,P> iterator;
            static iterator begin() { return iterator(); }
            static iterator end() { return iterator(0); }
    };

    /** @brief An N-dimensional simplex range with extent P */
    template<int N, int P>
    class SimplexRange<N,P>
    {
        public:
            // Iterators
            typedef SimplexRangeIterator<N,P> iterator;
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

            Range(Tuple<int,N> max) :
                max_(max)
            {
                min_.fill(0);
                validate();
            }

            Range(Tuple<int,N> min, Tuple<int,N> max) :
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
            Tuple<int,N> min_, max_, squash_;
    };

    /** @brief An N-dimensional simplex range */
    template<int N>
    class SimplexRange<N>
    {
        public:
            SimplexRange() = delete;

            SimplexRange(int max) :
                min_(0),
                max_(max)
            {
                validate();
            }

            SimplexRange(int min, int max) :
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
            friend class SimplexRangeIterator<N>;
            typedef SimplexRangeIterator<N> iterator;
            iterator begin() { return iterator(*this); }
            iterator end() { return iterator(*this,0); }

        private:
            void validate()
            {
                isEmpty_ = (min_ < max_) ? false : true;
            }

            bool isEmpty_;
            int min_, max_;
    };

    /*****************
     *** Iterators ***
     *****************/

    /** @brief An iterator for an N-dimensional range with extent P */
    template<int N, int P>
    class RangeIterator<N,P>
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

            bool operator==(const RangeIterator<N,P>& x) const
            {
                for (int i=0; i<N; ++i) {
                    if (index_[i] != x.index_[i]) {
                        return false;
                    }
                }
                return true;
            }

            bool operator!=(const RangeIterator<N,P>& x) const
            {
                return !(*this == x);
            }

            RangeIterator<N,P>& operator++()
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

            RangeIterator<N,P> operator++(int)
            {
                RangeIterator<N,P> tmp = *this;
                ++(*this);
                return tmp;
            }

            RangeIterator<N,P>& operator--()
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

            RangeIterator<N,P> operator--(int)
            {
                RangeIterator<N,P> tmp = *this;
                --(*this);
                return tmp;
            }

            const Tuple<int,N> index() const
            {
                return index_;
            }

            int& operator()(int i)
            {
                if (i < 0 || i >= N) {
                    throw std::out_of_range("Index is out of range.");
                }
                return index_[i];
            }

            const int& operator()(int i) const
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
            Tuple<int,N> index_;
    };

    /** @brief An iterator for an N-dimensional simplex range with extent P */
    template<int N, int P>
    class SimplexRangeIterator<N,P>
    {
        public:
            SimplexRangeIterator()
            {
                index_.fill(0);
                linearIndex_ = 0;
            }

            SimplexRangeIterator(const int)
            {
                index_.fill(0);
                index_[0] = P;
                linearIndex_ = ichoose(P+N-1,N);
            }

            bool operator==(const SimplexRangeIterator<N,P>& x) const
            {
                for (int i=0; i<N; ++i) {
                    if (index_[i] != x.index_[i]) {
                        return false;
                    }
                }
                return true;
            }

            bool operator!=(const SimplexRangeIterator<N,P>& x) const
            {
                return !(*this == x);
            }

            SimplexRangeIterator<N,P>& operator++()
            {
                for (int i=0; i<N; ++i) {
                    int s = N-i-1;
                    double sum = index_.sum();
                    double othersum = sum-index_[s];
                    if (++index_[s] != P-othersum) {
                        linearIndex_++;
                        assert(index_.sum() <= P && index_.matrix().minCoeff() >= 0);
                        return *this;
                    }
                    index_[s] = 0;
                }
                index_[0] = P;
                linearIndex_ = ichoose(P+N-1,N);
                assert(index_.sum() <= P && index_.matrix().minCoeff() >= 0);
                return *this;
            }

            SimplexRangeIterator<N,P> operator++(int)
            {
                SimplexRangeIterator<N,P> tmp = *this;
                ++(*this);
                return tmp;
            }

            SimplexRangeIterator<N,P>& operator--()
            {
                double sum = index_.sum();
                for (int i=0; i<N; ++i) {
                    int s = N-i-1;
                    if (index_[s]-- != 0) {
                        linearIndex_--;
                        double othersum = sum-index_[s];
                        if (s+1<N) index_[s+1] = P-othersum;
                        assert(index_.sum() <= P && index_.matrix().minCoeff() >= 0);
                        return *this;
                    }
                    index_[s] = 0;
                }
                assert(index_.sum() <= P && index_.matrix().minCoeff() >= 0);
                return *this;
            }

            SimplexRangeIterator<N,P> operator--(int)
            {
                SimplexRangeIterator<N,P> tmp = *this;
                --(*this);
                return tmp;
            }

            const Tuple<int,N> index() const
            {
                return index_;
            }

            int& operator()(int i)
            {
                if (i < 0 || i >= N) {
                    throw std::out_of_range("Index is out of range.");
                }
                return index_[i];
            }

            const int& operator()(int i) const
            {
                if (i < 0 || i >= N) {
                    throw std::out_of_range("Index is out of range.");
                }
                return index_[i];
            }

            int linearIndex() const
            {
                return linearIndex_;
            }

        protected:
            Tuple<int,N> index_;
            int linearIndex_;
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

            const Tuple<int,N> index() const
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
            Tuple<int,N> index_, min_, max_, squash_;
    };

    /** @brief An iterator for an N-dimensional simplex range */
    template<int N>
    class SimplexRangeIterator<N>
    {
        public:

            SimplexRangeIterator() {}

            SimplexRangeIterator(const SimplexRange<N>& range) :
                min_(range.min_),
                max_(range.max_),
                P_(max_-min_)
            {
                index_.fill(min_);
                linearIndex_ = 0;
                if (range.isEmpty()) {
                    index_[0] = max_;
                    linearIndex_ = ichoose(P_+N-1,N);
                }
            }

            SimplexRangeIterator(const SimplexRange<N>& range, const int) :
                min_(range.min_),
                max_(range.max_),
                P_(max_-min_)
            {
                index_.fill(min_);
                index_[0] = max_;
                linearIndex_ =ichoose(P_+N-1,N);
            }

            bool operator==(const SimplexRangeIterator<N>& x) const
            {
                for (int i=0; i<N; ++i) {
                    if (index_[i] != x.index_[i]) {
                        return false;
                    }
                }
                return true;
            }

            bool operator!=(const SimplexRangeIterator<N>& x) const
            {
                return !(*this == x);
            }

            SimplexRangeIterator<N>& operator++()
            {
                for (int i=0; i<N; ++i) {
                    int s = N-i-1;
                    double sum = index_.sum() - N*min_;
                    double othersum = sum - (index_[s]-min_);
                    if (++index_[s] != P_-othersum+min_) {
                        linearIndex_++;
                        assert(index_.sum()-N*min_ <= P_ && index_.matrix().minCoeff() >= min_);
                        return *this;
                    }
                    index_[s] = min_;
                }
                index_[0] = max_;
                linearIndex_ = ichoose(P_+N-1,N);
                assert(index_.sum()-N*min_ <= P_ && index_.matrix().minCoeff() >= min_);
                return *this;
            }

            SimplexRangeIterator<N> operator++(int)
            {
                SimplexRangeIterator<N> tmp = *this;
                ++(*this);
                return tmp;
            }

            SimplexRangeIterator<N>& operator--()
            {
                double sum = index_.sum() - N*min_;
                for (int i=0; i<N; ++i) {
                    int s = N-i-1;
                    if (index_[s]-- != min_) {
                        linearIndex_--;
                        double othersum = sum - (index_[s]-min_);
                        if (s+1<N) index_[s+1] = P_-othersum+min_;
                        assert(index_.sum()-N*min_ <= P_ && index_.matrix().minCoeff() >= min_);
                        return *this;
                    }
                    index_[s] = max_-1;
                }
                assert(index_.sum()-N*min_ <= P_ && index_.matrix().minCoeff() >= min_);
                return *this;
            }

            SimplexRangeIterator<N> operator--(int)
            {
                SimplexRangeIterator<N> tmp = *this;
                --(*this);
                return tmp;
            }

            const Tuple<int,N> index() const
            {
                return index_;
            }

            int& operator()(int i)
            {
                if (i < 0 || i >= N) {
                    throw std::out_of_range("Index is out of range.");
                }
                return index_[i];
            }

            const int& operator()(int i) const
            {
                if (i < 0 || i >= N) {
                    throw std::out_of_range("Index is out of range.");
                }
                return index_[i];
            }

            int linearIndex() const
            {
                return linearIndex_;
            }

        protected:
            Tuple<int,N> index_;
            int linearIndex_, min_, max_, P_;
    };
}

#endif


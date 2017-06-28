#ifndef ITERATOR_H
#define ITERATOR_H

#include <array>
#include <stdexcept>

namespace DG
{
    /** @brief An N-dimensional iterator */
    template<int... Args>
    class Iterator;

    /** @brief An N-dimensional iterator with extent P */
    template<int P, int N>
    class Iterator<P,N>
    {
        public:
            Iterator()
            {
                index_.fill(0);
            }

            static Iterator<P,N> begin()
            {
                Iterator<P,N> it;
                it.index_.fill(0);
                return it;
            }

            static Iterator<P,N> end()
            {
                Iterator<P,N> it;
                it.index_.fill(0);
                it.index_[0] = P;
                return it;
            }

            bool operator==(const Iterator<P,N>& x) const
            {
                for (int i=0; i<N; ++i) {
                    if (index_[i] != x.index_[i]) {
                        return false;
                    }
                }
                return true;
            }

            bool operator!=(const Iterator<P,N>& x) const
            {
                return !(*this == x);
            }

            Iterator<P,N>& operator++()
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

            Iterator<P,N> operator++(int)
            {
                Iterator<P,N> tmp = *this;
                ++(*this); 
                return tmp;
            }

            Iterator<P,N>& operator--()
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

            Iterator<P,N> operator--(int)
            {
                Iterator<P,N> tmp = *this;
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

        private:
            std::array<int,N> index_;
    };

    /** @brief An N-dimensional iterator */
    template<int N>
    class Iterator<N>
    {
        public:
            Iterator() = delete;
            
            Iterator(int max)
            {
                min_.fill(0);
                max_.fill(max);
                index_ = min_;
                validate();
            }

            Iterator(int min, int max)
            {
                min_.fill(min);
                max_.fill(max);
                index_ = min_;
                validate();
            }

            Iterator(std::array<int,N> min, std::array<int,N> max) :
                index_(min),
                min_(min),
                max_(max)
            {
                validate();
            }

            Iterator<N> begin()
            {
                Iterator<N> it(min_, max_);
                it.index_ = min_;
                return it;
            }

            Iterator<N> end()
            {
                Iterator<N> it(min_, max_);
                it.index_ = min_;
                it.index_[0] = max_[0];
                return it;
            }

            bool operator==(const Iterator<N>& x) const
            {
                for (int i=0; i<N; ++i) {
                    if (index_[i] != x.index_[i]) {
                        return false;
                    }
                }
                return true;
            }

            bool operator!=(const Iterator<N>& x) const
            {
                return !(*this == x);
            }

            Iterator<N>& operator++()
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

            Iterator<N> operator++(int)
            {
                Iterator<N> tmp = *this;
                ++(*this); 
                return tmp;
            }

            Iterator<N>& operator--()
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

            Iterator<N> operator--(int)
            {
                Iterator<N> tmp = *this;
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

        private:
            void validate()
            {
                bool all = true;
                for (int i=0; i<N; ++i) {
                    if (min_[i] >= max_[i]) {
                        max_[i] = min_[i]+1;
                        squash_[i] = 0;
                    } else {
                        squash_[i] = 1;
                        all = false;
                    }
                }
                if (all) index_ = end().index_;
            }

            std::array<int,N> index_, min_, max_, squash_;
    };
}

#endif


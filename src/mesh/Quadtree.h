#ifndef QUADTREE
#define QUADTREE

#include <vector>
#include <stdexcept>

namespace LDG
{
    /** @brief An n-dimensional coordinate */
    template<unsigned int N>
    struct Coordinate<N>
    {
            /** @brief Empty constructor */
            Coordinate() { for (unsigned int i = 0; i < N; ++i) { x[i] = 0; } }
            /** @brief Access operator */
            inline double& operator[] (int i) { return x[i]; }
            inline const double& operator[] (int i) const { return x[i]; }
            /** @brief Array of coordinate components */
            double x[N];
    };

    /** @brief Equals operator */
    template<unsigned int N>
    bool operator==(const Coordinate<N>& p, const Coordinate<N>& q) {
        for (unsigned int i = 0; i < N; i++) {
            if (p[i] != q[i]) {
                return false;
            }
        }
        return true;
    }

    /** @brief Unequals operator */
    template<unsigned int N>
    bool operator!=(const Coordinate<N>& p, const Coordinate<N>& q) {
        return !(p == q);
    }

    /** @brief An n-dimensional cell. The bounding box of the cell is specified
     *         by the coordinates of its lower left and upper right corners.
     */
    template<unsigned int N>
    struct Cell
    {
        /** @brief Construct a cell from bounding box coordinates */
        Cell(Coordinate<N> lower_, Coordinate<N> upper_) :
            lower(lower_), upper(upper_)
        {}

        /** @brief Compute the width of the cell in the dimension d */
        double width(unsigned int d) {
            if (d >= N) {
                throw std::invalid_argument("Requested dimension too large.");
            }
            return upper[d]-lower[d];
        }

        /** @brief Compute the volume of the cell */
        double volume() {
            double v = 1;
            for (int i=0; i<N; ++i) { v *= width(i); }
            return v;
        }

        /** @brief The lower left and upper right coordinates of the cell */
        Coordinate<N> lower, upper;
    };

    /** @brief An n-dimensional quadtree */
    template<class T, unsigned int N>
    class Quadtree
    {
        public:
            /** @brief A node in the tree */
            struct Node
            {
                Node(Cell<N> cell_) : cell(cell_) {
                    for (int i=0; i<numChildren; ++i) {
                        children[i] = nullptr;
                    }
                }

                T object;
                Cell<N> cell;
                bool isLeaf;
                const int numChildren = 1 << N;
                Node* children[numChildren];
            };

            /** @brief Construct an empty tree from a given domain
             *
             *  @param[in] domain : Cell representing the tree's domain
             */
            Quadtree(Cell<N> domain);

            /** @brief Construct a tree from a given domain that is uniformly
             *         refined n times.
             *
             *  @param[in] domain : Cell representing the tree's domain
             *  @param[in] n      : Number of refinements
             */
            Quadtree(Cell<N> domain, int n);

            /** @brief Destructor */
            ~Quadtree();

            /** @brief Uniformly refine the tree n times
             *
             *  @param[in] n : Number of refinements
             */
            void refine(int n);

        private:
            /** @brief The root of the tree */
            Node* root;
            /** @brief The number of levels in the tree */
            int numLevels;
    };
}

#include "Quadtree.tpp"

#endif

#ifndef QUADTREE
#define QUADTREE

#include <vector>
#include <stdexcept>

namespace DG
{
    /** @brief An n-dimensional coordinate */
    template<int N>
    struct Coordinate
    {
            /** @brief Constructor from value */
            Coordinate(double v = 0) { for (int i = 0; i < N; ++i) { x[i] = v; } }
            /** @brief Access operator */
            inline double& operator[] (int i) { return x[i]; }
            inline const double& operator[] (int i) const { return x[i]; }
            /** @brief Array of coordinate components */
            double x[N];
    };

    template<>
    struct Coordinate<2>
    {
        Coordinate(double v = 0) : x(v), y(v) {}
        Coordinate(double x_, double y_) : x(x_), y(y_) {}
        inline double& operator[] (int i) {
            if (i < 1 || i > 2) {
                throw std::out_of_range("Requested component does not exist.");
            }
            return i==1 ? x : y;
        }
        inline const double& operator[] (int i) const {
            if (i < 1 || i > 2) {
                throw std::out_of_range("Requested component does not exist.");
            }
            return i==1 ? x : y;
        }
        double x, y;
    };

    template<>
    struct Coordinate<3>
    {
        Coordinate(double v = 0) : x(v), y(v), z(v) {}
        Coordinate(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
        inline double& operator[] (int i) {
            if (i < 1 || i > 3) {
                throw std::out_of_range("Requested component does not exist.");
            }
            return i==1 ? x : (i==2 ? y : z);
        }
        inline const double& operator[] (int i) const {
            if (i < 1 || i > 3) {
                throw std::out_of_range("Requested component does not exist.");
            }
            return i==1 ? x : (i==2 ? y : z);
        }
        double x, y, z;
    };

    /** @brief Equals operator */
    template<int N>
    bool operator==(const Coordinate<N>& p, const Coordinate<N>& q) {
        for (int i = 0; i < N; i++) {
            if (p[i] != q[i]) {
                return false;
            }
        }
        return true;
    }

    /** @brief Unequals operator */
    template<int N>
    bool operator!=(const Coordinate<N>& p, const Coordinate<N>& q) {
        return !(p == q);
    }

    /** @brief An n-dimensional cell. The bounding box of the cell is specified
     *         by the coordinates of its lower left and upper right corners.
     */
    template<int N>
    struct Cell
    {
        /** @brief Construct a unit cell */
        Cell() :
            lower(Coordinate<N>(0)),
            upper(Coordinate<N>(1))
        {}

        /** @brief Construct a cell from bounding box coordinates */
        Cell(Coordinate<N> lower_, Coordinate<N> upper_) :
            lower(lower_), upper(upper_)
        {}

        /** @brief Compute the width of the cell in the dimension d */
        double width(int d) {
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
    template<int N>
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

                int id;
                Cell<N> cell;
                bool isLeaf;
                static const int numChildren = 1 << N;
                Node* children[numChildren];
            };

            /** @brief Construct an empty tree from the unit domain */
            Quadtree() :
                root(new Node(Cell<N>())),
                numLevels(1)
            {}

            /** @brief Construct a tree from the unit domain that is uniformly
             *         refined n times.
             *
             *  @param[in] n : Number of refinements
             */
             Quadtree(int n) :
                root(new Node(Cell<N>())),
                numLevels(n+1)
            {
                refine(n);
            }

            /** @brief Construct an empty tree from a given domain
             *
             *  @param[in] domain : Cell representing the tree's domain
             */
            Quadtree(Cell<N> domain) :
                root(new Node(domain)),
                numLevels(1)
            {}

            /** @brief Construct a tree from a given domain that is uniformly
             *         refined n times.
             *
             *  @param[in] domain : Cell representing the tree's domain
             *  @param[in] n      : Number of refinements
             */
            Quadtree(Cell<N> domain, int n) :
                root(new Node(domain)),
                numLevels(n+1)
            {
                refine(n);
            }

            /** @brief Destructor */
            ~Quadtree() {
                remove(root);
            }

            /** @brief Uniformly refine the tree */
            void refine() {
                refine(root);
            }

            /** @brief Uniformly refine the tree n times
             *
             *  @param[in] n : Number of refinements
             */
            void refine(int n) {
                for (int i=0; i<n; ++i) {
                    refine();
                }
            }

        private:
            /** @brief Uniformly refine starting from a given node
             *
             *  @param[in] node : The node at which to begin refinement
             */
            void refine(Node* node);

            /** @brief Remove a node and its children
             *
             *  @param[in] node : The node to remove
             */
            void remove(Node* node);

            /** @brief The root of the tree */
            Node* root;
            /** @brief The number of levels in the tree */
            int numLevels;
    };
}

#include "Quadtree.tpp"

#endif

#ifndef QUADTREE_H
#define QUADTREE_H

#include <vector>
#include <stdexcept>
#include <functional>
#include "common.h"
#include "ndarray.h"

namespace DG
{
    /** @brief A direction */
    enum Direction
    {
        kLeft,
        kRight
    };

    /** @brief A strategy for coarsening the quadtree */
    enum CoarseningStrategy
    {
        kEqualCoarsening,
        kRapidCoarsening
    };

    /** @brief An n-dimensional cell. The bounding box of the cell is specified
     *         by the coordinates of its lower left and upper right corners. */
    template<int N>
    struct Cell
    {
        /** @brief Construct a unit cell */
        Cell() :
            lower(Tuple<double,N>(0)),
            upper(Tuple<double,N>(1))
        {}

        /** @brief Construct a cell from bounding box coordinates */
        Cell(Tuple<double,N> lower_, Tuple<double,N> upper_) :
            lower(lower_),
            upper(upper_)
        {}

        Tuple<double,N> width() {
            return upper-lower;
        }

        /** @brief Compute the width of the cell in the dimension d */
        double width(int d) {
            if (d >= N) {
                throw std::invalid_argument("Requested dimension too large.");
            }
            return upper[d]-lower[d];
        }

        double maxWidth() {
            double max = width(0);
            for (int i=1; i<N; ++i) {
                if (width(i) > max) {
                    max = width(i);
                }
            }
            return max;
        }

        double minWidth() {
            double min = width(0);
            for (int i=1; i<N; ++i) {
                if (width(i) < min) {
                    min = width(i);
                }
            }
            return min;
        }

        /** @brief Compute the volume of the cell */
        double volume() {
            double v = 1;
            for (int i=0; i<N; ++i) { v *= width(i); }
            return v;
        }

        /** @brief The lower left and upper right coordinates of the cell */
        Tuple<double,N> lower, upper;
    };

    /** @brief An n-dimensional quadtree */
    template<int N>
    class Quadtree
    {
        public:
            /** @brief A node in the tree */
            struct Node
            {
                Node(Cell<N> cell_, int id_, int parent_) :
                    id(id_),
                    level(0),
                    height(0),
                    parent(parent_),
                    cell(cell_),
                    children()
                {}

                int id;
                int level;
                int height;
                int parent;
                Cell<N> cell;
                static const int numChildren = 1 << N;
                NDArray<int,2,N> children;
            };

            Quadtree(std::function<Tuple<double,N>(Tuple<double,N>)> h_) :
                h(h_),
                numLevels_(1)
            {
                tree.push_back(Node(Cell<N>(),0,-1));
                build(0,0);
            }

            Quadtree(Cell<N> domain, std::function<Tuple<double,N>(Tuple<double,N>)> h_) :
                h(h_),
                numLevels_(1)
            {
                tree.push_back(Node(domain,0,-1));
                build(0,0);
            }

            int size() const
            {
                return tree.size();
            }

            int numLevels() const
            {
                return numLevels_;
            }

            int height() const
            {
                return numLevels_-1;
            }

            const Node& operator[](int id)
            {
                return tree[id];
            }

            /** @brief Is this node a leaf for the given coarsening? */
            bool isLeaf(Node& node, int coarsening = 0)
            {
                assert(coarsening>=0 && coarsening<numLevels_);
                if (coarseningStrategy_ == kRapidCoarsening) {
                    return node.height == coarsening || (node.height < coarsening && tree[node.parent].height > coarsening);
                } else {
                    int clevel = numLevels_-1-coarsening;
                    return node.level == clevel || (node.level < clevel && node.height == 0);
                }
            }

            /** @brief Return a list of elements at a given layer in the hierarchy */
            std::vector<int> layer(int l)
            {
                assert(l>=0 && l<numLevels_);
                std::vector<int> layer;
                dfs([&](Node& node) {
                    if (isLeaf(node, l)) {
                        layer.push_back(node.id);
                    }
                }, l);
                return layer;
            }

            /** @brief Find the neighbors of a given node in a given dimension
             *         and direction */
            std::vector<int> neighbors(int id, int dim, Direction dir, int coarsening = 0);
            /** @brief Apply f to the tree in a depth-first order */
            void dfs(std::function<void(Node&)> f, int coarsening = 0) { dfs(f, 0, coarsening); }
            /** @brief Apply f to the tree in a breadth-first order */
            void bfs(std::function<void(Node&)> f, int coarsening = 0);
            /** @brief Search the tree with pruning based on a condition */
            void search(std::function<bool(Node&)> cond, std::function<void(Node&)> accum, int id, int coarsening = 0);
            /** @brief Is node B a neighbor of node A? */
            bool isNeighbor(Node& a, Node& b, int dim, Direction dir);
            /** @brief Is node B a parent of node A? */
            bool isParent(Node& a, Node& b);

        private:
            /** @brief Construct a node and its children */
            void build(int id, int level);
            /** @brief DFS helper */
            void dfs(std::function<void(Node&)> f, int id, int coarsening);
            /** @brief The nodes in the tree */
            std::vector<Node> tree;
            /** @brief Refinement function */
            std::function<Tuple<double,N>(Tuple<double,N>)> h;
            /** @brief The number of levels in the tree */
            int numLevels_;
            /** @brief Coarsening strategy */
            CoarseningStrategy coarseningStrategy_;
    };
}

#include "Quadtree.tpp"

#endif

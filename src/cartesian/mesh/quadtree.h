#ifndef QUADTREE_H
#define QUADTREE_H

#include <vector>
#include <stdexcept>
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

    /** @brief An N-dimensional cell. The bounding box of the cell is specified
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

        Tuple<double,N> width() const {
            return upper-lower;
        }

        /** @brief Compute the width of the cell in the dimension d */
        double width(int d) const {
            if (d < 0 || d > N) {
                throw std::out_of_range("Requested dimension does not exist.");
            }
            return upper[d]-lower[d];
        }

        double maxWidth() const {
            double max = width(0);
            for (int i=1; i<N; ++i) {
                if (width(i) > max) {
                    max = width(i);
                }
            }
            return max;
        }

        double minWidth() const {
            double min = width(0);
            for (int i=1; i<N; ++i) {
                if (width(i) < min) {
                    min = width(i);
                }
            }
            return min;
        }

        /** @brief Compute the volume of the cell */
        double volume() const {
            double v = 1;
            for (int i=0; i<N; ++i) { v *= width(i); }
            return v;
        }

        /** @brief The lower left and upper right coordinates of the cell */
        Tuple<double,N> lower, upper;
    };

    template<int N>
    bool operator==(const Cell<N>& cell1, const Cell<N>& cell2)
    {
        return (cell1.lower == cell2.lower).all() &&
               (cell1.upper == cell2.upper).all();
    }

    /** @brief An N-dimensional quadtree */
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

            template<typename T>
            Quadtree(const T& h, CoarseningStrategy coarseningStrategy, bool periodic_ = false) :
                numLevels_(1),
                coarseningStrategy_(coarseningStrategy),
                periodic(periodic_)
            {
                tree.push_back(Node(Cell<N>(),0,-1));
                build(0, 0, h);
            }

            template<typename T>
            Quadtree(Cell<N> domain, const T& h, CoarseningStrategy coarseningStrategy, bool periodic_ = false) :
                numLevels_(1),
                coarseningStrategy_(kEqualCoarsening),
                periodic(periodic_)
            {
                tree.push_back(Node(domain,0,-1));
                build(0, 0, h);
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

            Node& operator[](int id)
            {
                return tree[id];
            }

            const Node& operator[](int id) const
            {
                return tree[id];
            }

            /** @brief Is this node a leaf for the given coarsening? */
            bool isLeaf(const Node& node, int coarsening = 0) const
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
            std::vector<int> layer(int l) const
            {
                assert(l>=0 && l<numLevels_);
                std::vector<int> layer;
                dfs([&](const Node& node) {
                    if (isLeaf(node, l)) {
                        layer.push_back(node.id);
                    }
                }, l);
                return layer;
            }

            /** @brief Find the neighbors of a given node in a given dimension
             *         and direction */
            std::vector<int> neighbors(int id, int dim, Direction dir, int coarsening = 0) const;
            /** @brief Apply f to the tree in a depth-first order */
            template<typename T>
            void dfs(const T& f, int coarsening = 0) const
            {
                dfs(f, 0, coarsening);
            }
            /** @brief Apply f to the tree in a breadth-first order */
            template<typename T>
            void bfs(const T& f, int coarsening = 0) const;
            /** @brief Search the tree with pruning based on a condition */
            template<typename T1, typename T2>
            void search(const T1& cond, const T2& accum, int id, int coarsening = 0) const;
            /** @brief Is node B a neighbor of node A? */
            bool isNeighbor(const Node& a, const Node& b, int dim, Direction dir) const;
            /** @brief Is node B a parent of node A? */
            bool isParent(const Node& a, const Node& b) const;

        private:
            /** @brief Construct a node and its children */
            template<typename T>
            void build(int id, int level, const T& h);
            /** @brief DFS helper */
            template<typename T>
            void dfs(const T& f, int id, int coarsening) const;
            /** @brief The nodes in the tree */
            std::vector<Node> tree;
            /** @brief The number of levels in the tree */
            int numLevels_;
            /** @brief Coarsening strategy */
            CoarseningStrategy coarseningStrategy_;
            /** @brief Is this quadtree periodic? */
            bool periodic;
    };
}

#include "quadtree.tpp"

#endif

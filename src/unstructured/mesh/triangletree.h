#ifndef TRIANGLE_TREE_H
#define TRIANGLE_TREE_H

#include <vector>
#include "geometry.h"
#include "wireframe.h"

namespace DG
{
    class TriangleTree
    {
        public:
            /** @brief A node in the tree */
            struct Node
            {
                Node(Simplex<2> simplex_, int id_, int parent_) :
                    id(id_),
                    level(0),
                    height(0),
                    parent(parent_),
                    simplex(simplex_),
                    children()
                {}

                int id;
                int level;
                int height;
                int parent;
                Simplex<2> simplex;
                static const int numChildren = 4;
                std::array<int,numChildren> children;
            };

            TriangleTree(const Wireframe<2>& coarse, int nrefine)
            {
                numLevels_ = nrefine+1;
                for (int i=0; i<coarse.nt; ++i) {
                    Tuple<int,3> t = coarse.t[i];
                    Simplex<2> simplex;
                    for (int j=0; j<3; ++j) {
                        simplex.p[j] = coarse.p[t[j]];
                    }
                    int id = tree.size();
                    tree.push_back(Node(simplex,id,-1));
                    roots.push_back(id);
                    build(id, 0, nrefine);
                }
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
                return node.height == coarsening || (node.height < coarsening && tree[node.parent].height > coarsening);
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

            /** @brief Apply f to the tree in a depth-first order */
            template<typename T>
            void dfs(const T& f, int coarsening = 0) const
            {
                for (int root : roots) dfs(f, root, coarsening);
            }

        private:
            /** @brief Construct a node and its children */
            void build(int id, int level, int nrefine)
            {
                tree[id].level = level;
                tree[id].height = 0;
                if (nrefine > 0) {
                    // Build children
                    Simplex<2> child;
                    Tuple<double,2> p1 = tree[id].simplex.p[0];
                    Tuple<double,2> p2 = tree[id].simplex.p[1];
                    Tuple<double,2> p3 = tree[id].simplex.p[2];
                    Tuple<double,2> p12 = (p1+p2)/2.;
                    Tuple<double,2> p23 = (p2+p3)/2.;
                    Tuple<double,2> p13 = (p1+p3)/2.;

                    child.p[0] = p12;
                    child.p[1] = p23;
                    child.p[2] = p13;
                    int next = tree.size();
                    Node node1(child,next,id);
                    tree.push_back(node1);
                    tree[id].children[0] = next;
                    build(next, level+1, nrefine-1);

                    child.p[0] = p1;
                    child.p[1] = p12;
                    child.p[2] = p13;
                    next = tree.size();
                    Node node2(child,next,id);
                    tree.push_back(node2);
                    tree[id].children[1] = next;
                    build(next, level+1, nrefine-1);

                    child.p[0] = p2;
                    child.p[1] = p12;
                    child.p[2] = p23;
                    next = tree.size();
                    Node node3(child,next,id);
                    tree.push_back(node3);
                    tree[id].children[2] = next;
                    build(next, level+1, nrefine-1);

                    child.p[0] = p3;
                    child.p[1] = p13;
                    child.p[2] = p23;
                    next = tree.size();
                    Node node4(child,next,id);
                    tree.push_back(node4);
                    tree[id].children[3] = next;
                    build(next, level+1, nrefine-1);

                    // Set height
                    for (int i=0; i<4; ++i) {
                        tree[id].height = std::max(tree[tree[id].children[i]].height, tree[id].height);
                    }
                    tree[id].height++;
                }
            }

            /** @brief DFS helper */
            template<typename T>
            void dfs(const T& f, int id, int coarsening) const
            {
                f(tree[id]);
                if (!isLeaf(tree[id], coarsening)) {
                    for (auto it = tree[id].children.begin(); it != tree[id].children.end(); ++it) {
                        dfs(f, *it, coarsening);
                    }
                }
            }

            /** @brief The nodes in the tree */
            std::vector<Node> tree;
            /** @brief The roots of the trees */
            std::vector<int> roots;
            /** @brief The number of levels in the tree */
            int numLevels_;
    };
}

#endif

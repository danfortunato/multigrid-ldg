#ifndef QUADTREE_H
#define QUADTREE_H

#include <vector>
#include <stdexcept>
#include "common.h"

namespace DG
{
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

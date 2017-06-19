#include <vector>

namespace LDG
{
    /** @brief Uniformly refine starting from a given node
     *
     *  @param[in] node : The node at which to begin refinement
     */
    template<typename T, unsigned int N>
    Quadtree<T,N>::refine(Node* node)
    {
        if (node) {
            if (node->isLeaf) {
                std::vector<Cell<N>> cells(node->numChildren);
                cells[0] = node->cell;

                // Refine the node's cell by recursively dividing the cell in
                // each dimension. To do this, we use a vector of cells as a
                // scratch pad to track the divisions we've done so far, and
                // divide in the next dimension using these cells. At the end of
                // this procedure the vector will contain the cells refined in
                // all dimensions.
                for (int d = 0; d < N; ++i) {
                    for (int j = (1<<d)-1; j >= 0; --j) {

                        Coordinate<N> lower1 = cells[j].lower;
                        Coordinate<N> upper1 = cells[j].upper;
                        upper1[d] -= cells[j].width(d)/2;

                        Coordinate<N> lower2 = cells[j].lower;
                        Coordinate<N> upper2 = cells[j].upper;
                        lower2[d] += cells[j].width(d)/2;

                        cells[2*j]   = Cell<N>(lower1, upper1);
                        cells[2*j+1] = Cell<N>(lower2, upper2);
                    }
                }

                // Now that we've created all the child cells, we create the
                // nodes. There should be numChildren cells to create.
                assert(cells.size() == node->numChildren);
                for (Cell<N> c : cells) {
                    node->children[i] = new Node(c);
                }
            } else {
                // Not a leaf, refine the children
                for (int i = 0; i < node->numChildren; ++i) {
                    refine(node->children[i]);
                }
            }
        }
    }

    /** @brief Remove a node and its children
     *
     *  @param[in] node : The node to remove
     */
    template<typename T, unsigned int N>
    Quadtree<T,N>::remove(Node* node)
    {
        if (node) {
            if (!node->isLeaf) {
                for (int i = 0; i < node->numChildren; ++i) {
                    remove(node->children[i]);
                }
            }
            delete node;
        }
    }
}

#include <vector>
#include <algorithm>
#include <queue>

namespace DG
{
    /** @brief Construct a node and its children */
    template<int N>
    void Quadtree<N>::build(int id, int level)
    {
        numLevels_ = std::max(level+1,numLevels_);
        tree[id].level = level;
        tree[id].height = 0;
        if ((tree[id].cell.width() > h(tree[id].cell.lower + tree[id].cell.width()/2)).all()) {
            // Build children
            Tuple<double,N> mid = tree[id].cell.lower + tree[id].cell.width()/2;
            for (RangeIterator<2,N> it; it != Range<2,N>::end(); ++it) {
                Tuple<double,N> lower = tree[id].cell.lower;
                Tuple<double,N> upper = tree[id].cell.upper;
                for (int i=0; i<N; ++i) {
                    if (it(i)) {
                        lower[i] = mid[i];
                    } else {
                        upper[i] = mid[i];
                    }
                }
                int next = tree.size();
                Node node(Cell<N>(lower,upper),next,id);
                tree.push_back(node);
                tree[id].children(it.index()) = next;
                build(next,level+1);
            }
            // Set height
            for (NDArrayIterator<int,2,N> it = tree[id].children.begin(); it != tree[id].children.end(); ++it) {
                tree[id].height = std::max(tree[*it].height, tree[id].height);
            }
            tree[id].height++;
        }
    }

    /** @brief Find the neighbors of a given node in a given dimension and
     *         direction */
    template<int N>
    std::vector<int> Quadtree<N>::neighbors(int id, int dim, Direction dir)
    {
        std::vector<int> v;
        auto cond = [&](Node& node) {
            return isNeighbor(tree[id], node, dim, dir) || isParent(tree[id], node);
        };
        auto accum = [&](Node& node) {
            v.push_back(node.id);
        };
        search(cond, accum, 0);
        return v;
    }

    /** @brief Is node B a neighbor of node A? */
    template<int N>
    bool Quadtree<N>::isNeighbor(Node& a, Node& b, int dim, Direction dir)
    {
        // In the same plane?
        double planeA = (dir == kLeft) ? a.cell.lower[dim] : a.cell.upper[dim];
        double planeB = (dir == kLeft) ? b.cell.upper[dim] : b.cell.lower[dim];
        if (planeA != planeB) {
            return false;
        } else {
            for (int i=0; i<N; ++i) {
                if (i != dim) {
                    if (!(b.cell.upper[i] > a.cell.lower[i] && b.cell.lower[i] < a.cell.upper[i])) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /** @brief Is node B a parent of node A? */
    template<int N>
    bool Quadtree<N>::isParent(Node& a, Node& b)
    {
        return (a.cell.lower >= b.cell.lower).all() && (a.cell.upper <= b.cell.upper).all();
    }

    /** @brief Search the tree with pruning based on a condition */
    template<int N>
    void Quadtree<N>::search(std::function<bool(Node&)> cond, std::function<void(Node&)> accum, int id)
    {
        if (cond(tree[id])) {
            if (!tree[id].isLeaf()) {
                for (NDArrayIterator<int,2,N> it = tree[id].children.begin(); it != tree[id].children.end(); ++it) {
                    search(cond, accum, *it);
                }
            } else {
                accum(tree[id]);
            }
        }
    }

    /** @brief DFS helper */
    template<int N>
    void Quadtree<N>::dfs(std::function<void(Node&)> f, int id)
    {
        f(tree[id]);
        if (!tree[id].isLeaf()) {
            for (NDArrayIterator<int,2,N> it = tree[id].children.begin(); it != tree[id].children.end(); ++it) {
                dfs(f, *it);
            }
        }
    }

    /** @brief Apply f to the tree in a breadth-first order */
    template<int N>
    void Quadtree<N>::bfs(std::function<void(Node&)> f)
    {
        std::queue<int> q;
        q.push(0);
        while (!q.empty())
        {
            int id = q.front();
            f(tree[id]);
            q.pop();
            if (!tree[id].isLeaf())
            {
                for (NDArrayIterator<int,2,N> it = tree[id].children.begin(); it != tree[id].children.end(); ++it) {
                    q.push(*it);
                }
            }
        }
    }
}

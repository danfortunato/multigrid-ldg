#ifndef QUADTREE
#define QUADTREE

#include <vector>
#include <stdexcept>

namespace LDG
{
	/** @brief An n-dimensional coordinate */
	template<unsigned int DIM>
	struct Coordinate<DIM>
	{
			/** @brief Empty constructor */
			Coordinate() { for (unsigned int i = 0; i < DIM; ++i) { x[i] = 0; } }
		    /** @brief Access operator */
		    inline double& operator[] (int i) { return x[i]; }
		    inline const double& operator[] (int i) const { return x[i]; }
			/** @brief Array of coordinate components */
			double x[DIM];
	};

	/** @brief Equals operator */
	template<unsigned int DIM>
	bool operator==(const Coordinate<DIM>& p, const Coordinate<DIM>& q) {
	    for (unsigned int i = 0; i < DIM; i++) {
	    	if (p[i] != q[i]) {
	    		return false;
	    	}
	    }
	    return true;
	}

	/** @brief Unequals operator */
	template<unsigned int DIM>
	bool operator!=(const Coordinate<DIM>& p, const Coordinate<DIM>& q) {
	    return !(p == q);
	}

	/** @brief An n-dimensional cell. The bounding box of the cell is specified
	 *         by the coordinates of its lower left and upper right corners.
	 */
	template<unsigned int DIM>
	struct Cell
	{
		/** @brief Construct a cell from bounding box coordinates */
		Cell(Coordinate<DIM> lower_, Coordinate<DIM> upper_) :
			lower(lower_), upper(upper_)
		{}

		/** @brief Compute the width of the cell in the dimension d */
		double width(unsigned int d) {
			if (d >= DIM) {
				throw std::invalid_argument("Requested dimension too large.");
			}
			return upper[d]-lower[d];
		}

		/** @brief Compute the volume of the cell */
		double volume() {
			double v = 1;
			for (int i=0; i<DIM; ++i) { v *= width(i); }
			return v;
		}

		/** @brief The lower left and upper right coordinates of the cell */
		Coordinate<DIM> lower, upper;
	};

	/** @brief An n-dimensional quadtree */
	template<class T, unsigned int DIM>
	class Quadtree
	{
		public:
			/** @brief A node in the tree */
			struct Node
			{
				T object;
				Cell<DIM> cell;
				Node* children[DIM];
			};

			/** @brief Construct an empty tree from a given domain
			 *
			 *  @param[in] domain : Cell representing the tree's domain
			 */
			Quadtree(Cell<DIM> domain);

			/** @brief Construct a tree from a given domain that is uniformly
			 *         refined n times.
			 *
			 *  @param[in] domain : Cell representing the tree's domain
			 *  @param[in] n      : Number of refinements
			 */
			Quadtree(Cell<DIM> domain, int n);

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

#endif

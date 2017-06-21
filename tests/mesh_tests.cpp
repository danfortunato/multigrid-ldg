#include <iostream>
#include "mesh.h"
#include "quadtree.h"

bool quadtree_test();

int main()
{
	bool pass = true;
	pass = pass && quadtree_test();

	std::cout << (pass ? "Success" : "Fail") << std::endl;

	return 0;
}

bool quadtree_test()
{
	bool pass = true;
	const int N = 2;

	// Test unit quadtree
	DG::Quadtree<N> q1;
	DG::Quadtree<N> q2(2);

	// Test quadtree from a domain
	DG::Cell<N> domain(DG::Coordinate<N>(-3,1), DG::Coordinate<N>(2,4));
	DG::Quadtree<N> q3(domain);
	DG::Quadtree<N> q4(domain, 2);

	return pass;
}

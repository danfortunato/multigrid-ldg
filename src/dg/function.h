#ifndef FUNCTION_H
#define FUNCTION_H

#include <functional>
#include <algorithm>
#include <cmath>
#include <string>
#include <fstream>
#include "mesh.h"
#include "ndarray.h"

namespace DG
{
    // Forward declaration
    template<int P, int N>
    class Mesh;

    /** @brief A DG function defined over a mesh */
    template<int P, int N>
    class Function
    {
        public:
            /** @brief Construct an empty function over a mesh */
            Function(const Mesh<P,N>& mesh_) :
                mesh(&mesh_)
            {
                coeffs.reserve(mesh->ne);
            }

            /** @brief Construct a constant function over a mesh */
            Function(const Mesh<P,N>& mesh_, double value) :
                mesh(&mesh_)
            {
                coeffs.resize(mesh->ne, NDArray<double,P,N>(value));
            }

            /** @brief Construct a function over the mesh from a given function handle.
             *         This will be the nodal interpolant. */
            Function(const Mesh<P,N>& mesh_, std::function<double(Tuple<double,N>)> f) :
                mesh(&mesh_)
            {
                coeffs.reserve(mesh->ne);
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (RangeIterator<P,N> it; it != Range<P,N>::end(); ++it) {
                        coeffs[elem](it.index()) = f(mesh->elements[elem].dgnodes(it.index()));
                    }
                }
            }

            /** @brief Evaluate the function */
            double operator()(Tuple<double,N> x)
            {
                // TODO
                return 0;
            }

            /** @brief Compute the L^2 norm */
            double l2_norm()
            {
                // TODO
                return mesh->l2_norm(*this);
            }

            /** @brief Compute the maximum norm */
            double max_norm()
            {
                double norm = 0;
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (RangeIterator<P,N> it; it != Range<P,N>::end(); ++it) {
                        norm = std::max(norm, std::abs(coeffs[elem](it.index())));
                    }
                }
                return norm;
            }

            /** @brief Write the function to a file */
            void write(const std::string& file)
            {
                std::ofstream ofs(file);
                ofs.precision(std::numeric_limits<double>::max_digits10);

                ofs << P << " " << N << " " << mesh->ne << std::endl;
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (RangeIterator<P,N> it; it != Range<P,N>::end(); ++it) {
                        for (int i=0; i<N; ++i) {
                            ofs << mesh->elements[elem].dgnodes(it.index())[i] << " ";
                        }
                        ofs << coeffs[elem](it.index()) << std::endl;
                    }
                    ofs << std::endl;
                }

                ofs.close();
            }

            /** The basis coefficients */
            std::vector<NDArray<double,P,N>> coeffs;

        private:
            /** The mesh this function lives on */
            const Mesh<P,N>* mesh;
    };

    //operator +, -, *, /, +=, ==, etc...
}

#endif

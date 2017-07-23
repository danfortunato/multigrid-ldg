#ifndef FUNCTION_H
#define FUNCTION_H

#include <functional>
#include <algorithm>
#include <limits>
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
                coeffs(mesh_.ne),
                mesh(&mesh_)
            {}

            /** @brief Construct a constant function over a mesh */
            Function(const Mesh<P,N>& mesh_, double value) :
                coeffs(mesh_.ne, NDArray<double,P,N>(value)),
                mesh(&mesh_)
            {}

            /** @brief Construct a function over the mesh from a given function handle.
             *         This will be the nodal interpolant. */
            Function(const Mesh<P,N>& mesh_, std::function<double(Tuple<double,N>)> f) :
                mesh(&mesh_)
            {
                coeffs.resize(mesh->ne);
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (RangeIterator<P,N> it; it != Range<P,N>::end(); ++it) {
                        coeffs[elem](it.index()) = f(mesh->elements[elem].dgnodes(it.index()));
                    }
                }
            }

            /** @brief The total number of coefficients in the function */
            int size() const
            {
                return coeffs.size() * coeffs[0].size();
            }

            /** @brief Access the coefficient data for a given element
             *
             *  @note If no element is specified, this will return a pointer to
             *        the beginning of the entire coefficient array.
             */
            double* data(int elem = 0)
            {
                return coeffs[elem].data();
            }


            /** @brief Access the coefficient data for a given element (const version)
             *
             *  @note If no element is specified, this will return a pointer to
             *        the beginning of the entire coefficient array.
             */
            const double* data(int elem = 0) const
            {
                // return coeffs[0].data();
                return coeffs[elem].data();
            }

            /** @brief Represent the function's coefficients as a Vector */
            Map<Vector> vec()
            {
                return Map<Vector>(data(), size(), 1);
            }

            /** @brief Represent the function's coefficients as a Vector (const version) */
            const Map<const Vector> vec() const
            {
                return Map<const Vector>(data(), size(), 1);
            }

            /** @brief Represent the function's coefficients for a given element as a Vector */
            Map<KronVec<P,N>> vec(int elem)
            {
                return Map<KronVec<P,N>>(data(elem));
            }

            /** @brief Represent the function's coefficients for a given element as a Vector (const version) */
            const Map<const KronVec<P,N>> vec(int elem) const
            {
                return Map<const KronVec<P,N>>(data(elem));
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

            /** @brief Make the function mean-zero by projecting out the mean */
            void meanZero()
            {
                double mean = 0;
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (RangeIterator<P,N> it; it != Range<P,N>::end(); ++it) {
                        mean += coeffs[elem](it.index());
                    }
                }
                mean /= NDArray<double,P,N>::size() * coeffs.size();
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (RangeIterator<P,N> it; it != Range<P,N>::end(); ++it) {
                        coeffs[elem](it.index()) -= mean;
                    }
                }
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

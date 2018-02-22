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
    template<int N, int P>
    class Mesh;

    /** @brief A DG function defined over a mesh */
    template<int N, int P>
    class Function
    {
        public:
            /** @brief Construct an empty function over a mesh */
            Function(const Mesh<N,P>& mesh_) :
                coeffs(mesh_.ne),
                mesh(&mesh_),
                vec_(data(), size(), 1),
                elemvec_(data())
            {}

            /** @brief Construct a constant function over a mesh */
            Function(const Mesh<N,P>& mesh_, double value) :
                coeffs(mesh_.ne, SimplexArray<double,N,P>(value)),
                mesh(&mesh_),
                vec_(data(), size(), 1),
                elemvec_(data())
            {}

            /** @brief Construct a function over the mesh from a given function handle.
             *         This will be the nodal interpolant. */
            Function(const Mesh<N,P>& mesh_, const std::function<double(Tuple<double,N>)>& f) :
                mesh(&mesh_),
                vec_(data(), size(), 1),
                elemvec_(data())
            {
                coeffs.resize(mesh->ne);
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                        coeffs[elem](it.linearIndex()) = f(mesh->elements[elem].dgnodes(it.linearIndex()));
                    }
                }
            }

            /** @brief Reset the function from a given constant */
            void reset(double value = 0)
            {
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                        coeffs[elem](it.linearIndex()) = value;
                    }
                }
            }

            /** @brief Reset the function from a given function handle */
            void reset(const std::function<double(Tuple<double,N>)>& f)
            {
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                        coeffs[elem](it.linearIndex()) = f(mesh->elements[elem].dgnodes(it.linearIndex()));
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
                return coeffs[elem].data();
            }

            /** @brief Represent the function's coefficients as a Vector */
            Map<Vector>& vec()
            {
                new (&vec_) Map<Vector>(data(), size(), 1);
                return vec_;
            }

            /** @brief Represent the function's coefficients as a Vector (const version) */
            const Map<const Vector>& vec() const
            {
                new (&vec_) Map<const Vector>(data(), size(), 1);
                return vec_;
            }

            /** @brief Represent the function's coefficients for a given element as a Vector */
            Map<SimplexVec<N,P>>& vec(int elem)
            {
                new (&elemvec_) Map<SimplexVec<N,P>>(data(elem));
                return elemvec_;
            }

            /** @brief Represent the function's coefficients for a given element as a Vector (const version) */
            const Map<const SimplexVec<N,P>>& vec(int elem) const
            {
                new (&elemvec_) Map<const SimplexVec<N,P>>(data(elem));
                return elemvec_;
            }

            /** @brief Compute the maximum norm */
            double max_norm()
            {
                double norm = 0;
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                        norm = std::max(norm, std::abs(coeffs[elem](it.linearIndex())));
                    }
                }
                return norm;
            }

            /** @brief Make the function mean-zero by projecting out the mean */
            void meanZero()
            {
                double mean = 0;
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                        mean += coeffs[elem](it.linearIndex());
                    }
                }
                mean /= SimplexArray<double,N,P>::size() * coeffs.size();
                for (int elem = 0; elem < mesh->ne; ++elem) {
                    for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                        coeffs[elem](it.linearIndex()) -= mean;
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
                    for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                        for (int i=0; i<N; ++i) {
                            ofs << mesh->elements[elem].dgnodes(it.index())[i] << " ";
                        }
                        ofs << coeffs[elem](it.linearIndex()) << std::endl;
                    }
                    ofs << std::endl;
                }

                ofs.close();
            }

            /** The basis coefficients */
            std::vector<SimplexArray<double,N,P>> coeffs;

        private:
            /** The mesh this function lives on */
            const Mesh<N,P>* mesh;
            Map<Vector> vec_;
            Map<SimplexVec<N,P>> elemvec_;
    };
}

#endif

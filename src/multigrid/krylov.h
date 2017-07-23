#ifndef KRYLOV_H
#define KRYLOV_H

#include "common.h"
#include "sparseblockmatrix.h"

namespace DG
{
    /** @brief Preconditioned conjugate gradient
     *
     *  @param[in]     A      : The matrix
     *  @param[in]     b      : The right-hand side
     *  @param[in,out] x      : The solution
     *  @param[in]     precon : The preconditioner
     *  @param[in]     tol    : The desired tolerance
     *  @param[in]     maxit  : The maximum number of iterations
     *
     *  @note A preconditioner is any object that implements operator()
     */
    template<int P, typename VectorType, typename PreconditionerType>
    void pcg(const SparseBlockMatrix<P>& A, const VectorType& b, VectorType& x, PreconditionerType&& precon, double tol=1e-8, int maxit=200)
    {
        int i;
        double delta_0, delta_new, delta_old, alpha, beta;
        Vector r, d, q, s;

        r = b;
        multiply_add_mv(-1.0, A, x, 1.0, r);
        d = precon(r);
        delta_new = r.dot(d);
        delta_0 = delta_new;

        i = 0;
        while (i < maxit && delta_new > tol * tol * delta_0) {
            multiply_mv(A, d, q);
            alpha = delta_new / d.dot(q);
            x += alpha * d;
            if (i % 50 == 0) {
                r = b;
                multiply_add_mv(-1.0, A, x, 1.0, r);
            } else {
                r -= alpha * q;
            }
            s = precon(r);
            delta_old = delta_new;
            delta_new = r.dot(s);
            beta = delta_new / delta_old;
            d = s + beta * d;
            ++i;
        }
    }

    /** @brief Conjugate gradient
     *
     *  @param[in]     A     : The matrix
     *  @param[in]     b     : The right-hand side
     *  @param[in,out] x     : The solution
     *  @param[in]     tol   : The desired tolerance
     *  @param[in]     maxit : The maximum number of iterations
     */
    template<int P, typename VectorType>
    void pcg(const SparseBlockMatrix<P>& A, const VectorType& b, VectorType& x, double tol=1e-8, int maxit=200)
    {
        pcg(A, b, x, [](const auto& x) { return x; }, tol, maxit);
    }
}

#endif

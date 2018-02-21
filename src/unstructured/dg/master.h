#ifndef MASTER_H
#define MASTER_H

#include <array>
#include <vector>
#include "common.h"
#include "range.h"
#include "ndarray.h"
#include "wireframe.h"

namespace DG
{
    /** @brief The master simplex */
    template<int N, int P>
    struct Master
    {
        /** Order of polynomial */
        static const int p = P-1;
        /** Number of nodes per element */
        static const int npl = ichoose(P+N-1,N);
        /** The local DG nodes */
        static const SimplexArray<Tuple<double,N>,N,P> nodes;
        /** Vandermonde matrix */
        static const SimplexMat<N,P> vandermonde;
        /** Differentiation Vandermonde matrices */
        static const std::array<SimplexMat<N,P>,N> dvandermonde;
        /** Mass matrix */
        static const SimplexMat<N,P> mass;
        /** Differentiation matrices */
        static const std::array<SimplexMat<N,P>,N> diff;
        /** The linearIndex-th node in a given simplex */
        static Tuple<double,N> dgnodes(double linearIndex, const Simplex<N>& simplex = Simplex<N>())
        {
            assert(0 <= linearIndex && linearIndex < npl);
            Tuple<double,N> local = nodes(linearIndex);
            return simplex.p[0].matrix() + simplex.jacobian_mat()*local.matrix();
        }
    };

    /** @brief Equispaced nodes on a simplex */
    template<int N, int P>
    SimplexArray<Tuple<double,N>,N,P> simplexNodes()
    {
        SimplexArray<Tuple<double,N>,N,P> nodes;
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            *it = it.index().template cast<double>() / (P-1);
        }
        return nodes;
    }

    /** @brief Compute the normalized Jacobi polynomials */
    template<int N, int P>
    SimplexVec<N,P> jacobi(int n, int a, int b, const SimplexVec<N,P>& x)
    {
        double a1  = a+1;
        double b1  = b+1;
        double ab  = a+b;
        double ab1 = a+b+1;

        Eigen::Matrix<double,ichoose(P+N-1,N),Eigen::Dynamic> PL(ichoose(P+N-1,N), n+1);

        // Initial values P_0(x) and P_1(x)
        double gamma0 = std::pow(2.0,ab1)/ab1*std::tgamma(a1)*std::tgamma(b1)/std::tgamma(ab1);
        SimplexVec<N,P> next = SimplexVec<N,P>::Constant(1.0/std::sqrt(gamma0));
        if (n == 0) {
            return next;
        } else {
            PL.col(0) = next;
        }

        double gamma1 = a1*b1/(ab+3.0)*gamma0;
        next = ((ab+2.0)*x.array()/2.0+(a-b)/2.0) / std::sqrt(gamma1);
        if (n == 1) {
            return next;
        } else {
            PL.col(1) = next;
        }

        // Repeat value in recurrence
        double aold = 2.0/(2.0+ab)*std::sqrt(a1*b1/(ab+3.0));

        // Forward recurrence using the symmetry of the recurrence
        for (int i=1; i<n; ++i) {
            double h1 = 2.0*i+ab;
            double anew = 2.0/(h1+2.0)*std::sqrt((i+1)*(i+ab1)*(i+a1)*(i+b1)/(h1+1.0)/(h1+3.0));
            double bnew = -(a*a-b*b)/h1/(h1+2.0);
            PL.col(i+1) = ((x.array()-bnew)*PL.col(i).array() - aold*PL.col(i-1).array()) / anew;
            aold = anew;
        }

        return PL.col(n);
    }

    /** @brief Compute the derivative of the normalized Jacobi polynomials */
    template<int N, int P>
    SimplexVec<N,P> djacobi(int n, int a, int b, const SimplexVec<N,P>& x)
    {
        if (n == 0) {
            return SimplexVec<N,P>::Zero();
        } else {
            return std::sqrt(n*(n+a+b+1)) * jacobi<N,P>(n-1, a+1, b+1, x);
        }
    }

    /** @brief Evaluate the porder-th Koornwinder polynomial at the given nodes */
    template<int N, int P>
    SimplexVec<N,P> koornwinder(const SimplexArray<Tuple<double,N>,N,P>& nodes, const Tuple<int,N>& porder)
    {
        // Map nodes from [0,1] to [-1,1]
        Eigen::Matrix<double,ichoose(P+N-1,N),N> X;
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            X.row(it.linearIndex()) = 2*(*it)-1;
        }

        // Map (x,y,z,...) to (a,b,c,...) coordinates
        Eigen::Matrix<double,ichoose(P+N-1,N),N> A;
        SimplexVec<N,P> denom;
        for (int i=0; i<N-1; ++i) {
            denom.setConstant(3-N+i);
            for (int j=i+1; j<N; ++j) {
                denom -= X.col(j);
            }
            for (int k=0; k<denom.size(); ++k) {
                A(k,i) = (denom[k]!=0) ? 2*(1.0+X(k,i))/denom[k]-1 : -1;
            }
        }
        A.col(N-1) = X.col(N-1);

        // Compute the product of the Jacobi polynomials
        SimplexVec<N,P> v = SimplexVec<N,P>::Ones();
        int poly  = 0;
        int power = 0;
        for (int i=0; i<N; ++i) {
            SimplexVec<N,P> a = A.col(i);
            v = v.cwiseProduct(jacobi<N,P>(porder[i], poly, 0, a));
            v = v.cwiseProduct(((1-a.array()).pow(power)).matrix());
            poly  += 2*porder[i]+1;
            power += porder[i];
        }

        // Normalize
        v *= std::pow(2.0,ichoose(N,2)/2.0);

        return v;
    }

    /** @brief Evaluate the derivative of the porder-th Koornwinder polynomial
     *         in the dim-th dimension at the given nodes */
    template<int N, int P>
    SimplexVec<N,P> dkoornwinder(const SimplexArray<Tuple<double,N>,N,P>& nodes, const Tuple<int,N>& porder, int dim)
    {
        // Map nodes from [0,1] to [-1,1]
        Eigen::Matrix<double,ichoose(P+N-1,N),N> X;
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            X.row(it.linearIndex()) = 2*(*it)-1;
        }

        // Map (x,y,z,...) to (a,b,c,...) coordinates
        Eigen::Matrix<double,ichoose(P+N-1,N),N> A;
        SimplexVec<N,P> denom;
        for (int i=0; i<N-1; ++i) {
            denom.setConstant(3-N+i);
            for (int j=i+1; j<N; ++j) {
                denom -= X.col(j);
            }
            for (int k=0; k<denom.size(); ++k) {
                A(k,i) = (denom[k]!=0) ? 2*(1.0+X(k,i))/denom[k]-1 : -1;
            }
        }
        A.col(N-1) = X.col(N-1);

        // Compute the derivative using the chain rule
        SimplexVec<N,P> dv = SimplexVec<N,P>::Zero();
        for (int d=0; d<dim+1; ++d) {

            // Compute d(a_d)/d(x_dim)
            SimplexVec<N,P> chain = SimplexVec<N,P>::Ones();
            if (d != dim) chain = 0.5*(1+A.col(d).array());
            int divide = 0;
            for (int i=d+1; i<N; ++i) divide|=1<<i;

            // Compute  d/d(a_d)
            SimplexVec<N,P> da  = SimplexVec<N,P>::Zero();
            SimplexVec<N,P> fac = SimplexVec<N,P>::Ones();
            int poly  = 0;
            int power = 0;
            for (int i=0; i<N; ++i) {
                SimplexVec<N,P> a = A.col(i);
                if (i != d) {
                    fac = fac.cwiseProduct(jacobi<N,P>(porder[i], poly, 0, a));
                    if (power>0) {
                        if (divide&(1<<i)) {
                            fac = 2*fac.cwiseProduct(((1-a.array()).pow(power-1)).matrix());
                        } else {
                            fac = fac.cwiseProduct(((1-a.array()).pow(power)).matrix());
                        }
                    }
                } else {
                    da = djacobi<N,P>(porder[i], poly, 0, a);
                    if (power>0) {
                        da = da.cwiseProduct(((1-a.array()).pow(power)).matrix());
                        da -= jacobi<N,P>(porder[i], poly, 0, a).cwiseProduct(power*((1-a.array()).pow(power-1)).matrix());
                    }
                }
                poly  += 2*porder[i]+1;
                power += porder[i];
            }

            // Add contribution from d(a_d)/d(x_dim) d/d(a_d)
            dv += chain.cwiseProduct(fac).cwiseProduct(da);
        }

        // Normalize
        dv *= std::pow(2.0,ichoose(N,2)/2.0);

        return dv;
    }

    /** @brief The Vandermonde matrix on the unit simplex */
    template<int N, int P>
    SimplexMat<N,P> simplexVandermonde()
    {
        SimplexMat<N,P> vandermonde;
        auto nodes = simplexNodes<N,P>();
        for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
            vandermonde.col(it.linearIndex()) = koornwinder(nodes, it.index());
        }
        return vandermonde;
    }

    /** @brief The differentiation Vandermonde matrices on the unit simplex */
    template<int N, int P>
    std::array<SimplexMat<N,P>,N> simplexDVandermonde()
    {
        std::array<SimplexMat<N,P>,N> dvandermonde;
        auto nodes = simplexNodes<N,P>();
        for (int i=0; i<N; ++i) {
            for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
                dvandermonde[i].col(it.linearIndex()) = dkoornwinder(nodes, it.index(), i);
            }
        }
        return dvandermonde;
    }

    /** @brief The mass matrix on the unit simplex */
    template<int N, int P>
    SimplexMat<N,P> simplexMass()
    {
        SimplexMat<N,P> V = simplexVandermonde<N,P>();
        SimplexMat<N,P> Vinv = V.inverse();
        return Vinv.transpose() * Vinv;
    }

    /** @brief The differentiation matrices on the unit simplex */
    template<int N, int P>
    std::array<SimplexMat<N,P>,N> simplexDiff()
    {
        SimplexMat<N,P> V = simplexVandermonde<N,P>();
        std::array<SimplexMat<N,P>,N> dV = simplexDVandermonde<N,P>();
        SimplexMat<N,P> Vinv = V.inverse();
        std::array<SimplexMat<N,P>,N> diff;
        for (int i=0; i<N; ++i) {
            diff[i] = dV[i] * Vinv;
        }
        return diff;
    }

    template<int N, int P>
    const SimplexArray<Tuple<double,N>,N,P> Master<N,P>::nodes = simplexNodes<N,P>();
    template<int N, int P>
    const SimplexMat<N,P> Master<N,P>::vandermonde = simplexVandermonde<N,P>();
    template<int N, int P>
    const std::array<SimplexMat<N,P>,N> Master<N,P>::dvandermonde = simplexDVandermonde<N,P>();
    template<int N, int P>
    const SimplexMat<N,P> Master<N,P>::mass = simplexMass<N,P>();
    template<int N, int P>
    const std::array<SimplexMat<N,P>,N> Master<N,P>::diff = simplexDiff<N,P>();
}

#endif

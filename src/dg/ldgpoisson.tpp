#include "range.h"

namespace DG
{
    template<int P, int N>
    void LDGPoisson<P,N>::discretize()
    {
        construct_mass_matrix();
        construct_broken_gradient();
        construct_lifting_terms();
    }

    /** @brief Construct the block diagonal mass matrices */
    template<int P, int N>
    void LDGPoisson<P,N>::construct_mass_matrix()
    {
        for (const auto& e : mesh->elements) {
            int j = e.lid;
            M_builder.setBlock(j, j, e.mass());
            for (int i=0; i<N; ++i) {
                int k = N*j+i;
                MM_builder.setBlock(k, k, e.mass());
            }
        }
    }

    /** @brief Construct the broken gradient operator */
    template<int P, int N>
    void LDGPoisson<P,N>::construct_broken_gradient()
    {
        KronMat<P,N> diff;
        // Create the differentiation matrix in each dimension
        for (int i=0; i<N; ++i) {
            diff = KronMat<P,N>::Zero();
            // Compute the tensor product of the 1D differentiation matrix
            // with the appropriate identity matrices
            for (RangeIterator<P,N> it; it != Range<P,N>::end(); ++it) {
                for (RangeIterator<P,N> jt; jt != Range<P,N>::end(); ++jt) {
                    // There is only a nonzero entry when the indices for all
                    // the identity matrices are on the diagonal
                    bool nonzero = true;
                    for (int k=0; k<N; ++k) {
                        if (k!=i && it(k)!=jt(k)) {
                            nonzero = false;
                            break;
                        }
                    }
                    if (nonzero) {
                        int bi = it.linearIndex();
                        int bj = jt.linearIndex();
                        diff(bi,bj) = GaussLobatto<P>::diff(it(i),jt(i));
                    }
                }
            }

            // Place this gradient component in the global matrix for all
            // elements
            for (const auto& e : mesh->elements) {
                int j = e.lid;
                double jac = 1/e.cell.width(i);
                G_builder.addToBlock(N*j+i, j, jac*diff);
            }
        }
    }

    /** @brief Construct the lifting operator and penalty terms */
    template<int P, int N>
    void LDGPoisson<P,N>::construct_lifting_terms()
    {
        KronMat<P,N> L;
        SliceMat<P,N> slice_l, slice_r;

        for (const auto& f : mesh->faces) {

            // Is this an interior face?
            if (f.interiorQ()) {

                // Can we use the 1D mass matrix?
                if (f.canonical) {

                    // Compute slicing matrices
                    slice_l.setZero();
                    slice_r.setZero();
                    // Loop over the rows of the slicing matrix
                    for (RangeIterator<P,N-1> it; it != Range<P,N-1>::end(); ++it) {
                        int i = it.linearIndex();
                        // Copy it into jt, leaving space for the extra dimension
                        RangeIterator<P,N> jt;
                        for (int k=0; k<f.dim; ++k) jt(k) = it(k);
                        for (int k=f.dim; k<N-1; ++k) jt(k+1) = it(k);
                        // The face nodes for the left element lie on its right side
                        jt(f.dim) = (f.normal[f.dim] > 0) ? P-1 : 0;
                        int j = jt.linearIndex();
                        slice_l(i,j) = 1;
                        // The face nodes for the right element lie on its left side
                        jt(f.dim) = (f.normal[f.dim] > 0) ? 0 : P-1;
                        j = jt.linearIndex();
                        slice_r(i,j) = 1;
                    }

                    /*** Lifting operator ***/
                    // Since the normals are the coordinate vectors, the only
                    // nonzero component of the normal dot product is the in
                    // dimension this face lives on
                    // Diagonal contribution
                    L = slice_r.transpose() * f.mass() * slice_r;
                    L *= f.normal[f.dim];
                    L = mesh->elements[f.right].invmass() * L;
                    G_builder.addToBlock(N*f.right+f.dim, f.right, L);
                    // Off-diagonal contribution
                    L = -slice_r.transpose() * f.mass() * slice_l;
                    L *= f.normal[f.dim];
                    L = mesh->elements[f.right].invmass() * L;
                    G_builder.addToBlock(N*f.right+f.dim, f.left, L);

                    /*** Penalty terms ***/
                    // Diagonal contribution
                    L = tau0 * slice_r.transpose() * f.mass() * slice_r;
                    L *= f.normal[f.dim];
                    T_builder.addToBlock(f.right, f.right, L);
                    L = tau0 * slice_l.transpose() * f.mass() * slice_l;
                    L *= f.normal[f.dim];
                    T_builder.addToBlock(f.left, f.left, L);
                    // Off-diagonal contribution
                    L = -tau0 * slice_r.transpose() * f.mass() * slice_l;
                    L *= f.normal[f.dim];
                    T_builder.addToBlock(f.right, f.left, L);
                    L = -tau0 * slice_l.transpose() * f.mass() * slice_r;
                    L *= f.normal[f.dim];
                    T_builder.addToBlock(f.left, f.right, L);

                } else {

                    // Compute matrices to evaluate at quadrature points
                    EvalMat<P,Q,N> phi_l, phi_r;
                    KronVec<Q,N-1> w;
                    for (RangeIterator<Q,N-1> it; it != Range<Q,N-1>::end(); ++it) {
                        int i = it.linearIndex();
                        phi_l.row(i) = LagrangePoly<P>::eval(f.xl(it.index()));
                        phi_r.row(i) = LagrangePoly<P>::eval(f.xr(it.index()));
                        w[i] = 1;
                        for (int k=0; k<N-1; ++k) {
                            w[i] *= Quadrature<Q>::weights[it(k)];
                        }
                    }
                    // Scale the quadrature weights according to the area of the face
                    w *= f.area();
                    KronDiag<Q,N-1> W = w.asDiagonal();

                    /*** Lifting operator ***/
                    // Diagonal contribution
                    L = phi_r.transpose() * W * phi_r;
                    L *= f.normal[f.dim];
                    L = mesh->elements[f.right].invmass() * L;
                    G_builder.addToBlock(N*f.right+f.dim, f.right, L);
                    // Off-diagonal contribution
                    L = -phi_r.transpose() * W * phi_l;
                    L *= f.normal[f.dim];
                    L = mesh->elements[f.right].invmass() * L;
                    G_builder.addToBlock(N*f.right+f.dim, f.left, L);

                    /*** Penalty terms ***/
                    // Diagonal contribution
                    L = tau0 * phi_r.transpose() * W * phi_r;
                    L *= f.normal[f.dim];
                    T_builder.addToBlock(f.right, f.right, L);
                    L = tau0 * phi_l.transpose() * W * phi_l;
                    L *= f.normal[f.dim];
                    T_builder.addToBlock(f.left, f.left, L);
                    // Off-diagonal contribution
                    L = -tau0 * phi_r.transpose() * W * phi_l;
                    L *= f.normal[f.dim];
                    T_builder.addToBlock(f.right, f.left, L);
                    L = -tau0 * phi_l.transpose() * W * phi_r;
                    L *= f.normal[f.dim];
                    T_builder.addToBlock(f.left, f.right, L);
                }

            } else {

                if (bcs.bcmap.at(f.boundary()).type == kDirichlet) {

                    if (f.canonical) {

                        slice_l.setZero();
                        // Loop over the rows of the slicing matrix
                        for (RangeIterator<P,N-1> it; it != Range<P,N-1>::end(); ++it) {
                            int i = it.linearIndex();
                            // Copy it into jt, leaving space for the extra dimension
                            RangeIterator<P,N> jt;
                            for (int k=0; k<f.dim; ++k) jt(k) = it(k);
                            for (int k=f.dim; k<N-1; ++k) jt(k+1) = it(k);
                            // The face nodes for the element lie on its boundary side
                            jt(f.dim) = (f.normal[f.dim] > 0) ? P-1 : 0;
                            int j = jt.linearIndex();
                            slice_l(i,j) = 1;
                        }

                        /*** Lifting operator ***/
                        // Diagonal contribution
                        L = -slice_l.transpose() * f.mass() * slice_l;
                        L *= f.normal[f.dim];
                        L = mesh->elements[f.left].invmass() * L;
                        G_builder.addToBlock(N*f.left+f.dim, f.left, L);

                        /*** Penalty terms ***/
                        // Diagonal contribution
                        L = tauD * slice_l.transpose() * f.mass() * slice_l;
                        L *= f.normal[f.dim];
                        T_builder.addToBlock(f.left, f.left, L);

                    } else {

                        // Compute matrices to evaluate at quadrature points
                        EvalMat<P,Q,N> phi_l;
                        KronVec<Q,N-1> w;
                        for (RangeIterator<Q,N-1> it; it != Range<Q,N-1>::end(); ++it) {
                            int i = it.linearIndex();
                            phi_l.row(i) = LagrangePoly<P>::eval(f.xl(it.index()));
                            w[i] = 1;
                            for (int k=0; k<N-1; ++k) {
                                w[i] *= Quadrature<Q>::weights[it(k)];
                            }
                        }
                        // Scale the quadrature weights according to the area of the face
                        w *= f.area();
                        KronDiag<Q,N-1> W = w.asDiagonal();

                        /*** Lifting operator ***/
                        // Diagonal contribution
                        L = -phi_l.transpose() * W * phi_l;
                        L *= f.normal[f.dim];
                        L = mesh->elements[f.left].invmass() * L;
                        G_builder.addToBlock(N*f.left+f.dim, f.left, L);

                        /*** Penalty terms ***/
                        // Diagonal contribution
                        L = tauD * phi_l.transpose() * W * phi_l;
                        L *= f.normal[f.dim];
                        T_builder.addToBlock(f.left, f.left, L);
                    }
                }
            }
        }
    }

    template<int P, int N>
    void LDGPoisson<P,N>::assemble_system()
    {
        M = M_builder.build();
        G = G_builder.build();
        T = T_builder.build();

        // Create the discrete Laplacian
        // A = G^T M G + T ...
    }
}

#ifndef MASTER_H
#define MASTER_H

#include <array>
#include "common.h"
#include "range.h"
#include "ndarray.h"
#include "geometry.h"

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
        /** Volume of master simplex */
        static constexpr double volume = 1.0/ifac(N);
        /** The local DG nodes */
        static const SimplexArray<Tuple<double,N>,N,P> nodes;
        /** Vandermonde matrix */
        static const SimplexMat<N,P> vandermonde;
        /** Inverse Vandermonde matrix */
        static const SimplexMat<N,P> invvandermonde;
        /** Differentiation Vandermonde matrices */
        static const std::array<SimplexMat<N,P>,N> dvandermonde;
        /** Mass matrix */
        static const SimplexMat<N,P> mass;
        /** Inverse mass matrix */
        static const SimplexMat<N,P> invmass;
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

    template<int N, int P, int Q>
    struct Phi
    {
        /** @brief Basis functions evaluated at quadrature points */
        static const SimplexElemQuadMat<N,P,Q> phi;
    };

    /** Hack to initialize the number of quadrature points */
    template<int N, int P>
    constexpr int quadratureSize() { return 0; }

    template<int N, int P>
    struct Quadrature
    {
        /** Size */
        static const int size = quadratureSize<N,P>();
        /** Quadrature points */
        static const std::array<Tuple<double,N>,size> nodes;
        /** Quadrature weights */
        static const std::array<double,size> weights;
    };

    /** @brief Equispaced nodes on a simplex */
    template<int N, int P>
    typename std::enable_if<P!=1, SimplexArray<Tuple<double,N>,N,P>>::type simplexNodes()
    {
        SimplexArray<Tuple<double,N>,N,P> nodes;
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            *it = it.index().template cast<double>() / (P-1);
        }
        return nodes;
    }

    /** @brief Equispaced nodes on a simplex */
    template<int N, int P>
    typename std::enable_if<P==1, SimplexArray<Tuple<double,N>,N,1>>::type simplexNodes()
    {
        return {1.0/3.0*Tuple<double,N>::Ones()};
    }

    /** @brief Compute the normalized Jacobi polynomials */
    template<int Q>
    Vec<Q> jacobi(int n, int a, int b, const Vec<Q>& x)
    {
        double a1  = a+1;
        double b1  = b+1;
        double ab  = a+b;
        double ab1 = a+b+1;

        Eigen::Matrix<double,Q,Eigen::Dynamic> PL(Q, n+1);

        // Initial values P_0(x) and P_1(x)
        double gamma0 = std::pow(2.0,ab1)/ab1*std::tgamma(a1)*std::tgamma(b1)/std::tgamma(ab1);
        Vec<Q> next = Vec<Q>::Constant(1.0/std::sqrt(gamma0));
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
    template<int Q>
    Vec<Q> djacobi(int n, int a, int b, const Vec<Q>& x)
    {
        if (n == 0) {
            return Vec<Q>::Zero();
        } else {
            return std::sqrt(n*(n+a+b+1)) * jacobi(n-1, a+1, b+1, x);
        }
    }

    /** @brief Evaluate the porder-th Koornwinder polynomial at the given nodes */
    template<int N, int Q>
    Vec<Q> koornwinder(const std::array<Tuple<double,N>,Q>& nodes, const Tuple<int,N>& porder)
    {
        // Map nodes from [0,1] to [-1,1]
        Mat<Q,N> X;
        for (int i=0; i<(int)nodes.size(); ++i) {
            X.row(i) = 2*nodes[i]-1;
        }

        // Map (x,y,z,...) to (a,b,c,...) coordinates
        Mat<Q,N> A;
        Vec<Q> denom;
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
        Vec<Q> v = Vec<Q>::Ones();
        int poly  = 0;
        int power = 0;
        for (int i=0; i<N; ++i) {
            Vec<Q> a = A.col(i);
            v = v.cwiseProduct(jacobi(porder[i], poly, 0, a));
            v = v.cwiseProduct(((1-a.array()).pow(power)).matrix());
            poly  += 2*porder[i]+1;
            power += porder[i];
        }

        // Normalize
        v *= std::pow(2.0,ichoose(N+1,2)/2.0);

        return v;
    }

    /** @brief Evaluate the porder-th Koornwinder polynomial at the given nodes */
    template<int N, int P>
    SimplexVec<N,P> koornwinder(const SimplexArray<Tuple<double,N>,N,P>& nodes, const Tuple<int,N>& porder)
    {
        std::array<Tuple<double,N>,nodes.size()>* x = reinterpret_cast<std::array<Tuple<double,N>,nodes.size()>*>(const_cast<Tuple<double,N>*>(nodes.data()));
        return koornwinder<N,nodes.size()>(*x, porder);
    }

    /** @brief Evaluate the derivative of the porder-th Koornwinder polynomial
     *         in the dim-th dimension at the given nodes */
    template<int N, int Q>
    Vec<Q> dkoornwinder(const std::array<Tuple<double,N>,Q>& nodes, const Tuple<int,N>& porder, int dim)
    {
        // Map nodes from [0,1] to [-1,1]
        Mat<Q,N> X;
        for (int i=0; i<(int)nodes.size(); ++i) {
            X.row(i) = 2*nodes[i]-1;
        }

        // Map (x,y,z,...) to (a,b,c,...) coordinates
        Mat<Q,N> A;
        Vec<Q> denom;
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
        Vec<Q> dv = Vec<Q>::Zero();
        for (int d=0; d<dim+1; ++d) {

            // Compute d(a_d)/d(x_dim)
            Vec<Q> chain = Vec<Q>::Ones();
            if (d != dim) chain = 0.5*(1+A.col(d).array());
            int divide = 0;
            for (int i=d+1; i<N; ++i) divide|=1<<i;

            // Compute  d/d(a_d)
            Vec<Q> da  = Vec<Q>::Zero();
            Vec<Q> fac = Vec<Q>::Ones();
            int poly  = 0;
            int power = 0;
            for (int i=0; i<N; ++i) {
                Vec<Q> a = A.col(i);
                if (i != d) {
                    fac = fac.cwiseProduct(jacobi(porder[i], poly, 0, a));
                    if (power>0) {
                        if (divide&(1<<i)) {
                            fac = 2*fac.cwiseProduct(((1-a.array()).pow(power-1)).matrix());
                        } else {
                            fac = fac.cwiseProduct(((1-a.array()).pow(power)).matrix());
                        }
                    }
                } else {
                    da = djacobi(porder[i], poly, 0, a);
                    if (power>0) {
                        da = da.cwiseProduct(((1-a.array()).pow(power)).matrix());
                        da -= jacobi(porder[i], poly, 0, a).cwiseProduct(power*((1-a.array()).pow(power-1)).matrix());
                    }
                }
                poly  += 2*porder[i]+1;
                power += porder[i];
            }

            // Add contribution from d(a_d)/d(x_dim) d/d(a_d)
            dv += chain.cwiseProduct(fac).cwiseProduct(da);
        }

        // Normalize
        dv *= std::pow(2.0,ichoose(N+1,2)/2.0+1);

        return dv;
    }

    /** @brief Evaluate the derivative of the porder-th Koornwinder polynomial
     *         in the dim-th dimension at the given nodes */
    template<int N, int P>
    SimplexVec<N,P> dkoornwinder(const SimplexArray<Tuple<double,N>,N,P>& nodes, const Tuple<int,N>& porder, int dim)
    {
        std::array<Tuple<double,N>,nodes.size()>* x = reinterpret_cast<std::array<Tuple<double,N>,nodes.size()>*>(const_cast<Tuple<double,N>*>(nodes.data()));
        return dkoornwinder<N,nodes.size()>(*x, porder, dim);
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
        return Vinv.transpose() * Vinv / Master<N,P>::volume;
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

    template<int N, int P, int Q>
    SimplexElemQuadMat<N,P,Q> simplexPhi()
    {
        SimplexElemQuadMat<N,P,Q> phi;
        for (SimplexRangeIterator<N,P> it; it != SimplexRange<N,P>::end(); ++it) {
            phi.row(it.linearIndex()) = koornwinder<N,Quadrature<N,Q>::size>(Quadrature<N,Q>::nodes, it.index());
        }
        return simplexVandermonde<N,P>().inverse().transpose() * phi;
    }

    template<int N, int P>
    const SimplexArray<Tuple<double,N>,N,P> Master<N,P>::nodes = simplexNodes<N,P>();
    template<int N, int P>
    const SimplexMat<N,P> Master<N,P>::vandermonde = simplexVandermonde<N,P>();
    template<int N, int P>
    const SimplexMat<N,P> Master<N,P>::invvandermonde = simplexVandermonde<N,P>().inverse();
    template<int N, int P>
    const std::array<SimplexMat<N,P>,N> Master<N,P>::dvandermonde = simplexDVandermonde<N,P>();
    template<int N, int P>
    const SimplexMat<N,P> Master<N,P>::mass = simplexMass<N,P>();
    template<int N, int P>
    const SimplexMat<N,P> Master<N,P>::invmass = simplexMass<N,P>().inverse();
    template<int N, int P>
    const std::array<SimplexMat<N,P>,N> Master<N,P>::diff = simplexDiff<N,P>();

    template<int N, int P, int Q>
    const SimplexElemQuadMat<N,P,Q> Phi<N,P,Q>::phi = simplexPhi<N,P,Q>();

    /*** 1D ***/

    // Quadrature sizes
    template<> constexpr int quadratureSize<1,1>()  { return 1;  }
    template<> constexpr int quadratureSize<1,2>()  { return 2;  }
    template<> constexpr int quadratureSize<1,3>()  { return 3;  }
    template<> constexpr int quadratureSize<1,4>()  { return 4;  }
    template<> constexpr int quadratureSize<1,5>()  { return 5;  }
    template<> constexpr int quadratureSize<1,6>()  { return 6;  }
    template<> constexpr int quadratureSize<1,7>()  { return 7;  }
    template<> constexpr int quadratureSize<1,8>()  { return 8;  }
    template<> constexpr int quadratureSize<1,9>()  { return 9;  }
    template<> constexpr int quadratureSize<1,10>() { return 10; }
    template<> constexpr int quadratureSize<1,11>() { return 11; }
    template<> constexpr int quadratureSize<1,12>() { return 12; }

    // Quadrature nodes
    template<> const std::array<Tuple<double,1>,Quadrature<1,1>::size>  Quadrature<1,1>::nodes  = {Tuple<double,1>{0.50000000000000000000000000000000}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,2>::size>  Quadrature<1,2>::nodes  = {Tuple<double,1>{0.21132486540518711774542560974902}, Tuple<double,1>{0.78867513459481288225457439025098}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,3>::size>  Quadrature<1,3>::nodes  = {Tuple<double,1>{0.11270166537925831148207346002176}, Tuple<double,1>{0.50000000000000000000000000000000}, Tuple<double,1>{0.88729833462074168851792653997824}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,4>::size>  Quadrature<1,4>::nodes  = {Tuple<double,1>{0.069431844202973712388026755553595}, Tuple<double,1>{0.33000947820757186759866712044838}, Tuple<double,1>{0.66999052179242813240133287955162}, Tuple<double,1>{0.93056815579702628761197324444640}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,5>::size>  Quadrature<1,5>::nodes  = {Tuple<double,1>{0.046910077030668003601186560850304}, Tuple<double,1>{0.23076534494715845448184278964990}, Tuple<double,1>{0.50000000000000000000000000000000}, Tuple<double,1>{0.76923465505284154551815721035010}, Tuple<double,1>{0.95308992296933199639881343914970}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,6>::size>  Quadrature<1,6>::nodes  = {Tuple<double,1>{0.033765242898423986093849222753003}, Tuple<double,1>{0.16939530676686774316930020249005}, Tuple<double,1>{0.38069040695840154568474913915964}, Tuple<double,1>{0.61930959304159845431525086084036}, Tuple<double,1>{0.83060469323313225683069979750995}, Tuple<double,1>{0.96623475710157601390615077724700}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,7>::size>  Quadrature<1,7>::nodes  = {Tuple<double,1>{0.025446043828620737736905157976074}, Tuple<double,1>{0.12923440720030278006806761335961}, Tuple<double,1>{0.29707742431130141654669679396152}, Tuple<double,1>{0.50000000000000000000000000000000}, Tuple<double,1>{0.70292257568869858345330320603848}, Tuple<double,1>{0.87076559279969721993193238664039}, Tuple<double,1>{0.97455395617137926226309484202393}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,8>::size>  Quadrature<1,8>::nodes  = {Tuple<double,1>{0.019855071751231884158219565715264}, Tuple<double,1>{0.10166676129318663020422303176208}, Tuple<double,1>{0.23723379504183550709113047540538}, Tuple<double,1>{0.40828267875217509753026192881991}, Tuple<double,1>{0.59171732124782490246973807118009}, Tuple<double,1>{0.76276620495816449290886952459462}, Tuple<double,1>{0.89833323870681336979577696823792}, Tuple<double,1>{0.98014492824876811584178043428474}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,9>::size>  Quadrature<1,9>::nodes  = {Tuple<double,1>{0.015919880246186955082211898548164}, Tuple<double,1>{0.081984446336682102850285105965133}, Tuple<double,1>{0.19331428364970480134564898032926}, Tuple<double,1>{0.33787328829809553548073099267833}, Tuple<double,1>{0.50000000000000000000000000000000}, Tuple<double,1>{0.66212671170190446451926900732167}, Tuple<double,1>{0.80668571635029519865435101967074}, Tuple<double,1>{0.91801555366331789714971489403487}, Tuple<double,1>{0.98408011975381304491778810145184}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,10>::size> Quadrature<1,10>::nodes = {Tuple<double,1>{0.013046735741414139961017993957774}, Tuple<double,1>{0.067468316655507744633951655788253}, Tuple<double,1>{0.16029521585048779688283631744256}, Tuple<double,1>{0.28330230293537640460036702841711}, Tuple<double,1>{0.42556283050918439455758699943514}, Tuple<double,1>{0.57443716949081560544241300056486}, Tuple<double,1>{0.71669769706462359539963297158289}, Tuple<double,1>{0.83970478414951220311716368255744}, Tuple<double,1>{0.93253168334449225536604834421175}, Tuple<double,1>{0.98695326425858586003898200604223}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,11>::size> Quadrature<1,11>::nodes = {Tuple<double,1>{0.010885670926971503598030999438571}, Tuple<double,1>{0.056468700115952350462421115348036}, Tuple<double,1>{0.13492399721297533795329187398442}, Tuple<double,1>{0.24045193539659409203713716527070}, Tuple<double,1>{0.36522842202382751383423400729957}, Tuple<double,1>{0.50000000000000000000000000000000}, Tuple<double,1>{0.63477157797617248616576599270043}, Tuple<double,1>{0.75954806460340590796286283472930}, Tuple<double,1>{0.86507600278702466204670812601558}, Tuple<double,1>{0.94353129988404764953757888465196}, Tuple<double,1>{0.98911432907302849640196900056143}};
    template<> const std::array<Tuple<double,1>,Quadrature<1,12>::size> Quadrature<1,12>::nodes = {Tuple<double,1>{0.0092196828766403746547254549253596}, Tuple<double,1>{0.047941371814762571660767066940452}, Tuple<double,1>{0.11504866290284765648155308339359}, Tuple<double,1>{0.20634102285669127635164879052973}, Tuple<double,1>{0.31608425050090990312365423167814}, Tuple<double,1>{0.43738329574426554226377931526807}, Tuple<double,1>{0.56261670425573445773622068473193}, Tuple<double,1>{0.68391574949909009687634576832186}, Tuple<double,1>{0.79365897714330872364835120947027}, Tuple<double,1>{0.88495133709715234351844691660641}, Tuple<double,1>{0.95205862818523742833923293305955}, Tuple<double,1>{0.99078031712335962534527454507464}};

    // Quadrature weights
    template<> const std::array<double,Quadrature<1,1>::size>  Quadrature<1,1>::weights  = {1.0000000000000000000000000000000};
    template<> const std::array<double,Quadrature<1,2>::size>  Quadrature<1,2>::weights  = {0.50000000000000000000000000000000, 0.50000000000000000000000000000000};
    template<> const std::array<double,Quadrature<1,3>::size>  Quadrature<1,3>::weights  = {0.27777777777777777777777777777778, 0.44444444444444444444444444444444, 0.27777777777777777777777777777778};
    template<> const std::array<double,Quadrature<1,4>::size>  Quadrature<1,4>::weights  = {0.17392742256872692868653197461100, 0.32607257743127307131346802538900, 0.32607257743127307131346802538900, 0.17392742256872692868653197461100};
    template<> const std::array<double,Quadrature<1,5>::size>  Quadrature<1,5>::weights  = {0.11846344252809454375713202035996, 0.23931433524968323402064575741782, 0.28444444444444444444444444444444, 0.23931433524968323402064575741782, 0.11846344252809454375713202035996};
    template<> const std::array<double,Quadrature<1,6>::size>  Quadrature<1,6>::weights  = {0.085662246189585172520148071086366, 0.18038078652406930378491675691886, 0.23395696728634552369493517199478, 0.23395696728634552369493517199478, 0.18038078652406930378491675691886, 0.085662246189585172520148071086366};
    template<> const std::array<double,Quadrature<1,7>::size>  Quadrature<1,7>::weights  = {0.064742483084434846635305716339541, 0.13985269574463833395073388571189, 0.19091502525255947247518488774449, 0.20897959183673469387755102040816, 0.19091502525255947247518488774449, 0.13985269574463833395073388571189, 0.064742483084434846635305716339541};
    template<> const std::array<double,Quadrature<1,8>::size>  Quadrature<1,8>::weights  = {0.050614268145188129576265677154981, 0.11119051722668723527217799721312, 0.15685332293894364366898110099330, 0.18134189168918099148257522463860, 0.18134189168918099148257522463860, 0.15685332293894364366898110099330, 0.11119051722668723527217799721312, 0.050614268145188129576265677154981};
    template<> const std::array<double,Quadrature<1,9>::size>  Quadrature<1,9>::weights  = {0.040637194180787205985946079055262, 0.090324080347428702029236015621456, 0.13030534820146773115937143470932, 0.15617353852000142003431520329222, 0.16511967750062988158226253464349, 0.15617353852000142003431520329222, 0.13030534820146773115937143470932, 0.090324080347428702029236015621456, 0.040637194180787205985946079055262};
    template<> const std::array<double,Quadrature<1,10>::size> Quadrature<1,10>::weights = {0.033335672154344068796784404946666, 0.074725674575290296572888169828849, 0.10954318125799102199776746711408, 0.13463335965499817754561346078473, 0.14776211235737643508694649732567, 0.14776211235737643508694649732567, 0.13463335965499817754561346078473, 0.10954318125799102199776746711408, 0.074725674575290296572888169828849, 0.033335672154344068796784404946666};
    template<> const std::array<double,Quadrature<1,11>::size> Quadrature<1,11>::weights = {0.027834283558086833241376860221274, 0.062790184732452312317347149611970, 0.093145105463867125713048820715828, 0.11659688229599523995926185242159, 0.13140227225512333109034443494525, 0.13646254338895031535724176416817, 0.13140227225512333109034443494525, 0.11659688229599523995926185242159, 0.093145105463867125713048820715828, 0.062790184732452312317347149611970, 0.027834283558086833241376860221274};
    template<> const std::array<double,Quadrature<1,12>::size> Quadrature<1,12>::weights = {0.023587668193255913597307980742509, 0.053469662997659215480127359096998, 0.080039164271673113167326264771680, 0.10158371336153296087453222790490, 0.11674626826917740438042494946244, 0.12457352290670139250028121802148, 0.12457352290670139250028121802148, 0.11674626826917740438042494946244, 0.10158371336153296087453222790490, 0.080039164271673113167326264771680, 0.053469662997659215480127359096998, 0.023587668193255913597307980742509};

    /*** 2D ***/

    // Quadrature sizes
    template<> constexpr int quadratureSize<2,1>()  { return 1;  }
    template<> constexpr int quadratureSize<2,2>()  { return 1;  }
    template<> constexpr int quadratureSize<2,3>()  { return 3;  }
    template<> constexpr int quadratureSize<2,4>()  { return 4;  }
    template<> constexpr int quadratureSize<2,5>()  { return 6;  }
    template<> constexpr int quadratureSize<2,6>()  { return 7;  }
    template<> constexpr int quadratureSize<2,7>()  { return 12; }
    template<> constexpr int quadratureSize<2,8>()  { return 13; }
    template<> constexpr int quadratureSize<2,9>()  { return 16; }
    template<> constexpr int quadratureSize<2,10>() { return 19; }
    template<> constexpr int quadratureSize<2,11>() { return 25; }

    // Quadrature nodes
    template<> const std::array<Tuple<double,2>,Quadrature<2,1>::size>  Quadrature<2,1>::nodes  = {{{3.33333333333333333E-01, 3.33333333333333333E-01}}};
    template<> const std::array<Tuple<double,2>,Quadrature<2,2>::size>  Quadrature<2,2>::nodes  = {{{3.33333333333333333E-01, 3.33333333333333333E-01}}};
    template<> const std::array<Tuple<double,2>,Quadrature<2,3>::size>  Quadrature<2,3>::nodes  = {{{6.66666666666666667E-01, 1.66666666666666667E-01}, {1.66666666666666667E-01, 6.66666666666666667E-01}, {1.66666666666666667E-01, 1.66666666666666667E-01}}};
    template<> const std::array<Tuple<double,2>,Quadrature<2,4>::size>  Quadrature<2,4>::nodes  = {{{3.33333333333333333E-01, 3.33333333333333333E-01}, {6.00000000000000000E-01, 2.00000000000000000E-01}, {2.00000000000000000E-01, 6.00000000000000000E-01}, {2.00000000000000000E-01, 2.00000000000000000E-01}}};
    template<> const std::array<Tuple<double,2>,Quadrature<2,5>::size>  Quadrature<2,5>::nodes  = {{{1.081030181680700E-01, 4.459484909159650E-01}, {4.459484909159650E-01, 1.081030181680700E-01}, {4.459484909159650E-01, 4.459484909159650E-01}, {8.168475729804590E-01, 9.157621350977100E-02}, {9.157621350977100E-02, 8.168475729804590E-01}, {9.157621350977100E-02, 9.157621350977100E-02}}};
    template<> const std::array<Tuple<double,2>,Quadrature<2,6>::size>  Quadrature<2,6>::nodes  = {{{3.33333333333333E-01, 3.33333333333333E-01}, {5.97158717897700E-02, 4.70142064105115E-01}, {4.70142064105115E-01, 5.97158717897700E-02}, {4.70142064105115E-01, 4.70142064105115E-01}, {7.97426985353087E-01, 1.01286507323456E-01}, {1.01286507323456E-01, 7.97426985353087E-01}, {1.01286507323456E-01, 1.01286507323456E-01}}};
    template<> const std::array<Tuple<double,2>,Quadrature<2,7>::size>  Quadrature<2,7>::nodes  = {{{5.01426509658179E-01, 2.49286745170910E-01}, {2.49286745170910E-01, 5.01426509658179E-01}, {2.49286745170910E-01, 2.49286745170910E-01}, {8.73821971016996E-01, 6.30890144915020E-02}, {6.30890144915020E-02, 8.73821971016996E-01}, {6.30890144915020E-02, 6.30890144915020E-02}, {5.31450498448170E-02, 3.10352451033784E-01}, {6.36502499121399E-01, 5.31450498448170E-02}, {3.10352451033784E-01, 6.36502499121399E-01}, {5.31450498448170E-02, 6.36502499121399E-01}, {6.36502499121399E-01, 3.10352451033784E-01}, {3.10352451033784E-01, 5.31450498448170E-02}}};
    template<> const std::array<Tuple<double,2>,Quadrature<2,8>::size>  Quadrature<2,8>::nodes  = {{{3.33333333333333E-01, 3.33333333333333E-01}, {4.79308067841920E-01, 2.60345966079040E-01}, {2.60345966079040E-01, 4.79308067841920E-01}, {2.60345966079040E-01, 2.60345966079040E-01}, {8.69739794195568E-01, 6.51301029022160E-02}, {6.51301029022160E-02, 8.69739794195568E-01}, {6.51301029022160E-02, 6.51301029022160E-02}, {4.86903154253160E-02, 3.12865496004874E-01}, {6.38444188569810E-01, 4.86903154253160E-02}, {3.12865496004874E-01, 6.38444188569810E-01}, {4.86903154253160E-02, 6.38444188569810E-01}, {6.38444188569810E-01, 3.12865496004874E-01}, {3.12865496004874E-01, 4.86903154253160E-02}}};
    template<> const std::array<Tuple<double,2>,Quadrature<2,9>::size>  Quadrature<2,9>::nodes  = {{{3.33333333333333E-01, 3.33333333333333E-01}, {8.14148234145540E-02, 4.59292588292723E-01}, {4.59292588292723E-01, 8.14148234145540E-02}, {4.59292588292723E-01, 4.59292588292723E-01}, {6.58861384496480E-01, 1.70569307751760E-01}, {1.70569307751760E-01, 6.58861384496480E-01}, {1.70569307751760E-01, 1.70569307751760E-01}, {8.98905543365938E-01, 5.05472283170310E-02}, {5.05472283170310E-02, 8.98905543365938E-01}, {5.05472283170310E-02, 5.05472283170310E-02}, {8.39477740995800E-03, 2.63112829634638E-01}, {7.28492392955404E-01, 8.39477740995800E-03}, {2.63112829634638E-01, 7.28492392955404E-01}, {8.39477740995800E-03, 7.28492392955404E-01}, {7.28492392955404E-01, 2.63112829634638E-01}, {2.63112829634638E-01, 8.39477740995800E-03}}};
    template<> const std::array<Tuple<double,2>,Quadrature<2,10>::size> Quadrature<2,10>::nodes = {{{3.33333333333333E-01, 3.33333333333333E-01}, {2.06349616025250E-02, 4.89682519198738E-01}, {4.89682519198738E-01, 2.06349616025250E-02}, {4.89682519198738E-01, 4.89682519198738E-01}, {1.25820817014127E-01, 4.37089591492937E-01}, {4.37089591492937E-01, 1.25820817014127E-01}, {4.37089591492937E-01, 4.37089591492937E-01}, {6.23592928761935E-01, 1.88203535619033E-01}, {1.88203535619033E-01, 6.23592928761935E-01}, {1.88203535619033E-01, 1.88203535619033E-01}, {9.10540973211095E-01, 4.47295133944530E-02}, {4.47295133944530E-02, 9.10540973211095E-01}, {4.47295133944530E-02, 4.47295133944530E-02}, {3.68384120547360E-02, 2.21962989160766E-01}, {7.41198598784498E-01, 3.68384120547360E-02}, {2.21962989160766E-01, 7.41198598784498E-01}, {3.68384120547360E-02, 7.41198598784498E-01}, {7.41198598784498E-01, 2.21962989160766E-01}, {2.21962989160766E-01, 3.68384120547360E-02}}};
    template<> const std::array<Tuple<double,2>,Quadrature<2,11>::size> Quadrature<2,11>::nodes = {{{3.33333333333333E-01, 3.33333333333333E-01}, {2.88447332326850E-02, 4.85577633383657E-01}, {4.85577633383657E-01, 2.88447332326850E-02}, {4.85577633383657E-01, 4.85577633383657E-01}, {7.81036849029926E-01, 1.09481575485037E-01}, {1.09481575485037E-01, 7.81036849029926E-01}, {1.09481575485037E-01, 1.09481575485037E-01}, {1.41707219414880E-01, 3.07939838764121E-01}, {5.50352941820999E-01, 1.41707219414880E-01}, {3.07939838764121E-01, 5.50352941820999E-01}, {1.41707219414880E-01, 5.50352941820999E-01}, {5.50352941820999E-01, 3.07939838764121E-01}, {3.07939838764121E-01, 1.41707219414880E-01}, {2.50035347626860E-02, 2.46672560639903E-01}, {7.28323904597411E-01, 2.50035347626860E-02}, {2.46672560639903E-01, 7.28323904597411E-01}, {2.50035347626860E-02, 7.28323904597411E-01}, {7.28323904597411E-01, 2.46672560639903E-01}, {2.46672560639903E-01, 2.50035347626860E-02}, {9.54081540029900E-03, 6.68032510122000E-02}, {9.23655933587500E-01, 9.54081540029900E-03}, {6.68032510122000E-02, 9.23655933587500E-01}, {9.54081540029900E-03, 9.23655933587500E-01}, {9.23655933587500E-01, 6.68032510122000E-02}, {6.68032510122000E-02, 9.54081540029900E-03}}};

    // Quadrature weights
    template<> const std::array<double,Quadrature<2,1>::size>  Quadrature<2,1>::weights  = {1.00000000000000000};
    template<> const std::array<double,Quadrature<2,2>::size>  Quadrature<2,2>::weights  = {1.00000000000000000};
    template<> const std::array<double,Quadrature<2,3>::size>  Quadrature<2,3>::weights  = {0.33333333333333333, 0.33333333333333333, 0.33333333333333333};
    template<> const std::array<double,Quadrature<2,4>::size>  Quadrature<2,4>::weights  = {-0.56250000000000000, 0.52083333333333333, 0.52083333333333333, 0.52083333333333333};
    template<> const std::array<double,Quadrature<2,5>::size>  Quadrature<2,5>::weights  = {0.223381589678011, 0.223381589678011, 0.223381589678011, 0.109951743655322, 0.109951743655322, 0.109951743655322};
    template<> const std::array<double,Quadrature<2,6>::size>  Quadrature<2,6>::weights  = {0.225000000000000, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827};
    template<> const std::array<double,Quadrature<2,7>::size>  Quadrature<2,7>::weights  = {0.116786275726379, 0.116786275726379, 0.116786275726379, 0.050844906370207, 0.050844906370207, 0.050844906370207, 0.082851075618374, 0.082851075618374, 0.082851075618374, 0.082851075618374, 0.082851075618374, 0.082851075618374};
    template<> const std::array<double,Quadrature<2,8>::size>  Quadrature<2,8>::weights  = {-0.149570044467682, 0.175615257433208, 0.175615257433208, 0.175615257433208, 0.053347235608838, 0.053347235608838, 0.053347235608838, 0.077113760890257, 0.077113760890257, 0.077113760890257, 0.077113760890257, 0.077113760890257, 0.077113760890257};
    template<> const std::array<double,Quadrature<2,9>::size>  Quadrature<2,9>::weights  = {0.144315607677787, 0.095091634267285, 0.095091634267285, 0.095091634267285, 0.103217370534718, 0.103217370534718, 0.103217370534718, 0.032458497623198, 0.032458497623198, 0.032458497623198, 0.027230314174435, 0.027230314174435, 0.027230314174435, 0.027230314174435, 0.027230314174435, 0.027230314174435};
    template<> const std::array<double,Quadrature<2,10>::size> Quadrature<2,10>::weights = {0.097135796282799, 0.031334700227139, 0.031334700227139, 0.031334700227139, 0.077827541004774, 0.077827541004774, 0.077827541004774, 0.079647738927210, 0.079647738927210, 0.079647738927210, 0.025577675658698, 0.025577675658698, 0.025577675658698, 0.043283539377289, 0.043283539377289, 0.043283539377289, 0.043283539377289, 0.043283539377289, 0.043283539377289};
    template<> const std::array<double,Quadrature<2,11>::size> Quadrature<2,11>::weights = {0.090817990382754, 0.036725957756467, 0.036725957756467, 0.036725957756467, 0.045321059435528, 0.045321059435528, 0.045321059435528, 0.072757916845420, 0.072757916845420, 0.072757916845420, 0.072757916845420, 0.072757916845420, 0.072757916845420, 0.028327242531057, 0.028327242531057, 0.028327242531057, 0.028327242531057, 0.028327242531057, 0.028327242531057, 0.009421666963733, 0.009421666963733, 0.009421666963733, 0.009421666963733, 0.009421666963733, 0.009421666963733};

    /*** 3D ***/

    // Quadrature sizes
    template<> constexpr int quadratureSize<3,1>()  { return 4;   }
    template<> constexpr int quadratureSize<3,2>()  { return 4;   }
    template<> constexpr int quadratureSize<3,3>()  { return 4;   }
    template<> constexpr int quadratureSize<3,4>()  { return 5;   }
    template<> constexpr int quadratureSize<3,5>()  { return 11;  }
    template<> constexpr int quadratureSize<3,6>()  { return 14;  }
    template<> constexpr int quadratureSize<3,7>()  { return 24;  }
    template<> constexpr int quadratureSize<3,8>()  { return 31;  }
    template<> constexpr int quadratureSize<3,9>()  { return 43;  }
    template<> constexpr int quadratureSize<3,10>() { return 53;  }
    template<> constexpr int quadratureSize<3,11>() { return 126; }

    // Quadrature nodes
    template<> const std::array<Tuple<double,3>,Quadrature<3,1>::size>  Quadrature<3,1>::nodes  = {{{0.13819660112501, 0.13819660112501, 0.13819660112501}, {0.58541019662497, 0.13819660112501, 0.13819660112501}, {0.13819660112501, 0.58541019662497, 0.13819660112501}, {0.13819660112501, 0.13819660112501, 0.58541019662497}}};
    template<> const std::array<Tuple<double,3>,Quadrature<3,2>::size>  Quadrature<3,2>::nodes  = {{{0.13819660112501, 0.13819660112501, 0.13819660112501}, {0.58541019662497, 0.13819660112501, 0.13819660112501}, {0.13819660112501, 0.58541019662497, 0.13819660112501}, {0.13819660112501, 0.13819660112501, 0.58541019662497}}};
    template<> const std::array<Tuple<double,3>,Quadrature<3,3>::size>  Quadrature<3,3>::nodes  = {{{0.13819660112501, 0.13819660112501, 0.13819660112501}, {0.58541019662497, 0.13819660112501, 0.13819660112501}, {0.13819660112501, 0.58541019662497, 0.13819660112501}, {0.13819660112501, 0.13819660112501, 0.58541019662497}}};
    template<> const std::array<Tuple<double,3>,Quadrature<3,4>::size>  Quadrature<3,4>::nodes  = {{{0.25000000000000, 0.25000000000000, 0.25000000000000}, {0.16666666666667, 0.16666666666667, 0.16666666666667}, {0.16666666666667, 0.16666666666667, 0.50000000000000}, {0.16666666666667, 0.50000000000000, 0.16666666666667}, {0.50000000000000, 0.16666666666667, 0.16666666666667}}};
    template<> const std::array<Tuple<double,3>,Quadrature<3,5>::size>  Quadrature<3,5>::nodes  = {{{0.25000000000000, 0.25000000000000, 0.25000000000000}, {0.07142857142857, 0.07142857142857, 0.07142857142857}, {0.07142857142857, 0.07142857142857, 0.78571428571429}, {0.07142857142857, 0.78571428571429, 0.07142857142857}, {0.78571428571429, 0.07142857142857, 0.07142857142857}, {0.39940357616680, 0.39940357616680, 0.10059642383320}, {0.39940357616680, 0.10059642383320, 0.39940357616680}, {0.10059642383320, 0.39940357616680, 0.39940357616680}, {0.39940357616680, 0.10059642383320, 0.10059642383320}, {0.10059642383320, 0.39940357616680, 0.10059642383320}, {0.10059642383320, 0.10059642383320, 0.39940357616680}}};
    template<> const std::array<Tuple<double,3>,Quadrature<3,6>::size>  Quadrature<3,6>::nodes  = {{{0.09273525031089, 0.09273525031089, 0.09273525031089}, {0.72179424906733, 0.09273525031089, 0.09273525031089}, {0.09273525031089, 0.72179424906733, 0.09273525031089}, {0.09273525031089, 0.09273525031089, 0.72179424906733}, {0.31088591926330, 0.31088591926330, 0.31088591926330}, {0.06734224221010, 0.31088591926330, 0.31088591926330}, {0.31088591926330, 0.06734224221010, 0.31088591926330}, {0.31088591926330, 0.31088591926330, 0.06734224221010}, {0.45449629587435, 0.45449629587435, 0.04550370412565}, {0.45449629587435, 0.04550370412565, 0.45449629587435}, {0.04550370412565, 0.45449629587435, 0.45449629587435}, {0.45449629587435, 0.04550370412565, 0.04550370412565}, {0.04550370412565, 0.45449629587435, 0.04550370412565}, {0.04550370412565, 0.04550370412565, 0.45449629587435}}};
    template<> const std::array<Tuple<double,3>,Quadrature<3,7>::size>  Quadrature<3,7>::nodes  = {{{0.21460287125915, 0.21460287125915, 0.21460287125915}, {0.35619138622254, 0.21460287125915, 0.21460287125915}, {0.21460287125915, 0.35619138622254, 0.21460287125915}, {0.21460287125915, 0.21460287125915, 0.35619138622254}, {0.04067395853461, 0.04067395853461, 0.04067395853461}, {0.87797812439617, 0.04067395853461, 0.04067395853461}, {0.04067395853461, 0.87797812439617, 0.04067395853461}, {0.04067395853461, 0.04067395853461, 0.87797812439617}, {0.32233789014228, 0.32233789014228, 0.32233789014228}, {0.03298632957317, 0.32233789014228, 0.32233789014228}, {0.32233789014228, 0.03298632957317, 0.32233789014228}, {0.32233789014228, 0.32233789014228, 0.03298632957317}, {0.06366100187502, 0.06366100187502, 0.26967233145832}, {0.06366100187502, 0.26967233145832, 0.06366100187502}, {0.06366100187502, 0.06366100187502, 0.60300566479165}, {0.06366100187502, 0.60300566479165, 0.06366100187502}, {0.06366100187502, 0.26967233145832, 0.60300566479165}, {0.06366100187502, 0.60300566479165, 0.26967233145832}, {0.26967233145832, 0.06366100187502, 0.06366100187502}, {0.26967233145832, 0.06366100187502, 0.60300566479165}, {0.26967233145832, 0.60300566479165, 0.06366100187502}, {0.60300566479165, 0.06366100187502, 0.26967233145832}, {0.60300566479165, 0.06366100187502, 0.06366100187502}, {0.60300566479165, 0.26967233145832, 0.06366100187502}}};
    template<> const std::array<Tuple<double,3>,Quadrature<3,8>::size>  Quadrature<3,8>::nodes  = {{{0.50000000000000, 0.50000000000000, 0}, {0.50000000000000, 0, 0.50000000000000}, {0, 0.50000000000000, 0.50000000000000}, {0, 0, 0.50000000000000}, {0, 0.50000000000000, 0}, {0.50000000000000, 0, 0}, {0.25000000000000, 0.25000000000000, 0.25000000000000}, {0.07821319233032, 0.07821319233032, 0.07821319233032}, {0.07821319233032, 0.07821319233032, 0.76536042300905}, {0.07821319233032, 0.76536042300905, 0.07821319233032}, {0.76536042300905, 0.07821319233032, 0.07821319233032}, {0.12184321666391, 0.12184321666391, 0.12184321666391}, {0.12184321666391, 0.12184321666391, 0.63447035000828}, {0.12184321666391, 0.63447035000828, 0.12184321666391}, {0.63447035000828, 0.12184321666391, 0.12184321666391}, {0.33253916444642, 0.33253916444642, 0.33253916444642}, {0.33253916444642, 0.33253916444642, 0.00238250666074}, {0.33253916444642, 0.00238250666074, 0.33253916444642}, {0.00238250666074, 0.33253916444642, 0.33253916444642}, {0.10000000000000, 0.10000000000000, 0.20000000000000}, {0.10000000000000, 0.20000000000000, 0.10000000000000}, {0.10000000000000, 0.10000000000000, 0.60000000000000}, {0.10000000000000, 0.60000000000000, 0.10000000000000}, {0.10000000000000, 0.20000000000000, 0.60000000000000}, {0.10000000000000, 0.60000000000000, 0.20000000000000}, {0.20000000000000, 0.10000000000000, 0.10000000000000}, {0.20000000000000, 0.10000000000000, 0.60000000000000}, {0.20000000000000, 0.60000000000000, 0.10000000000000}, {0.60000000000000, 0.10000000000000, 0.20000000000000}, {0.60000000000000, 0.10000000000000, 0.10000000000000}, {0.60000000000000, 0.20000000000000, 0.10000000000000}}};
    template<> const std::array<Tuple<double,3>,Quadrature<3,9>::size>  Quadrature<3,9>::nodes  = {{{0.25000000000000, 0.25000000000000, 0.25000000000000}, {0.20682993161067, 0.20682993161067, 0.20682993161067}, {0.20682993161067, 0.20682993161067, 0.37951020516798}, {0.20682993161067, 0.37951020516798, 0.20682993161067}, {0.37951020516798, 0.20682993161067, 0.20682993161067}, {0.08210358831055, 0.08210358831055, 0.08210358831055}, {0.08210358831055, 0.08210358831055, 0.75368923506836}, {0.08210358831055, 0.75368923506836, 0.08210358831055}, {0.75368923506836, 0.08210358831055, 0.08210358831055}, {0.00578195050520, 0.00578195050520, 0.00578195050520}, {0.00578195050520, 0.00578195050520, 0.98265414848441}, {0.00578195050520, 0.98265414848441, 0.00578195050520}, {0.98265414848441, 0.00578195050520, 0.00578195050520}, {0.05053274001889, 0.05053274001889, 0.44946725998111}, {0.05053274001889, 0.44946725998111, 0.05053274001889}, {0.44946725998111, 0.05053274001889, 0.05053274001889}, {0.05053274001889, 0.44946725998111, 0.44946725998111}, {0.44946725998111, 0.05053274001889, 0.44946725998111}, {0.44946725998111, 0.44946725998111, 0.05053274001889}, {0.22906653611681, 0.22906653611681, 0.03563958278853}, {0.22906653611681, 0.03563958278853, 0.22906653611681}, {0.22906653611681, 0.22906653611681, 0.50622734497784}, {0.22906653611681, 0.50622734497784, 0.22906653611681}, {0.22906653611681, 0.03563958278853, 0.50622734497784}, {0.22906653611681, 0.50622734497784, 0.03563958278853}, {0.03563958278853, 0.22906653611681, 0.22906653611681}, {0.03563958278853, 0.22906653611681, 0.50622734497784}, {0.03563958278853, 0.50622734497784, 0.22906653611681}, {0.50622734497784, 0.22906653611681, 0.03563958278853}, {0.50622734497784, 0.22906653611681, 0.22906653611681}, {0.50622734497784, 0.03563958278853, 0.22906653611681}, {0.03660774955320, 0.03660774955320, 0.19048604193463}, {0.03660774955320, 0.19048604193463, 0.03660774955320}, {0.03660774955320, 0.03660774955320, 0.73629845895897}, {0.03660774955320, 0.73629845895897, 0.03660774955320}, {0.03660774955320, 0.19048604193463, 0.73629845895897}, {0.03660774955320, 0.73629845895897, 0.19048604193463}, {0.19048604193463, 0.03660774955320, 0.03660774955320}, {0.19048604193463, 0.03660774955320, 0.73629845895897}, {0.19048604193463, 0.73629845895897, 0.03660774955320}, {0.73629845895897, 0.03660774955320, 0.19048604193463}, {0.73629845895897, 0.03660774955320, 0.03660774955320}, {0.73629845895897, 0.19048604193463, 0.03660774955320}}};
    template<> const std::array<Tuple<double,3>,Quadrature<3,10>::size> Quadrature<3,10>::nodes = {{{0.25000000000000, 0.25000000000000, 0.25000000000000}, {0.04835103854974, 0.04835103854974, 0.04835103854974}, {0.04835103854974, 0.04835103854974, 0.85494688435079}, {0.04835103854974, 0.85494688435079, 0.04835103854974}, {0.85494688435079, 0.04835103854974, 0.04835103854974}, {0.32457928011788, 0.32457928011788, 0.32457928011788}, {0.32457928011788, 0.32457928011788, 0.02626215964635}, {0.32457928011788, 0.02626215964635, 0.32457928011788}, {0.02626215964635, 0.32457928011788, 0.32457928011788}, {0.11461654022399, 0.11461654022399, 0.11461654022399}, {0.11461654022399, 0.11461654022399, 0.65615037932801}, {0.11461654022399, 0.65615037932801, 0.11461654022399}, {0.65615037932801, 0.11461654022399, 0.11461654022399}, {0.22548995191151, 0.22548995191151, 0.22548995191151}, {0.22548995191151, 0.22548995191151, 0.32353014426546}, {0.22548995191151, 0.32353014426546, 0.22548995191151}, {0.32353014426546, 0.22548995191151, 0.22548995191151}, {0.13162780924687, 0.13162780924687, 0.08366470161718}, {0.13162780924687, 0.08366470161718, 0.13162780924687}, {0.13162780924687, 0.13162780924687, 0.65307967988908}, {0.13162780924687, 0.65307967988908, 0.13162780924687}, {0.13162780924687, 0.08366470161718, 0.65307967988908}, {0.13162780924687, 0.65307967988908, 0.08366470161718}, {0.08366470161718, 0.13162780924687, 0.13162780924687}, {0.08366470161718, 0.13162780924687, 0.65307967988908}, {0.08366470161718, 0.65307967988908, 0.13162780924687}, {0.65307967988908, 0.13162780924687, 0.08366470161718}, {0.65307967988908, 0.13162780924687, 0.13162780924687}, {0.65307967988908, 0.08366470161718, 0.13162780924687}, {0.43395146141141, 0.43395146141141, 0.10776985954943}, {0.43395146141141, 0.10776985954943, 0.43395146141141}, {0.43395146141141, 0.43395146141141, 0.02432721762776}, {0.43395146141141, 0.02432721762776, 0.43395146141141}, {0.43395146141141, 0.10776985954943, 0.02432721762776}, {0.43395146141141, 0.02432721762776, 0.10776985954943}, {0.10776985954943, 0.43395146141141, 0.43395146141141}, {0.10776985954943, 0.43395146141141, 0.02432721762776}, {0.10776985954943, 0.02432721762776, 0.43395146141141}, {0.02432721762776, 0.43395146141141, 0.10776985954943}, {0.02432721762776, 0.43395146141141, 0.43395146141141}, {0.02432721762776, 0.10776985954943, 0.43395146141141}, {-0.00137627731814, -0.00137627731814, 0.27655347263681}, {-0.00137627731814, 0.27655347263681, -0.00137627731814}, {-0.00137627731814, -0.00137627731814, 0.72619908199947}, {-0.00137627731814, 0.72619908199947, -0.00137627731814}, {-0.00137627731814, 0.27655347263681, 0.72619908199947}, {-0.00137627731814, 0.72619908199947, 0.27655347263681}, {0.27655347263681, -0.00137627731814, -0.00137627731814}, {0.27655347263681, -0.00137627731814, 0.72619908199947}, {0.27655347263681, 0.72619908199947, -0.00137627731814}, {0.72619908199947, -0.00137627731814, 0.27655347263681}, {0.72619908199947, -0.00137627731814, -0.00137627731814}, {0.72619908199947, 0.27655347263681, -0.00137627731814}}};
    template<> const std::array<Tuple<double,3>,Quadrature<3,11>::size> Quadrature<3,11>::nodes = {{{0.07142857142857, 0.07142857142857, 0.78571428571429}, {0.07142857142857, 0.21428571428571, 0.64285714285714}, {0.07142857142857, 0.35714285714286, 0.50000000000000}, {0.07142857142857, 0.50000000000000, 0.35714285714286}, {0.07142857142857, 0.64285714285714, 0.21428571428571}, {0.07142857142857, 0.78571428571429, 0.07142857142857}, {0.21428571428571, 0.07142857142857, 0.64285714285714}, {0.21428571428571, 0.21428571428571, 0.50000000000000}, {0.21428571428571, 0.35714285714286, 0.35714285714286}, {0.21428571428571, 0.50000000000000, 0.21428571428571}, {0.21428571428571, 0.64285714285714, 0.07142857142857}, {0.35714285714286, 0.07142857142857, 0.50000000000000}, {0.35714285714286, 0.21428571428571, 0.35714285714286}, {0.35714285714286, 0.35714285714286, 0.21428571428571}, {0.35714285714286, 0.50000000000000, 0.07142857142857}, {0.50000000000000, 0.07142857142857, 0.35714285714286}, {0.50000000000000, 0.21428571428571, 0.21428571428571}, {0.50000000000000, 0.35714285714286, 0.07142857142857}, {0.64285714285714, 0.07142857142857, 0.21428571428571}, {0.64285714285714, 0.21428571428571, 0.07142857142857}, {0.78571428571429, 0.07142857142857, 0.07142857142857}, {0.07142857142857, 0.07142857142857, 0.64285714285714}, {0.07142857142857, 0.21428571428571, 0.50000000000000}, {0.07142857142857, 0.35714285714286, 0.35714285714286}, {0.07142857142857, 0.50000000000000, 0.21428571428571}, {0.07142857142857, 0.64285714285714, 0.07142857142857}, {0.21428571428571, 0.07142857142857, 0.50000000000000}, {0.21428571428571, 0.21428571428571, 0.35714285714286}, {0.21428571428571, 0.35714285714286, 0.21428571428571}, {0.21428571428571, 0.50000000000000, 0.07142857142857}, {0.35714285714286, 0.07142857142857, 0.35714285714286}, {0.35714285714286, 0.21428571428571, 0.21428571428571}, {0.35714285714286, 0.35714285714286, 0.07142857142857}, {0.50000000000000, 0.07142857142857, 0.21428571428571}, {0.50000000000000, 0.21428571428571, 0.07142857142857}, {0.64285714285714, 0.07142857142857, 0.07142857142857}, {0.07142857142857, 0.07142857142857, 0.50000000000000}, {0.07142857142857, 0.21428571428571, 0.35714285714286}, {0.07142857142857, 0.35714285714286, 0.21428571428571}, {0.07142857142857, 0.50000000000000, 0.07142857142857}, {0.21428571428571, 0.07142857142857, 0.35714285714286}, {0.21428571428571, 0.21428571428571, 0.21428571428571}, {0.21428571428571, 0.35714285714286, 0.07142857142857}, {0.35714285714286, 0.07142857142857, 0.21428571428571}, {0.35714285714286, 0.21428571428571, 0.07142857142857}, {0.50000000000000, 0.07142857142857, 0.07142857142857}, {0.07142857142857, 0.07142857142857, 0.35714285714286}, {0.07142857142857, 0.21428571428571, 0.21428571428571}, {0.07142857142857, 0.35714285714286, 0.07142857142857}, {0.21428571428571, 0.07142857142857, 0.21428571428571}, {0.21428571428571, 0.21428571428571, 0.07142857142857}, {0.35714285714286, 0.07142857142857, 0.07142857142857}, {0.07142857142857, 0.07142857142857, 0.21428571428571}, {0.07142857142857, 0.21428571428571, 0.07142857142857}, {0.21428571428571, 0.07142857142857, 0.07142857142857}, {0.07142857142857, 0.07142857142857, 0.07142857142857}, {0.08333333333333, 0.08333333333333, 0.75000000000000}, {0.08333333333333, 0.25000000000000, 0.58333333333333}, {0.08333333333333, 0.41666666666667, 0.41666666666667}, {0.08333333333333, 0.58333333333333, 0.25000000000000}, {0.08333333333333, 0.75000000000000, 0.08333333333333}, {0.25000000000000, 0.08333333333333, 0.58333333333333}, {0.25000000000000, 0.25000000000000, 0.41666666666667}, {0.25000000000000, 0.41666666666667, 0.25000000000000}, {0.25000000000000, 0.58333333333333, 0.08333333333333}, {0.41666666666667, 0.08333333333333, 0.41666666666667}, {0.41666666666667, 0.25000000000000, 0.25000000000000}, {0.41666666666667, 0.41666666666667, 0.08333333333333}, {0.58333333333333, 0.08333333333333, 0.25000000000000}, {0.58333333333333, 0.25000000000000, 0.08333333333333}, {0.75000000000000, 0.08333333333333, 0.08333333333333}, {0.08333333333333, 0.08333333333333, 0.58333333333333}, {0.08333333333333, 0.25000000000000, 0.41666666666667}, {0.08333333333333, 0.41666666666667, 0.25000000000000}, {0.08333333333333, 0.58333333333333, 0.08333333333333}, {0.25000000000000, 0.08333333333333, 0.41666666666667}, {0.25000000000000, 0.25000000000000, 0.25000000000000}, {0.25000000000000, 0.41666666666667, 0.08333333333333}, {0.41666666666667, 0.08333333333333, 0.25000000000000}, {0.41666666666667, 0.25000000000000, 0.08333333333333}, {0.58333333333333, 0.08333333333333, 0.08333333333333}, {0.08333333333333, 0.08333333333333, 0.41666666666667}, {0.08333333333333, 0.25000000000000, 0.25000000000000}, {0.08333333333333, 0.41666666666667, 0.08333333333333}, {0.25000000000000, 0.08333333333333, 0.25000000000000}, {0.25000000000000, 0.25000000000000, 0.08333333333333}, {0.41666666666667, 0.08333333333333, 0.08333333333333}, {0.08333333333333, 0.08333333333333, 0.25000000000000}, {0.08333333333333, 0.25000000000000, 0.08333333333333}, {0.25000000000000, 0.08333333333333, 0.08333333333333}, {0.08333333333333, 0.08333333333333, 0.08333333333333}, {0.10000000000000, 0.10000000000000, 0.70000000000000}, {0.10000000000000, 0.30000000000000, 0.50000000000000}, {0.10000000000000, 0.50000000000000, 0.30000000000000}, {0.10000000000000, 0.70000000000000, 0.10000000000000}, {0.30000000000000, 0.10000000000000, 0.50000000000000}, {0.30000000000000, 0.30000000000000, 0.30000000000000}, {0.30000000000000, 0.50000000000000, 0.10000000000000}, {0.50000000000000, 0.10000000000000, 0.30000000000000}, {0.50000000000000, 0.30000000000000, 0.10000000000000}, {0.70000000000000, 0.10000000000000, 0.10000000000000}, {0.10000000000000, 0.10000000000000, 0.50000000000000}, {0.10000000000000, 0.30000000000000, 0.30000000000000}, {0.10000000000000, 0.50000000000000, 0.10000000000000}, {0.30000000000000, 0.10000000000000, 0.30000000000000}, {0.30000000000000, 0.30000000000000, 0.10000000000000}, {0.50000000000000, 0.10000000000000, 0.10000000000000}, {0.10000000000000, 0.10000000000000, 0.30000000000000}, {0.10000000000000, 0.30000000000000, 0.10000000000000}, {0.30000000000000, 0.10000000000000, 0.10000000000000}, {0.10000000000000, 0.10000000000000, 0.10000000000000}, {0.12500000000000, 0.12500000000000, 0.62500000000000}, {0.12500000000000, 0.37500000000000, 0.37500000000000}, {0.12500000000000, 0.62500000000000, 0.12500000000000}, {0.37500000000000, 0.12500000000000, 0.37500000000000}, {0.37500000000000, 0.37500000000000, 0.12500000000000}, {0.62500000000000, 0.12500000000000, 0.12500000000000}, {0.12500000000000, 0.12500000000000, 0.37500000000000}, {0.12500000000000, 0.37500000000000, 0.12500000000000}, {0.37500000000000, 0.12500000000000, 0.12500000000000}, {0.12500000000000, 0.12500000000000, 0.12500000000000}, {0.16666666666667, 0.16666666666667, 0.50000000000000}, {0.16666666666667, 0.50000000000000, 0.16666666666667}, {0.50000000000000, 0.16666666666667, 0.16666666666667}, {0.16666666666667, 0.16666666666667, 0.16666666666667}, {0.25000000000000, 0.25000000000000, 0.25000000000000}}};

    // Quadrature weights
    template<> const std::array<double,Quadrature<3,1>::size>  Quadrature<3,1>::weights  = {0.25000000000000, 0.25000000000000, 0.25000000000000, 0.25000000000000};
    template<> const std::array<double,Quadrature<3,2>::size>  Quadrature<3,2>::weights  = {0.25000000000000, 0.25000000000000, 0.25000000000000, 0.25000000000000};
    template<> const std::array<double,Quadrature<3,3>::size>  Quadrature<3,3>::weights  = {0.25000000000000, 0.25000000000000, 0.25000000000000, 0.25000000000000};
    template<> const std::array<double,Quadrature<3,4>::size>  Quadrature<3,4>::weights  = {-0.80000000000000, 0.45000000000000, 0.45000000000000, 0.45000000000000, 0.45000000000000};
    template<> const std::array<double,Quadrature<3,5>::size>  Quadrature<3,5>::weights  = {-0.07893333333333, 0.04573333333333, 0.04573333333333, 0.04573333333333, 0.04573333333333, 0.14933333333333, 0.14933333333333, 0.14933333333333, 0.14933333333333, 0.14933333333333, 0.14933333333333};
    template<> const std::array<double,Quadrature<3,6>::size>  Quadrature<3,6>::weights  = {0.07349304311637, 0.07349304311637, 0.07349304311637, 0.07349304311637, 0.11268792571802, 0.11268792571802, 0.11268792571802, 0.11268792571802, 0.04254602077708, 0.04254602077708, 0.04254602077708, 0.04254602077708, 0.04254602077708, 0.04254602077703};
    template<> const std::array<double,Quadrature<3,7>::size>  Quadrature<3,7>::weights  = {0.03992275025817, 0.03992275025817, 0.03992275025817, 0.03992275025817, 0.01007721105532, 0.01007721105532, 0.01007721105532, 0.01007721105532, 0.05535718154366, 0.05535718154366, 0.05535718154366, 0.05535718154366, 0.04821428571429, 0.04821428571429, 0.04821428571429, 0.04821428571429, 0.04821428571429, 0.04821428571429, 0.04821428571429, 0.04821428571429, 0.04821428571429, 0.04821428571429, 0.04821428571429, 0.04821428571428};
    template<> const std::array<double,Quadrature<3,8>::size>  Quadrature<3,8>::weights  = {0.00582010582011, 0.00582010582011, 0.00582010582011, 0.00582010582011, 0.00582010582011, 0.00582010582011, 0.10958534079666, 0.06359964914649, 0.06359964914649, 0.06359964914649, 0.06359964914649, -0.37510644068602, -0.37510644068602, -0.37510644068602, -0.37510644068602, 0.02934855157844, 0.02934855157844, 0.02934855157844, 0.02934855157844, 0.16534391534393, 0.16534391534393, 0.16534391534393, 0.16534391534393, 0.16534391534393, 0.16534391534393, 0.16534391534393, 0.16534391534393, 0.16534391534393, 0.16534391534393, 0.16534391534393, 0.16534391534386};
    template<> const std::array<double,Quadrature<3,9>::size>  Quadrature<3,9>::weights  = {-0.12300113195185, 0.08550183493721, 0.08550183493721, 0.08550183493721, 0.08550183493721, 0.01180219987880, 0.01180219987880, 0.01180219987880, 0.01180219987880, 0.00101900465456, 0.00101900465456, 0.00101900465456, 0.00101900465456, 0.02747810294681, 0.02747810294681, 0.02747810294681, 0.02747810294681, 0.02747810294681, 0.02747810294681, 0.03422691485209, 0.03422691485209, 0.03422691485209, 0.03422691485209, 0.03422691485209, 0.03422691485209, 0.03422691485209, 0.03422691485209, 0.03422691485209, 0.03422691485209, 0.03422691485209, 0.03422691485209, 0.01284311484697, 0.01284311484697, 0.01284311484697, 0.01284311484697, 0.01284311484697, 0.01284311484697, 0.01284311484697, 0.01284311484697, 0.01284311484697, 0.01284311484697, 0.01284311484697, 0.01284311484690};
    template<> const std::array<double,Quadrature<3,10>::size> Quadrature<3,10>::weights = {-0.82679422995669, 0.01119201941451, 0.01119201941451, 0.01119201941451, 0.01119201941451, 0.02585654381696, 0.02585654381696, 0.02585654381696, 0.02585654381696, -0.54110859888723, -0.54110859888723, -0.54110859888723, -0.54110859888723, 0.26803545721508, 0.26803545721508, 0.26803545721508, 0.26803545721508, 0.20820243530731, 0.20820243530731, 0.20820243530731, 0.20820243530731, 0.20820243530731, 0.20820243530731, 0.20820243530731, 0.20820243530731, 0.20820243530731, 0.20820243530731, 0.20820243530731, 0.20820243530731, 0.02011550341596, 0.02011550341596, 0.02011550341596, 0.02011550341596, 0.02011550341596, 0.02011550341596, 0.02011550341596, 0.02011550341596, 0.02011550341596, 0.02011550341596, 0.02011550341596, 0.02011550341596, 0.00258977325334, 0.00258977325334, 0.00258977325334, 0.00258977325334, 0.00258977325334, 0.00258977325334, 0.00258977325334, 0.00258977325334, 0.00258977325334, 0.00258977325334, 0.00258977325334, 0.00258977325330};
    template<> const std::array<double,Quadrature<3,11>::size> Quadrature<3,11>::weights = {0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, 0.27217694439047, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, -0.69914085914082, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, 0.61162373987892, -0.21015231681897, -0.21015231681897, -0.21015231681897, -0.21015231681897, -0.21015231681897, -0.21015231681897, -0.21015231681897, -0.21015231681897, -0.21015231681897, -0.21015231681897, 0.02440848214286, 0.02440848214286, 0.02440848214286, 0.02440848214286, -0.00056437389765};

    namespace Cartesian
    {
        /** @brief Gauss-Lobatto nodes on [0,1] */
        template<int P>
        struct GaussLobatto
        {
            /** Nodal points */
            static const double nodes[P];
        };

        /** @brief The master element: [0,1]^N */
        template<int N, int P>
        struct Master
        {
            /** Order of polynomial */
            static const int p = P-1;
            /** Number of nodes per element */
            static const int npl = ipow(P,N);
            /** The index-th node in a given cell */
            static Tuple<double,N> dgnodes(const Tuple<int,N>& index, const Cell<N>& cell = Cell<N>())
            {
                assert((0 <= index).all() && (index < P).all());
                Tuple<double,N> node;
                for (int i=0; i<N; ++i) {
                    node[i] = GaussLobatto<P>::nodes[index[i]];
                }
                return cell.lower + cell.width() * node;
            }
        };

        /** @brief The Lagrange polynomials */
        template<int P>
        struct LagrangePoly
        {
            /** The Lagrange polynomial denominators */
            static const double denom[P];
            /** Evaluate the P 1-D Lagrange polynomials at the point x */
            static Vec<P> eval(double x)
            {
                Vec<P> lagrange;
                // First compute the products to the left of the split
                double alpha = 1;
                lagrange[0] = denom[0];
                for (int i=0; i<P-1; ++i) {
                    alpha *= x - GaussLobatto<P>::nodes[i];
                    lagrange[i+1] = alpha * denom[i+1];
                }
                // Now compute the products to the right of the split
                double beta = 1;
                for (int i=P-1; i>0; --i) {
                    beta *= x - GaussLobatto<P>::nodes[i];
                    lagrange[i-1] *= beta;
                }
                return lagrange;
            }
            /** Evaluate the P^N N-D Lagrange polynomials at the coordinate x */
            template<int N>
            static KronVec<N,P> eval(const Tuple<double,N>& x)
            {
                // Precompute the P 1-D Lagrange polynomials evaluated at each
                // component of x
                Mat<P,N> g;
                for (int j=0; j<N; ++j) {
                    g.col(j) = eval(x[j]);
                }
                // Tensor product the components together
                KronVec<N,P> lagrange;
                for (RangeIterator<N,P> it; it != Range<N,P>::end(); ++it) {
                    int i = it.linearIndex();
                    lagrange(i) = 1;
                    for (int j=0; j<N; ++j) {
                        lagrange(i) *= g(it(j),j);
                    }
                }
                return lagrange;
            }
        };

        /*** 1D ***/

        // Gauss-Lobatto nodes
        template<> const double GaussLobatto<1>::nodes[] = {0.50000000000000000000000000000000};
        template<> const double GaussLobatto<2>::nodes[] = {0, 1.0000000000000000000000000000000};
        template<> const double GaussLobatto<3>::nodes[] = {0, 0.50000000000000000000000000000000, 1.0000000000000000000000000000000};
        template<> const double GaussLobatto<4>::nodes[] = {0, 0.27639320225002103035908263312687, 0.72360679774997896964091736687313, 1.0000000000000000000000000000000};
        template<> const double GaussLobatto<5>::nodes[] = {0, 0.17267316464601142810085377187657, 0.50000000000000000000000000000000, 0.82732683535398857189914622812343, 1.0000000000000000000000000000000};
        template<> const double GaussLobatto<6>::nodes[] = {0, 0.11747233803526765357449851302033, 0.35738424175967745184292450297956, 0.64261575824032254815707549702044, 0.88252766196473234642550148697967, 1.0000000000000000000000000000000};

        template<> const double LagrangePoly<1>::denom[] = {1.0000000000000000000000000000000};
        template<> const double LagrangePoly<2>::denom[] = {-1.0000000000000000000000000000000, 1.0000000000000000000000000000000};
        template<> const double LagrangePoly<3>::denom[] = {2.0000000000000000000000000000000, -4.0000000000000000000000000000000, 2.0000000000000000000000000000000};
        template<> const double LagrangePoly<4>::denom[] = {-5.0000000000000000000000000000000, 11.180339887498948482045868343656, -11.180339887498948482045868343656, 5.0000000000000000000000000000000};
        template<> const double LagrangePoly<5>::denom[] = {14.000000000000000000000000000000, -32.666666666666666666666666666667, 37.333333333333333333333333333333, -32.666666666666666666666666666667, 14.000000000000000000000000000000};
        template<> const double LagrangePoly<6>::denom[] = {-42.000000000000000000000000000000, 100.07221064631794721756762446717, -121.16745708464368404487465576678, 121.16745708464368404487465576678, -100.07221064631794721756762446717, 42.000000000000000000000000000000};
    }
}

#endif

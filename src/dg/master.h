#ifndef MASTER_H
#define MASTER_H

#include <array>
#include <vector>
#include "common.h"
#include "range.h"

namespace DG
{
    /** @brief The master element: [-1,1]^N */
    template<int P, int N>
    struct Master
    {
        /** Order of polynomial */
        static const int p = P-1;
        /** Number of nodes per element */
        static const int npl = ipow(P,N);
        /** Mass matrix */
        static const KronMat<P,N> mass;
    };

    /** @brief Gauss-Lobatto nodes on [-1,1] */
    template<int P>
    struct GaussLobatto
    {
        /** Nodal points */
        static const double nodes[P];
        /** Differentiation matrix */
        static const Mat<P,P> diff;
    };

    /** @brief Quadrature nodes and weights on [-1,1] */
    template<int P>
    struct Quadrature
    {
        /** Number of quadrature points in each dimension */
        static const int Q = 2*P;
        /** Quadrature points */
        static const double nodes[Q];
        /** Quadrature weights */
        static const double weights[Q];
    };

    /** @brief Compute a tensor product of matrices */
    template<int P, int N>
    KronMat<P,N> tensorProduct(const KronMat<P,1>& A)
    {
        KronMat<P,N> M;
        for (RangeIterator<P,N> it; it != Range<P,N>::end(); ++it) {
            for (RangeIterator<P,N> jt; jt != Range<P,N>::end(); ++jt) {
                int i = it.linearIndex();
                int j = jt.linearIndex();
                M(i,j) = 1.0;
                for (int k=0; k<N; ++k) {
                    M(i,j) *= A(it(k),jt(k));
                }
            }
        }
        return M;
    }

    /*** 1D ***/

    // Mass matrix
    template<> const KronMat<1,1> Master<1,1>::mass = (KronMat<1,1>() << 2.0000000000000000000000000000000).finished();
    template<> const KronMat<2,1> Master<2,1>::mass = (KronMat<2,1>() << 0.66666666666666666666666666666667, 0.33333333333333333333333333333333, 0.33333333333333333333333333333333, 0.66666666666666666666666666666667).finished();
    template<> const KronMat<3,1> Master<3,1>::mass = (KronMat<3,1>() << 0.26666666666666666666666666666667, 0.13333333333333333333333333333333, -0.066666666666666666666666666666667, 0.13333333333333333333333333333333, 1.0666666666666666666666666666667, 0.13333333333333333333333333333333, -0.066666666666666666666666666666667, 0.13333333333333333333333333333333, 0.26666666666666666666666666666667).finished();
    template<> const KronMat<4,1> Master<4,1>::mass = (KronMat<4,1>() << 0.14285714285714285714285714285714, 0.053239713749994992771646992112649, -0.053239713749994992771646992112649, 0.023809523809523809523809523809524, 0.053239713749994992771646992112649, 0.71428571428571428571428571428571, 0.11904761904761904761904761904762, -0.053239713749994992771646992112649, -0.053239713749994992771646992112649, 0.11904761904761904761904761904762, 0.71428571428571428571428571428571, 0.053239713749994992771646992112649, 0.023809523809523809523809523809524, -0.053239713749994992771646992112649, 0.053239713749994992771646992112649, 0.14285714285714285714285714285714).finished();
    template<> const KronMat<5,1> Master<5,1>::mass = (KronMat<5,1>() << 0.088888888888888888888888888888889, 0.025925925925925925925925925925926, -0.029629629629629629629629629629630, 0.025925925925925925925925925925926, -0.011111111111111111111111111111111, 0.025925925925925925925925925925926, 0.48395061728395061728395061728395, 0.069135802469135802469135802469136, -0.060493827160493827160493827160494, 0.025925925925925925925925925925926, -0.029629629629629629629629629629630, 0.069135802469135802469135802469136, 0.63209876543209876543209876543210, 0.069135802469135802469135802469136, -0.029629629629629629629629629629630, 0.025925925925925925925925925925926, -0.060493827160493827160493827160494, 0.069135802469135802469135802469136, 0.48395061728395061728395061728395, 0.025925925925925925925925925925926, -0.011111111111111111111111111111111, 0.025925925925925925925925925925926, -0.029629629629629629629629629629630, 0.025925925925925925925925925925926, 0.088888888888888888888888888888889).finished();
    template<> const KronMat<6,1> Master<6,1>::mass = (KronMat<6,1>() << 0.060606060606060606060606060606061, 0.014440434436698116481611489822103, -0.017484481541795625403300816127962, 0.017484481541795625403300816127962, -0.014440434436698116481611489822103, 0.0060606060606060606060606060606061, 0.014440434436698116481611489822103, 0.34406814208895180028782982564730, 0.041659779045053090968982247215709, -0.041659779045053090968982247215709, 0.034406814208895180028782982564730, -0.014440434436698116481611489822103, -0.017484481541795625403300816127962, 0.041659779045053090968982247215709, 0.50441670639589668456065502283755, 0.050441670639589668456065502283755, -0.041659779045053090968982247215709, 0.017484481541795625403300816127962, 0.017484481541795625403300816127962, -0.041659779045053090968982247215709, 0.050441670639589668456065502283755, 0.50441670639589668456065502283755, 0.041659779045053090968982247215709, -0.017484481541795625403300816127962, -0.014440434436698116481611489822103, 0.034406814208895180028782982564730, -0.041659779045053090968982247215709, 0.041659779045053090968982247215709, 0.34406814208895180028782982564730, 0.014440434436698116481611489822103, 0.0060606060606060606060606060606061, -0.014440434436698116481611489822103, 0.017484481541795625403300816127962, -0.017484481541795625403300816127962, 0.014440434436698116481611489822103, 0.060606060606060606060606060606061).finished();

    // Differentiation matrix
    template<> const Mat<1,1> GaussLobatto<1>::diff = (KronMat<1,1>() << 0).finished();
    template<> const Mat<2,2> GaussLobatto<2>::diff = (KronMat<2,1>() << -0.50000000000000000000000000000000, 0.50000000000000000000000000000000, -0.50000000000000000000000000000000, 0.50000000000000000000000000000000).finished();
    template<> const Mat<3,3> GaussLobatto<3>::diff = (KronMat<3,1>() << -1.5000000000000000000000000000000, 2.0000000000000000000000000000000, -0.50000000000000000000000000000000, -0.50000000000000000000000000000000, 0, 0.50000000000000000000000000000000, 0.50000000000000000000000000000000, -2.0000000000000000000000000000000, 1.5000000000000000000000000000000).finished();
    template<> const Mat<4,4> GaussLobatto<4>::diff = (KronMat<4,1>() << -3.0000000000000000000000000000000, 4.0450849718747371205114670859141, -1.5450849718747371205114670859141, 0.50000000000000000000000000000000, -0.80901699437494742410229341718282, 0, 1.1180339887498948482045868343656, -0.30901699437494742410229341718282, 0.30901699437494742410229341718282, -1.1180339887498948482045868343656, 0, 0.80901699437494742410229341718282, -0.50000000000000000000000000000000, 1.5450849718747371205114670859141, -4.0450849718747371205114670859141, 3.0000000000000000000000000000000).finished();
    template<> const Mat<5,5> GaussLobatto<5>::diff = (KronMat<5,1>() << -5.0000000000000000000000000000000, 6.7565024887242400038430275296747, -2.6666666666666666666666666666667, 1.4101641779424266628236391369920, -0.50000000000000000000000000000000, -1.2409902530309828578487193421851, 0, 1.7457431218879390501287798833250, -0.76376261582597333443134119895467, 0.25900974696901714215128065781486, 0.37500000000000000000000000000000, -1.3365845776954533352548470981707, 0, 1.3365845776954533352548470981707, -0.37500000000000000000000000000000, -0.25900974696901714215128065781486, 0.76376261582597333443134119895467, -1.7457431218879390501287798833250, 0, 1.2409902530309828578487193421851, 0.50000000000000000000000000000000, -1.4101641779424266628236391369920, 2.6666666666666666666666666666667, -6.7565024887242400038430275296747, 5.0000000000000000000000000000000).finished();
    template<> const Mat<6,6> GaussLobatto<6>::diff = (KronMat<6,1>() << -7.5000000000000000000000000000000, 10.141415936319669280234529270517, -4.0361872703053480052745286479697, 2.2446846481761668242712971406912, -1.3499133141904880992312977632383, 0.50000000000000000000000000000000, -1.7863649483390948939724838715075, 0, 2.5234267774294554319088376503766, -1.1528281585359293413318230756306, 0.65354750742980016720074791705584, -0.23778117798423136380527862029438, 0.48495104785356916930595719533400, -1.7212569528302333832160646940475, 0, 1.7529619663678659788775709519313, -0.78635667222324073743954873067208, 0.26970061083203897247208527745425, -0.26970061083203897247208527745425, 0.78635667222324073743954873067208, -1.7529619663678659788775709519313, 0, 1.7212569528302333832160646940475, -0.48495104785356916930595719533400, 0.23778117798423136380527862029438, -0.65354750742980016720074791705584, 1.1528281585359293413318230756306, -2.5234267774294554319088376503766, 0, 1.7863649483390948939724838715075, -0.50000000000000000000000000000000, 1.3499133141904880992312977632383, -2.2446846481761668242712971406912, 4.0361872703053480052745286479697, -10.141415936319669280234529270517, 7.5000000000000000000000000000000).finished();

    // Gauss-Lobatto nodes
    template<> const double GaussLobatto<1>::nodes[] = {0};
    template<> const double GaussLobatto<2>::nodes[] = {-1.0000000000000000000000000000000, 1.0000000000000000000000000000000};
    template<> const double GaussLobatto<3>::nodes[] = {-1.0000000000000000000000000000000, 0, 1.0000000000000000000000000000000};
    template<> const double GaussLobatto<4>::nodes[] = {-1.0000000000000000000000000000000, -0.44721359549995793928183473374626, 0.44721359549995793928183473374626, 1.0000000000000000000000000000000};
    template<> const double GaussLobatto<5>::nodes[] = {-1.0000000000000000000000000000000, -0.65465367070797714379829245624686, 0, 0.65465367070797714379829245624686, 1.0000000000000000000000000000000};
    template<> const double GaussLobatto<6>::nodes[] = {-1.0000000000000000000000000000000, -0.76505532392946469285100297395934, -0.28523151648064509631415099404088, 0.28523151648064509631415099404088, 0.76505532392946469285100297395934, 1.0000000000000000000000000000000};

    // Quadrature nodes
    template<> const double Quadrature<1>::nodes[] = {-0.57735026918962576450914878050196, 0.57735026918962576450914878050196};
    template<> const double Quadrature<2>::nodes[] = {-0.86113631159405257522394648889281, -0.33998104358485626480266575910324, 0.33998104358485626480266575910324, 0.86113631159405257522394648889281};
    template<> const double Quadrature<3>::nodes[] = {-0.93246951420315202781230155449399, -0.66120938646626451366139959501991, -0.23861918608319690863050172168071, 0.23861918608319690863050172168071, 0.66120938646626451366139959501991, 0.93246951420315202781230155449399};
    template<> const double Quadrature<4>::nodes[] = {-0.96028985649753623168356086856947, -0.79666647741362673959155393647583, -0.52553240991632898581773904918925, -0.18343464249564980493947614236018, 0.18343464249564980493947614236018, 0.52553240991632898581773904918925, 0.79666647741362673959155393647583, 0.96028985649753623168356086856947};
    template<> const double Quadrature<5>::nodes[] = {-0.97390652851717172007796401208445, -0.86506336668898451073209668842349, -0.67940956829902440623432736511487, -0.43339539412924719079926594316578, -0.14887433898163121088482600112972, 0.14887433898163121088482600112972, 0.43339539412924719079926594316578, 0.67940956829902440623432736511487, 0.86506336668898451073209668842349, 0.97390652851717172007796401208445};
    template<> const double Quadrature<6>::nodes[] = {-0.98156063424671925069054909014928, -0.90411725637047485667846586611910, -0.76990267419430468703689383321282, -0.58731795428661744729670241894053, -0.36783149899818019375269153664372, -0.12523340851146891547244136946385, 0.12523340851146891547244136946385, 0.36783149899818019375269153664372, 0.58731795428661744729670241894053, 0.76990267419430468703689383321282, 0.90411725637047485667846586611910, 0.98156063424671925069054909014928};
    
    // Quadrature weights
    template<> const double Quadrature<1>::weights[] = {1.0000000000000000000000000000000, 1.0000000000000000000000000000000};
    template<> const double Quadrature<2>::weights[] = {0.34785484513745385737306394922200, 0.65214515486254614262693605077800, 0.65214515486254614262693605077800, 0.34785484513745385737306394922200};
    template<> const double Quadrature<3>::weights[] = {0.17132449237917034504029614217273, 0.36076157304813860756983351383772, 0.46791393457269104738987034398955, 0.46791393457269104738987034398955, 0.36076157304813860756983351383772, 0.17132449237917034504029614217273};
    template<> const double Quadrature<4>::weights[] = {0.10122853629037625915253135430996, 0.22238103445337447054435599442624, 0.31370664587788728733796220198660, 0.36268378337836198296515044927720, 0.36268378337836198296515044927720, 0.31370664587788728733796220198660, 0.22238103445337447054435599442624, 0.10122853629037625915253135430996};
    template<> const double Quadrature<5>::weights[] = {0.066671344308688137593568809893332, 0.14945134915058059314577633965770, 0.21908636251598204399553493422816, 0.26926671930999635509122692156947, 0.29552422471475287017389299465134, 0.29552422471475287017389299465134, 0.26926671930999635509122692156947, 0.21908636251598204399553493422816, 0.14945134915058059314577633965770, 0.066671344308688137593568809893332};
    template<> const double Quadrature<6>::weights[] = {0.047175336386511827194615961485017, 0.10693932599531843096025471819400, 0.16007832854334622633465252954336, 0.20316742672306592174906445580980, 0.23349253653835480876084989892488, 0.24914704581340278500056243604295, 0.24914704581340278500056243604295, 0.23349253653835480876084989892488, 0.20316742672306592174906445580980, 0.16007832854334622633465252954336, 0.10693932599531843096025471819400, 0.047175336386511827194615961485017};

    /*** 2D ***/

    // Mass matrix
    template<> const KronMat<1,2> Master<1,2>::mass = tensorProduct<1,2>(Master<1,1>::mass);
    template<> const KronMat<2,2> Master<2,2>::mass = tensorProduct<2,2>(Master<2,1>::mass);
    template<> const KronMat<3,2> Master<3,2>::mass = tensorProduct<3,2>(Master<3,1>::mass);
    template<> const KronMat<4,2> Master<4,2>::mass = tensorProduct<4,2>(Master<4,1>::mass);
    template<> const KronMat<5,2> Master<5,2>::mass = tensorProduct<5,2>(Master<5,1>::mass);
    template<> const KronMat<6,2> Master<6,2>::mass = tensorProduct<6,2>(Master<6,1>::mass);

    /*** 3D ***/

    // Mass matrix
    template<> const KronMat<1,3> Master<1,3>::mass = tensorProduct<1,3>(Master<1,1>::mass);
    template<> const KronMat<2,3> Master<2,3>::mass = tensorProduct<2,3>(Master<2,1>::mass);
    template<> const KronMat<3,3> Master<3,3>::mass = tensorProduct<3,3>(Master<3,1>::mass);
    template<> const KronMat<4,3> Master<4,3>::mass = tensorProduct<4,3>(Master<4,1>::mass);
    template<> const KronMat<5,3> Master<5,3>::mass = tensorProduct<5,3>(Master<5,1>::mass);
    template<> const KronMat<6,3> Master<6,3>::mass = tensorProduct<6,3>(Master<6,1>::mass);
}

#endif

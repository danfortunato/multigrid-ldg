#ifndef MASTER
#define MASTER

namespace DG
{
    typedef Eigen::Matrix<double,P,    P,    Eigen::RowMajor> Mat1D;
    typedef Eigen::Matrix<double,P*P,  P*P,  Eigen::RowMajor> Mat2D;
    typedef Eigen::Matrix<double,P*P*P,P*P*P,Eigen::RowMajor> Mat3D;

    template<int P>
    class Master
    {   
        static Mat1D mass1D;
        static Mat2D mass2D;
    };

    // WARNING: Auto-generated code below!
    // Anything added between the following comments is subject to deletion.

    /* Begin code generation */
    
    template<>
    class Master<1>
    {
        static Mat1D mass1D = (Mat1D() <<
            2.000000000000000
        ).finished();

        static Mat2D mass2D = (Mat2D() <<
            4.000000000000000
        ).finished();
    }

    template<>
    class Master<2>
    {
        static Mat1D mass1D = (Mat1D() <<
            0.6666666666666667,
            0.3333333333333333,
            0.3333333333333333,
            0.6666666666666667
        ).finished();

        static Mat2D mass2D = (Mat2D() <<
            0.4444444444444444,
            0.2222222222222222,
            0.2222222222222222,
            0.1111111111111111,
            0.2222222222222222,
            0.4444444444444444,
            0.1111111111111111,
            0.2222222222222222,
            0.2222222222222222,
            0.1111111111111111,
            0.4444444444444444,
            0.2222222222222222,
            0.1111111111111111,
            0.2222222222222222,
            0.2222222222222222,
            0.4444444444444444
        ).finished();
    }

    template<>
    class Master<3>
    {
        static Mat1D mass1D = (Mat1D() <<
            0.2666666666666667,
            0.1333333333333333,
            -0.06666666666666667,
            0.1333333333333333,
            1.066666666666667,
            0.1333333333333333,
            -0.06666666666666667,
            0.1333333333333333,
            0.2666666666666667
        ).finished();

        static Mat2D mass2D = (Mat2D() <<
            0.07111111111111111,
            0.03555555555555556,
            -0.01777777777777778,
            0.03555555555555556,
            0.01777777777777778,
            -0.008888888888888889,
            -0.01777777777777778,
            -0.008888888888888889,
            0.004444444444444444,
            0.03555555555555556,
            0.2844444444444444,
            0.03555555555555556,
            0.01777777777777778,
            0.1422222222222222,
            0.01777777777777778,
            -0.008888888888888889,
            -0.07111111111111111,
            -0.008888888888888889,
            -0.01777777777777778,
            0.03555555555555556,
            0.07111111111111111,
            -0.008888888888888889,
            0.01777777777777778,
            0.03555555555555556,
            0.004444444444444444,
            -0.008888888888888889,
            -0.01777777777777778,
            0.03555555555555556,
            0.01777777777777778,
            -0.008888888888888889,
            0.2844444444444444,
            0.1422222222222222,
            -0.07111111111111111,
            0.03555555555555556,
            0.01777777777777778,
            -0.008888888888888889,
            0.01777777777777778,
            0.1422222222222222,
            0.01777777777777778,
            0.1422222222222222,
            1.137777777777778,
            0.1422222222222222,
            0.01777777777777778,
            0.1422222222222222,
            0.01777777777777778,
            -0.008888888888888889,
            0.01777777777777778,
            0.03555555555555556,
            -0.07111111111111111,
            0.1422222222222222,
            0.2844444444444444,
            -0.008888888888888889,
            0.01777777777777778,
            0.03555555555555556,
            -0.01777777777777778,
            -0.008888888888888889,
            0.004444444444444444,
            0.03555555555555556,
            0.01777777777777778,
            -0.008888888888888889,
            0.07111111111111111,
            0.03555555555555556,
            -0.01777777777777778,
            -0.008888888888888889,
            -0.07111111111111111,
            -0.008888888888888889,
            0.01777777777777778,
            0.1422222222222222,
            0.01777777777777778,
            0.03555555555555556,
            0.2844444444444444,
            0.03555555555555556,
            0.004444444444444444,
            -0.008888888888888889,
            -0.01777777777777778,
            -0.008888888888888889,
            0.01777777777777778,
            0.03555555555555556,
            -0.01777777777777778,
            0.03555555555555556,
            0.07111111111111111
        ).finished();
    }

    template<>
    class Master<4>
    {
        static Mat1D mass1D = (Mat1D() <<
            0.1428571428571429,
            0.05323971374999499,
            -0.05323971374999499,
            0.02380952380952381,
            0.05323971374999499,
            0.7142857142857143,
            0.1190476190476190,
            -0.05323971374999499,
            -0.05323971374999499,
            0.1190476190476190,
            0.7142857142857143,
            0.05323971374999499,
            0.02380952380952381,
            -0.05323971374999499,
            0.05323971374999499,
            0.1428571428571429
        ).finished();

        static Mat2D mass2D = (Mat2D() <<
            0.02040816326530612,
            0.007605673392856428,
            -0.007605673392856428,
            0.003401360544217687,
            0.007605673392856428,
            0.002834467120181406,
            -0.002834467120181406,
            0.001267612232142738,
            -0.007605673392856428,
            -0.002834467120181406,
            0.002834467120181406,
            -0.001267612232142738,
            0.003401360544217687,
            0.001267612232142738,
            -0.001267612232142738,
            0.0005668934240362812,
            0.007605673392856428,
            0.1020408163265306,
            0.01700680272108844,
            -0.007605673392856428,
            0.002834467120181406,
            0.03802836696428214,
            0.006338061160713690,
            -0.002834467120181406,
            -0.002834467120181406,
            -0.03802836696428214,
            -0.006338061160713690,
            0.002834467120181406,
            0.001267612232142738,
            0.01700680272108844,
            0.002834467120181406,
            -0.001267612232142738,
            -0.007605673392856428,
            0.01700680272108844,
            0.1020408163265306,
            0.007605673392856428,
            -0.002834467120181406,
            0.006338061160713690,
            0.03802836696428214,
            0.002834467120181406,
            0.002834467120181406,
            -0.006338061160713690,
            -0.03802836696428214,
            -0.002834467120181406,
            -0.001267612232142738,
            0.002834467120181406,
            0.01700680272108844,
            0.001267612232142738,
            0.003401360544217687,
            -0.007605673392856428,
            0.007605673392856428,
            0.02040816326530612,
            0.001267612232142738,
            -0.002834467120181406,
            0.002834467120181406,
            0.007605673392856428,
            -0.001267612232142738,
            0.002834467120181406,
            -0.002834467120181406,
            -0.007605673392856428,
            0.0005668934240362812,
            -0.001267612232142738,
            0.001267612232142738,
            0.003401360544217687,
            0.007605673392856428,
            0.002834467120181406,
            -0.002834467120181406,
            0.001267612232142738,
            0.1020408163265306,
            0.03802836696428214,
            -0.03802836696428214,
            0.01700680272108844,
            0.01700680272108844,
            0.006338061160713690,
            -0.006338061160713690,
            0.002834467120181406,
            -0.007605673392856428,
            -0.002834467120181406,
            0.002834467120181406,
            -0.001267612232142738,
            0.002834467120181406,
            0.03802836696428214,
            0.006338061160713690,
            -0.002834467120181406,
            0.03802836696428214,
            0.5102040816326531,
            0.08503401360544218,
            -0.03802836696428214,
            0.006338061160713690,
            0.08503401360544218,
            0.01417233560090703,
            -0.006338061160713690,
            -0.002834467120181406,
            -0.03802836696428214,
            -0.006338061160713690,
            0.002834467120181406,
            -0.002834467120181406,
            0.006338061160713690,
            0.03802836696428214,
            0.002834467120181406,
            -0.03802836696428214,
            0.08503401360544218,
            0.5102040816326531,
            0.03802836696428214,
            -0.006338061160713690,
            0.01417233560090703,
            0.08503401360544218,
            0.006338061160713690,
            0.002834467120181406,
            -0.006338061160713690,
            -0.03802836696428214,
            -0.002834467120181406,
            0.001267612232142738,
            -0.002834467120181406,
            0.002834467120181406,
            0.007605673392856428,
            0.01700680272108844,
            -0.03802836696428214,
            0.03802836696428214,
            0.1020408163265306,
            0.002834467120181406,
            -0.006338061160713690,
            0.006338061160713690,
            0.01700680272108844,
            -0.001267612232142738,
            0.002834467120181406,
            -0.002834467120181406,
            -0.007605673392856428,
            -0.007605673392856428,
            -0.002834467120181406,
            0.002834467120181406,
            -0.001267612232142738,
            0.01700680272108844,
            0.006338061160713690,
            -0.006338061160713690,
            0.002834467120181406,
            0.1020408163265306,
            0.03802836696428214,
            -0.03802836696428214,
            0.01700680272108844,
            0.007605673392856428,
            0.002834467120181406,
            -0.002834467120181406,
            0.001267612232142738,
            -0.002834467120181406,
            -0.03802836696428214,
            -0.006338061160713690,
            0.002834467120181406,
            0.006338061160713690,
            0.08503401360544218,
            0.01417233560090703,
            -0.006338061160713690,
            0.03802836696428214,
            0.5102040816326531,
            0.08503401360544218,
            -0.03802836696428214,
            0.002834467120181406,
            0.03802836696428214,
            0.006338061160713690,
            -0.002834467120181406,
            0.002834467120181406,
            -0.006338061160713690,
            -0.03802836696428214,
            -0.002834467120181406,
            -0.006338061160713690,
            0.01417233560090703,
            0.08503401360544218,
            0.006338061160713690,
            -0.03802836696428214,
            0.08503401360544218,
            0.5102040816326531,
            0.03802836696428214,
            -0.002834467120181406,
            0.006338061160713690,
            0.03802836696428214,
            0.002834467120181406,
            -0.001267612232142738,
            0.002834467120181406,
            -0.002834467120181406,
            -0.007605673392856428,
            0.002834467120181406,
            -0.006338061160713690,
            0.006338061160713690,
            0.01700680272108844,
            0.01700680272108844,
            -0.03802836696428214,
            0.03802836696428214,
            0.1020408163265306,
            0.001267612232142738,
            -0.002834467120181406,
            0.002834467120181406,
            0.007605673392856428,
            0.003401360544217687,
            0.001267612232142738,
            -0.001267612232142738,
            0.0005668934240362812,
            -0.007605673392856428,
            -0.002834467120181406,
            0.002834467120181406,
            -0.001267612232142738,
            0.007605673392856428,
            0.002834467120181406,
            -0.002834467120181406,
            0.001267612232142738,
            0.02040816326530612,
            0.007605673392856428,
            -0.007605673392856428,
            0.003401360544217687,
            0.001267612232142738,
            0.01700680272108844,
            0.002834467120181406,
            -0.001267612232142738,
            -0.002834467120181406,
            -0.03802836696428214,
            -0.006338061160713690,
            0.002834467120181406,
            0.002834467120181406,
            0.03802836696428214,
            0.006338061160713690,
            -0.002834467120181406,
            0.007605673392856428,
            0.1020408163265306,
            0.01700680272108844,
            -0.007605673392856428,
            -0.001267612232142738,
            0.002834467120181406,
            0.01700680272108844,
            0.001267612232142738,
            0.002834467120181406,
            -0.006338061160713690,
            -0.03802836696428214,
            -0.002834467120181406,
            -0.002834467120181406,
            0.006338061160713690,
            0.03802836696428214,
            0.002834467120181406,
            -0.007605673392856428,
            0.01700680272108844,
            0.1020408163265306,
            0.007605673392856428,
            0.0005668934240362812,
            -0.001267612232142738,
            0.001267612232142738,
            0.003401360544217687,
            -0.001267612232142738,
            0.002834467120181406,
            -0.002834467120181406,
            -0.007605673392856428,
            0.001267612232142738,
            -0.002834467120181406,
            0.002834467120181406,
            0.007605673392856428,
            0.003401360544217687,
            -0.007605673392856428,
            0.007605673392856428,
            0.02040816326530612
        ).finished();
    }

    template<>
    class Master<5>
    {
        static Mat1D mass1D = (Mat1D() <<
            0.08888888888888889,
            0.02592592592592593,
            -0.02962962962962963,
            0.02592592592592593,
            -0.01111111111111111,
            0.02592592592592593,
            0.4839506172839506,
            0.06913580246913580,
            -0.06049382716049383,
            0.02592592592592593,
            -0.02962962962962963,
            0.06913580246913580,
            0.6320987654320988,
            0.06913580246913580,
            -0.02962962962962963,
            0.02592592592592593,
            -0.06049382716049383,
            0.06913580246913580,
            0.4839506172839506,
            0.02592592592592593,
            -0.01111111111111111,
            0.02592592592592593,
            -0.02962962962962963,
            0.02592592592592593,
            0.08888888888888889
        ).finished();

        static Mat2D mass2D = (Mat2D() <<
            0.007901234567901235,
            0.002304526748971193,
            -0.002633744855967078,
            0.002304526748971193,
            -0.0009876543209876543,
            0.002304526748971193,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.0002880658436213992,
            -0.002633744855967078,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.0007681755829903978,
            0.0003292181069958848,
            0.002304526748971193,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.0002880658436213992,
            -0.0009876543209876543,
            -0.0002880658436213992,
            0.0003292181069958848,
            -0.0002880658436213992,
            0.0001234567901234568,
            0.002304526748971193,
            0.04301783264746228,
            0.006145404663923182,
            -0.005377229080932785,
            0.002304526748971193,
            0.0006721536351165981,
            0.01254686785550983,
            0.001792409693644262,
            -0.001568358481938729,
            0.0006721536351165981,
            -0.0007681755829903978,
            -0.01433927754915409,
            -0.002048468221307727,
            0.001792409693644262,
            -0.0007681755829903978,
            0.0006721536351165981,
            0.01254686785550983,
            0.001792409693644262,
            -0.001568358481938729,
            0.0006721536351165981,
            -0.0002880658436213992,
            -0.005377229080932785,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.0002880658436213992,
            -0.002633744855967078,
            0.006145404663923182,
            0.05618655692729767,
            0.006145404663923182,
            -0.002633744855967078,
            -0.0007681755829903978,
            0.001792409693644262,
            0.01638774577046182,
            0.001792409693644262,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.002048468221307727,
            -0.01872885230909922,
            -0.002048468221307727,
            0.0008779149519890261,
            -0.0007681755829903978,
            0.001792409693644262,
            0.01638774577046182,
            0.001792409693644262,
            -0.0007681755829903978,
            0.0003292181069958848,
            -0.0007681755829903978,
            -0.007023319615912209,
            -0.0007681755829903978,
            0.0003292181069958848,
            0.002304526748971193,
            -0.005377229080932785,
            0.006145404663923182,
            0.04301783264746228,
            0.002304526748971193,
            0.0006721536351165981,
            -0.001568358481938729,
            0.001792409693644262,
            0.01254686785550983,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.001792409693644262,
            -0.002048468221307727,
            -0.01433927754915409,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.001568358481938729,
            0.001792409693644262,
            0.01254686785550983,
            0.0006721536351165981,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.0007681755829903978,
            -0.005377229080932785,
            -0.0002880658436213992,
            -0.0009876543209876543,
            0.002304526748971193,
            -0.002633744855967078,
            0.002304526748971193,
            0.007901234567901235,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            0.002304526748971193,
            0.0003292181069958848,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.0007681755829903978,
            -0.002633744855967078,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            0.002304526748971193,
            0.0001234567901234568,
            -0.0002880658436213992,
            0.0003292181069958848,
            -0.0002880658436213992,
            -0.0009876543209876543,
            0.002304526748971193,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.0002880658436213992,
            0.04301783264746228,
            0.01254686785550983,
            -0.01433927754915409,
            0.01254686785550983,
            -0.005377229080932785,
            0.006145404663923182,
            0.001792409693644262,
            -0.002048468221307727,
            0.001792409693644262,
            -0.0007681755829903978,
            -0.005377229080932785,
            -0.001568358481938729,
            0.001792409693644262,
            -0.001568358481938729,
            0.0006721536351165981,
            0.002304526748971193,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.0002880658436213992,
            0.0006721536351165981,
            0.01254686785550983,
            0.001792409693644262,
            -0.001568358481938729,
            0.0006721536351165981,
            0.01254686785550983,
            0.2342081999695168,
            0.03345831428135955,
            -0.02927602499618961,
            0.01254686785550983,
            0.001792409693644262,
            0.03345831428135955,
            0.004779759183051364,
            -0.004182289285169944,
            0.001792409693644262,
            -0.001568358481938729,
            -0.02927602499618961,
            -0.004182289285169944,
            0.003659503124523701,
            -0.001568358481938729,
            0.0006721536351165981,
            0.01254686785550983,
            0.001792409693644262,
            -0.001568358481938729,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.001792409693644262,
            0.01638774577046182,
            0.001792409693644262,
            -0.0007681755829903978,
            -0.01433927754915409,
            0.03345831428135955,
            0.3059045877152873,
            0.03345831428135955,
            -0.01433927754915409,
            -0.002048468221307727,
            0.004779759183051364,
            0.04370065538789819,
            0.004779759183051364,
            -0.002048468221307727,
            0.001792409693644262,
            -0.004182289285169944,
            -0.03823807346441091,
            -0.004182289285169944,
            0.001792409693644262,
            -0.0007681755829903978,
            0.001792409693644262,
            0.01638774577046182,
            0.001792409693644262,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.001568358481938729,
            0.001792409693644262,
            0.01254686785550983,
            0.0006721536351165981,
            0.01254686785550983,
            -0.02927602499618961,
            0.03345831428135955,
            0.2342081999695168,
            0.01254686785550983,
            0.001792409693644262,
            -0.004182289285169944,
            0.004779759183051364,
            0.03345831428135955,
            0.001792409693644262,
            -0.001568358481938729,
            0.003659503124523701,
            -0.004182289285169944,
            -0.02927602499618961,
            -0.001568358481938729,
            0.0006721536351165981,
            -0.001568358481938729,
            0.001792409693644262,
            0.01254686785550983,
            0.0006721536351165981,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            0.002304526748971193,
            -0.005377229080932785,
            0.01254686785550983,
            -0.01433927754915409,
            0.01254686785550983,
            0.04301783264746228,
            -0.0007681755829903978,
            0.001792409693644262,
            -0.002048468221307727,
            0.001792409693644262,
            0.006145404663923182,
            0.0006721536351165981,
            -0.001568358481938729,
            0.001792409693644262,
            -0.001568358481938729,
            -0.005377229080932785,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            0.002304526748971193,
            -0.002633744855967078,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.0007681755829903978,
            0.0003292181069958848,
            0.006145404663923182,
            0.001792409693644262,
            -0.002048468221307727,
            0.001792409693644262,
            -0.0007681755829903978,
            0.05618655692729767,
            0.01638774577046182,
            -0.01872885230909922,
            0.01638774577046182,
            -0.007023319615912209,
            0.006145404663923182,
            0.001792409693644262,
            -0.002048468221307727,
            0.001792409693644262,
            -0.0007681755829903978,
            -0.002633744855967078,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.0007681755829903978,
            0.0003292181069958848,
            -0.0007681755829903978,
            -0.01433927754915409,
            -0.002048468221307727,
            0.001792409693644262,
            -0.0007681755829903978,
            0.001792409693644262,
            0.03345831428135955,
            0.004779759183051364,
            -0.004182289285169944,
            0.001792409693644262,
            0.01638774577046182,
            0.3059045877152873,
            0.04370065538789819,
            -0.03823807346441091,
            0.01638774577046182,
            0.001792409693644262,
            0.03345831428135955,
            0.004779759183051364,
            -0.004182289285169944,
            0.001792409693644262,
            -0.0007681755829903978,
            -0.01433927754915409,
            -0.002048468221307727,
            0.001792409693644262,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.002048468221307727,
            -0.01872885230909922,
            -0.002048468221307727,
            0.0008779149519890261,
            -0.002048468221307727,
            0.004779759183051364,
            0.04370065538789819,
            0.004779759183051364,
            -0.002048468221307727,
            -0.01872885230909922,
            0.04370065538789819,
            0.3995488492607834,
            0.04370065538789819,
            -0.01872885230909922,
            -0.002048468221307727,
            0.004779759183051364,
            0.04370065538789819,
            0.004779759183051364,
            -0.002048468221307727,
            0.0008779149519890261,
            -0.002048468221307727,
            -0.01872885230909922,
            -0.002048468221307727,
            0.0008779149519890261,
            -0.0007681755829903978,
            0.001792409693644262,
            -0.002048468221307727,
            -0.01433927754915409,
            -0.0007681755829903978,
            0.001792409693644262,
            -0.004182289285169944,
            0.004779759183051364,
            0.03345831428135955,
            0.001792409693644262,
            0.01638774577046182,
            -0.03823807346441091,
            0.04370065538789819,
            0.3059045877152873,
            0.01638774577046182,
            0.001792409693644262,
            -0.004182289285169944,
            0.004779759183051364,
            0.03345831428135955,
            0.001792409693644262,
            -0.0007681755829903978,
            0.001792409693644262,
            -0.002048468221307727,
            -0.01433927754915409,
            -0.0007681755829903978,
            0.0003292181069958848,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.0007681755829903978,
            -0.002633744855967078,
            -0.0007681755829903978,
            0.001792409693644262,
            -0.002048468221307727,
            0.001792409693644262,
            0.006145404663923182,
            -0.007023319615912209,
            0.01638774577046182,
            -0.01872885230909922,
            0.01638774577046182,
            0.05618655692729767,
            -0.0007681755829903978,
            0.001792409693644262,
            -0.002048468221307727,
            0.001792409693644262,
            0.006145404663923182,
            0.0003292181069958848,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.0007681755829903978,
            -0.002633744855967078,
            0.002304526748971193,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.0002880658436213992,
            -0.005377229080932785,
            -0.001568358481938729,
            0.001792409693644262,
            -0.001568358481938729,
            0.0006721536351165981,
            0.006145404663923182,
            0.001792409693644262,
            -0.002048468221307727,
            0.001792409693644262,
            -0.0007681755829903978,
            0.04301783264746228,
            0.01254686785550983,
            -0.01433927754915409,
            0.01254686785550983,
            -0.005377229080932785,
            0.002304526748971193,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.0002880658436213992,
            0.0006721536351165981,
            0.01254686785550983,
            0.001792409693644262,
            -0.001568358481938729,
            0.0006721536351165981,
            -0.001568358481938729,
            -0.02927602499618961,
            -0.004182289285169944,
            0.003659503124523701,
            -0.001568358481938729,
            0.001792409693644262,
            0.03345831428135955,
            0.004779759183051364,
            -0.004182289285169944,
            0.001792409693644262,
            0.01254686785550983,
            0.2342081999695168,
            0.03345831428135955,
            -0.02927602499618961,
            0.01254686785550983,
            0.0006721536351165981,
            0.01254686785550983,
            0.001792409693644262,
            -0.001568358481938729,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.001792409693644262,
            0.01638774577046182,
            0.001792409693644262,
            -0.0007681755829903978,
            0.001792409693644262,
            -0.004182289285169944,
            -0.03823807346441091,
            -0.004182289285169944,
            0.001792409693644262,
            -0.002048468221307727,
            0.004779759183051364,
            0.04370065538789819,
            0.004779759183051364,
            -0.002048468221307727,
            -0.01433927754915409,
            0.03345831428135955,
            0.3059045877152873,
            0.03345831428135955,
            -0.01433927754915409,
            -0.0007681755829903978,
            0.001792409693644262,
            0.01638774577046182,
            0.001792409693644262,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.001568358481938729,
            0.001792409693644262,
            0.01254686785550983,
            0.0006721536351165981,
            -0.001568358481938729,
            0.003659503124523701,
            -0.004182289285169944,
            -0.02927602499618961,
            -0.001568358481938729,
            0.001792409693644262,
            -0.004182289285169944,
            0.004779759183051364,
            0.03345831428135955,
            0.001792409693644262,
            0.01254686785550983,
            -0.02927602499618961,
            0.03345831428135955,
            0.2342081999695168,
            0.01254686785550983,
            0.0006721536351165981,
            -0.001568358481938729,
            0.001792409693644262,
            0.01254686785550983,
            0.0006721536351165981,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            0.002304526748971193,
            0.0006721536351165981,
            -0.001568358481938729,
            0.001792409693644262,
            -0.001568358481938729,
            -0.005377229080932785,
            -0.0007681755829903978,
            0.001792409693644262,
            -0.002048468221307727,
            0.001792409693644262,
            0.006145404663923182,
            -0.005377229080932785,
            0.01254686785550983,
            -0.01433927754915409,
            0.01254686785550983,
            0.04301783264746228,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            0.002304526748971193,
            -0.0009876543209876543,
            -0.0002880658436213992,
            0.0003292181069958848,
            -0.0002880658436213992,
            0.0001234567901234568,
            0.002304526748971193,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.0002880658436213992,
            -0.002633744855967078,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.0007681755829903978,
            0.0003292181069958848,
            0.002304526748971193,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.0002880658436213992,
            0.007901234567901235,
            0.002304526748971193,
            -0.002633744855967078,
            0.002304526748971193,
            -0.0009876543209876543,
            -0.0002880658436213992,
            -0.005377229080932785,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.0002880658436213992,
            0.0006721536351165981,
            0.01254686785550983,
            0.001792409693644262,
            -0.001568358481938729,
            0.0006721536351165981,
            -0.0007681755829903978,
            -0.01433927754915409,
            -0.002048468221307727,
            0.001792409693644262,
            -0.0007681755829903978,
            0.0006721536351165981,
            0.01254686785550983,
            0.001792409693644262,
            -0.001568358481938729,
            0.0006721536351165981,
            0.002304526748971193,
            0.04301783264746228,
            0.006145404663923182,
            -0.005377229080932785,
            0.002304526748971193,
            0.0003292181069958848,
            -0.0007681755829903978,
            -0.007023319615912209,
            -0.0007681755829903978,
            0.0003292181069958848,
            -0.0007681755829903978,
            0.001792409693644262,
            0.01638774577046182,
            0.001792409693644262,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.002048468221307727,
            -0.01872885230909922,
            -0.002048468221307727,
            0.0008779149519890261,
            -0.0007681755829903978,
            0.001792409693644262,
            0.01638774577046182,
            0.001792409693644262,
            -0.0007681755829903978,
            -0.002633744855967078,
            0.006145404663923182,
            0.05618655692729767,
            0.006145404663923182,
            -0.002633744855967078,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.0007681755829903978,
            -0.005377229080932785,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.001568358481938729,
            0.001792409693644262,
            0.01254686785550983,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.001792409693644262,
            -0.002048468221307727,
            -0.01433927754915409,
            -0.0007681755829903978,
            0.0006721536351165981,
            -0.001568358481938729,
            0.001792409693644262,
            0.01254686785550983,
            0.0006721536351165981,
            0.002304526748971193,
            -0.005377229080932785,
            0.006145404663923182,
            0.04301783264746228,
            0.002304526748971193,
            0.0001234567901234568,
            -0.0002880658436213992,
            0.0003292181069958848,
            -0.0002880658436213992,
            -0.0009876543209876543,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            0.002304526748971193,
            0.0003292181069958848,
            -0.0007681755829903978,
            0.0008779149519890261,
            -0.0007681755829903978,
            -0.002633744855967078,
            -0.0002880658436213992,
            0.0006721536351165981,
            -0.0007681755829903978,
            0.0006721536351165981,
            0.002304526748971193,
            -0.0009876543209876543,
            0.002304526748971193,
            -0.002633744855967078,
            0.002304526748971193,
            0.007901234567901235
        ).finished();
    }

    /* End code generation */
}

#endif

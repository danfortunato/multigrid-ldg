#include <functional>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include "common.h"
#include "quadtree.h"
#include "mesh.h"
#include "function.h"
#include "boundaryconditions.h"
#include "ldgpoisson.h"
#include "timer.h"
#include "multigrid.h"

namespace DG
{
    /** @brief The type of solve to run */
    enum SolveType
    {
        kCG,
        kMGPCG,
        kVCycles
    };

    enum TestType
    {
        kSolution,
        kConvergence
    };

    /** @brief The parameters to use in the DG method */
    struct Parameters
    {
        int p = 2;                                        // Polynomial order
        double dx = 0.125;                                // Mesh spacing
        bool adaptive = false;                            // Adaptive mesh refinement
        CoarseningStrategy coarsening = kEqualCoarsening; // Coarsening strategy
        DG::BoundaryType bctype = DG::kDirichlet;         // Type of boundary condition
        double tau0 = 0;                                  // Interior penalty parameter
        double tauD = 1000;                               // Dirichlet penalty parameter
        double cgtol = 1e-8;                              // Tolerance for CG
        int cgmaxit = 200;                                // Max iterations for CG
        DG::SolveType solvetype = DG::kMGPCG;             // Type of solver
        DG::TestType testtype = DG::kConvergence;         // Type of test
        int ncycle = 1;                                   // Number of V-cycles
        DG::MG::Parameters multigrid;                     // Multigrid parameters
        bool dump = false;                                // Flag to dump extra data
    };
}

void print_help()
{
    std::cout << std::endl;
    std::cout << "Options are:" << std::endl << std::endl;
    std::cout << std::setw(20) << std::left << "   --p"             << std::setw(20) << "Polynomial order" << std::endl;
    std::cout << std::setw(20) << std::left << "   --h"             << std::setw(20) << "Mesh spacing" << std::endl;
    std::cout << std::setw(20) << std::left << "   --adaptive"      << std::setw(20) << "Adaptive mesh refinement" << std::endl;
    std::cout << std::setw(20) << std::left << "   --coarsening"    << std::setw(20) << "Coarsening strategy [equal, rapid]" << std::endl;
    std::cout << std::setw(20) << std::left << "   --bc"            << std::setw(20) << "Type of boundary conditions [dirichlet, neumann, periodic]" << std::endl;
    std::cout << std::setw(20) << std::left << "   --tau0"          << std::setw(20) << "Interior penalty parameter" << std::endl;
    std::cout << std::setw(20) << std::left << "   --tauD"          << std::setw(20) << "Dirichlet penalty parameter" << std::endl;
    std::cout << std::setw(20) << std::left << "   --solver"        << std::setw(20) << "Type of solver [cg, mgpcg, vcycles]" << std::endl;
    std::cout << std::setw(20) << std::left << "   --test"          << std::setw(20) << "Type of test [solution, convergence]" << std::endl;
    std::cout << std::setw(20) << std::left << "   --cg.tol"        << std::setw(20) << "Tolerance for CG" << std::endl;
    std::cout << std::setw(20) << std::left << "   --cg.maxit"      << std::setw(20) << "Maximum number of iterations for CG" << std::endl;
    std::cout << std::setw(20) << std::left << "   --mg.relaxation" << std::setw(20) << "Relaxation method [jacobi, gauss-seidel]" << std::endl;
    std::cout << std::setw(20) << std::left << "   --mg.omega"      << std::setw(20) << "Relaxation weight" << std::endl;
    std::cout << std::setw(20) << std::left << "   --mg.solver"     << std::setw(20) << "Coarse solver [cholesky, relaxation]" << std::endl;
    std::cout << std::setw(20) << std::left << "   --mg.npre"       << std::setw(20) << "Number of pre-smooths" << std::endl;
    std::cout << std::setw(20) << std::left << "   --mg.npost"      << std::setw(20) << "Number of post-smooths" << std::endl;
    std::cout << std::setw(20) << std::left << "   --mg.ncycle"     << std::setw(20) << "Number of V-cycles" << std::endl;
    std::cout << std::setw(20) << std::left << "   --mg.restriction"<< std::setw(20) << "Restriction method [RAT, RATRAT]" << std::endl;
    std::cout << std::setw(20) << std::left << "   --dump"          << std::setw(20) << "Data to dump" << std::endl;
    std::cout << std::setw(20) << std::left << "   --timer"         << std::setw(20) << "Timer on/off" << std::endl;
    std::cout << std::endl;
}

void parse_args(int argc, char* argv[], DG::Parameters& params)
{
    std::vector<std::string> args(argv, argv+argc);
    int i=1;
    while (i < argc) {
        if (args[i] == "--p") {
            params.p = atoi(argv[i+1]);
            i+=2;
        } else if (args[i] == "--h") {
            params.dx = atof(argv[i+1]);
            i+=2;
        } else if (args[i] == "--adaptive") {
            params.adaptive = true;
            i++;
        } else if (args[i] == "--coarsening") {
            if (args[i+1] == "equal") {
                params.coarsening = DG::kEqualCoarsening;
            } else if (args[i+1] == "rapid") {
                params.coarsening = DG::kRapidCoarsening;
            } else {
                print_help();
                exit(0);
            }
            i+=2;
        } else if (args[i] == "--bc") {
            if (args[i+1] == "dirichlet") {
                params.bctype = DG::kDirichlet;
            } else if (args[i+1] == "neumann") {
                params.bctype = DG::kNeumann;
            } else if (args[i+1] == "periodic") {
                params.bctype = DG::kPeriodic;
            } else {
                print_help();
                exit(0);
            }
            i+=2;
        } else if (args[i] == "--tau0") {
            params.tau0 = atof(argv[i+1]);
            i+=2;
        } else if (args[i] == "--tauD") {
            params.tauD = atof(argv[i+1]);
            i+=2;
        } else if (args[i] == "--solver") {
            if (args[i+1] == "cg") {
                params.solvetype = DG::kCG;
            } else if (args[i+1] == "mgpcg") {
                params.solvetype = DG::kMGPCG;
            }  else if (args[i+1] == "vcycles") {
                params.solvetype = DG::kVCycles;
            } else {
                print_help();
                exit(0);
            }
            i+=2;
        } else if (args[i] == "--test") {
            if (args[i+1] == "solution") {
                params.testtype = DG::kSolution;
            } else if (args[i+1] == "convergence") {
                params.testtype = DG::kConvergence;
            } else {
                print_help();
                exit(0);
            }
            i+=2;
        } else if (args[i] == "--cg.tol") {
            params.cgtol = atof(argv[i+1]);
            i+=2;
        } else if (args[i] == "--cg.maxit") {
            params.cgmaxit = atoi(argv[i+1]);
            i+=2;
        } else if (args[i] == "--mg.relaxation") {
            if (args[i+1] == "jacobi") {
                params.multigrid.relaxation = DG::MG::kJacobi;
            } else if (args[i+1] == "gauss-seidel") {
                params.multigrid.relaxation = DG::MG::kGaussSeidel;
            } else {
                print_help();
                exit(0);
            }
            i+=2;
        } else if (args[i] == "--mg.omega") {
            params.multigrid.omega = atof(argv[i+1]);
            i+=2;
        } else if (args[i] == "--mg.solver") {
            if (args[i+1] == "cholesky") {
                params.multigrid.solver = DG::MG::kCholesky;
            } else if (args[i+1] == "relaxation") {
                params.multigrid.solver = DG::MG::kRelaxation;
            } else {
                print_help();
                exit(0);
            }
            i+=2;
        } else if (args[i] == "--mg.npre") {
            params.multigrid.npre = atof(argv[i+1]);
            i+=2;
        } else if (args[i] == "--mg.npost") {
            params.multigrid.npost = atof(argv[i+1]);
            i+=2;
        } else if (args[i] == "--mg.ncycle") {
            params.ncycle = atoi(argv[i+1]);
            i+=2;
        } else if (args[i] == "--mg.restriction") {
            if (args[i+1] == "RAT") {
                params.multigrid.restriction = DG::MG::kRAT;
            } else if (args[i+1] == "RATRAT") {
                params.multigrid.restriction = DG::MG::kRATRAT;
            } else {
                print_help();
                exit(0);
            }
            i+=2;
        } else if (args[i] == "--dump") {
            params.dump = true;
            i++;
        } else if (args[i] == "--timer") {
            DG::Timer::on();
            i++;
        } else {
            print_help();
            exit(0);
        }
    }
}

/** @brief Run a simulation with the given parameters */
template<int N, int P>
void run(const DG::Parameters& params)
{
    auto h_uniform  = [dx = params.dx](const DG::Tuple<double,N>& x) {
        return DG::Tuple<double,N>(dx);
    };
    auto h_adaptive = [dx = params.dx](const DG::Tuple<double,N>& x) {
        double a = (x[0] > 0.5) ? dx : (x[0] > 0.25) ? 0.25*dx : 0.5*dx;
        return DG::Tuple<double,N>(a);
    };
    auto h = [&](const DG::Tuple<double,N>& x) {
        return params.adaptive ? h_adaptive(x) : h_uniform(x);
    };

    DG::Timer::tic();
    DG::Quadtree<N> qt(h, params.coarsening, params.bctype == DG::kPeriodic);
    DG::Timer::toc("Build quadtree");

    DG::Timer::tic();
    DG::Mesh<N,P> mesh(qt);
    DG::Timer::toc("Build mesh");

    // Set up the test
    DG::Function<N,P> u(mesh), u_true(mesh), f(mesh);
    DG::BoundaryConditions<N,P> bcs(mesh);
    std::function<double(DG::Tuple<double,N>)> ufun, ffun;
    if (params.testtype == DG::kSolution) {
        switch (params.bctype) {
            case DG::kDirichlet:
                ufun = [](DG::Tuple<double,N> x) { return cos(2*M_PI*x[0])*cos(2*M_PI*x[1]) + x[0] + x[1]; };
                ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*cos(2*M_PI*x[0])*cos(2*M_PI*x[1]); };
                bcs = DG::BoundaryConditions<N,P>::Dirichlet(mesh, ufun);
                break;
            case DG::kNeumann:
                ufun = [](DG::Tuple<double,N> x) { return cos(2*M_PI*x[0])*cos(2*M_PI*x[1]) + x[0]*(1-x[0]) + x[1]*(1-x[1]); };
                ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*cos(2*M_PI*x[0])*cos(2*M_PI*x[1]) + 4.0; };
                bcs = DG::BoundaryConditions<N,P>::Neumann(mesh, -1.0);
                break;
            case DG::kPeriodic:
                ufun = [](DG::Tuple<double,N> x) { return sin(2*M_PI*x[0])*sin(2*M_PI*x[1]); };
                ffun = [](DG::Tuple<double,N> x) { return 8*M_PI*M_PI*sin(2*M_PI*x[0])*sin(2*M_PI*x[1]); };
                bcs = DG::BoundaryConditions<N,P>::Periodic(mesh);
                break;
            default:
                throw std::invalid_argument("Unknown boundary condition.");
        }
        u.reset(0);
        u_true.reset(ufun);
        f.reset(ffun);
    } else if (params.testtype == DG::kConvergence) {
        u.vec().setRandom();
        u_true.reset(0);
        f.reset(0);
        switch (params.bctype) {
            case DG::kDirichlet:
                bcs = DG::BoundaryConditions<N,P>::Dirichlet(mesh);
                break;
            case DG::kNeumann:
                bcs = DG::BoundaryConditions<N,P>::Neumann(mesh);
                break;
            case DG::kPeriodic:
                bcs = DG::BoundaryConditions<N,P>::Periodic(mesh);
                break;
            default:
                throw std::invalid_argument("Unknown boundary condition.");
        }
    } else {
        throw std::invalid_argument("Unknown test type.");
    }

    if (params.testtype == DG::kSolution) u_true.write("data/u_true.fun");

    // Discretize the Poisson problem
    DG::LDGPoisson<N,P> poisson(mesh, bcs, params.tau0, params.tauD);

    // Add the forcing function to the RHS
    DG::Function<N,P> rhs = poisson.computeRHS(f);
    if (params.bctype == DG::kNeumann ||
        params.bctype == DG::kPeriodic) {
        rhs.meanZero();
    }

    // Solve
    if (params.solvetype == DG::kCG) {
        DG::Timer::tic();
        DG::pcg(poisson.ops()->A, rhs.vec(), u.vec(), params.cgtol, params.cgmaxit);
        DG::Timer::toc("Solve using CG");
    } else {
        DG::Timer::tic();
        DG::InterpolationHierarchy<N,P> hierarchy(qt);
        DG::Multigrid<N,P> mg(poisson.ops(), hierarchy, params.multigrid);
        DG::Timer::toc("Build multigrid hierarchy");

        if (params.solvetype == DG::kMGPCG) {
            auto precon = [&mg](const DG::Vector& b) {
                mg.solution().setZero();
                mg.rhs() = b;
                mg.vcycle();
                return mg.solution();
            };
            DG::Timer::tic();
            DG::pcg(poisson.ops()->A, rhs.vec(), u.vec(), precon, params.cgtol, params.cgmaxit);
            DG::Timer::toc("Solve using MGPCG");
        } else if (params.solvetype == DG::kVCycles) {
            DG::Timer::tic();
            mg.solution() = u.vec();
            mg.rhs() = rhs.vec();
            DG::Vector r = mg.residual();
            double rnorm_old;
            std::cout << std::endl;
            std::cout << std::setw(10) << std::left << "Cycle";
            std::cout << std::setw(20) << std::left << "Residual";
            std::cout << std::setw(20) << std::left << "Convergence";
            std::cout << std::endl << std::endl;
            for (int i=0; i<params.ncycle; ++i) {
                mg.vcycle();
                rnorm_old = r.norm();
                r = mg.residual();
                std::cout << std::setw(10) << std::left << i;
                std::cout << std::setw(20) << std::left << r.norm();
                std::cout << std::setw(20) << std::left << r.norm()/rnorm_old;
                std::cout << std::endl;
            }
            u.vec() = mg.solution();
            std::stringstream ss;
            ss << "Solve using " << params.ncycle << " V-cycle" << (params.ncycle>1 ? "s" : "");
            DG::Timer::toc(ss.str());
        } else {
            throw std::invalid_argument("Unknown solve type.");
        }

        if (params.dump) {
            DG::Timer::tic();
            DG::Matrix V = mg.vcycle_matrix();
            DG::Timer::toc("Compute V-cycle matrix");
            DG::Timer::tic();
            std::ofstream ofs("data/V.dat");
            ofs.precision(std::numeric_limits<double>::max_digits10);
            ofs << V << '\n';
            ofs.close();
            DG::Timer::toc("Write V-cycle matrix");
        }
    }

    // Output the data
    DG::Timer::tic();
    u.write("data/u.fun");
    if (params.testtype == DG::kSolution) u_true.write("data/u_true.fun");
    if (params.dump) poisson.dump();
    DG::Timer::toc("Output data");
}

int main(int argc, char* argv[])
{
    DG::Timer::off();
    const int N = 2; // Dimension
    DG::Parameters params;
    parse_args(argc, argv, params);

    switch (params.p) {
        case 0:
            run<N,1>(params);
            break;
        case 1:
            run<N,2>(params);
            break;
        case 2:
            run<N,3>(params);
            break;
        case 3:
            run<N,4>(params);
            break;
        case 4:
            run<N,5>(params);
            break;
        case 5:
            run<N,6>(params);
            break;
        default:
            throw std::invalid_argument("Invalid polynomial order.");
    }

    return 0;
}

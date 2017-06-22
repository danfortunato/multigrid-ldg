#ifndef BOUNDARY_CONDITIONS
#define BOUNDARY_CONDITIONS

#include "mesh.h"
#include "function.h"

namespace DG
{
    enum BoundaryType
    {
        kDirichlet,
        kNeumann,
        kRobin
    };

    /** @brief A boundary condition
     *
     *  A boundary condition is a pair consisting of the type of the boundary
     *  condition and the boundary value */
    template<int P, int N>
    struct BoundaryCondition
    {
        BoundaryCondition(BoundaryType type_) : type(type_), f(0) {}
        BoundaryCondition(BoundaryType type_, Function<P,N> f_) : type(type_), f(f_) {}
        BoundaryType type;
        Function<P,N> f;
    };

    /** @brief A collection of boundary conditions for a mesh */
    template<int P, int N>
    struct BoundaryConditions
    {
        /** Construct empty boundary conditions for a mesh */
        BoundaryConditions(Mesh<P,N>& mesh_) : mesh(mesh_) {}

        /** Construct uniform boundary conditions for a mesh */
        BoundaryConditions(Mesh<P,N>& mesh_, BoundaryCondition<P,N> bc) :
            mesh(mesh_)
        {
            for (int i = 0; i < mesh.boundaryIndices.size(); ++i) {
                bcmap.emplace(mesh.boundaryIndices[i], bc);
            }
        }

        /** Construct zero Dirichlet conditions for a mesh */
        static BoundaryConditions<P,N> Dirichlet(Mesh<P,N>& mesh_)
        {
            return BoundaryConditions<P,N>(mesh, BoundaryCondition<P,N>(kDirichlet));
        }

        /** Construct given Dirichlet conditions for a mesh */
        static BoundaryConditions<P,N> Dirichlet(Mesh<P,N>& mesh, Function<P,N> f)
        {
            return BoundaryConditions<P,N>(mesh, BoundaryCondition<P,N>(kDirichlet, f));
        }

        /** Construct zero Neumann conditions for a mesh */
        static BoundaryConditions<P,N> Neumann(Mesh<P,N>& mesh)
        {
            return BoundaryConditions<P,N>(mesh, BoundaryCondition<P,N>(kNeumann));
        }

        /** Construct given Neumann conditions for a mesh */
        static BoundaryConditions<P,N> Neumann(Mesh<P,N>& mesh, Function<P,N> f)
        {
            return BoundaryConditions<P,N>(mesh, BoundaryCondition<P,N>(kNeumann, f));
        }

        /** Construct zero Robin conditions for a mesh */
        static BoundaryConditions<P,N> Robin(Mesh<P,N>& mesh)
        {
            return BoundaryConditions<P,N>(mesh, BoundaryCondition<P,N>(kRobin));
        }

        /** Construct given Robin conditions for a mesh */
        static BoundaryConditions<P,N> Robin(Mesh<P,N>& mesh, Function<P,N> f)
        {
            return BoundaryConditions<P,N>(mesh, BoundaryCondition<P,N>(kRobin, f));
        }
        
        /** The mesh */
        const Mesh<P,N>& mesh;

        /** A mapping between the geometric boundary indices and the boundary
         *  condition to apply on each geometric boundary */
        std::map<int, BoundaryCondition> bcmap;
    };
}

#endif

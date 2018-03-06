#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <unordered_map>
#include <functional>
#include "common.h"
#include "mesh.h"

namespace DG
{
    enum BoundaryType
    {
        kDirichlet,
        kNeumann
    };

    /** @brief A boundary condition
     *
     *  A boundary condition is a pair consisting of the type of the boundary
     *  condition and the boundary value */
    template<int N, int P>
    struct BoundaryCondition
    {
        BoundaryCondition(BoundaryType type_, double value = 0) : type(type_), f([value](Tuple<double,N>) { return value; }) {}
        BoundaryCondition(BoundaryType type_, const std::function<double(Tuple<double,N>)>& f_) : type(type_), f(f_) {}
        BoundaryType type;
        std::function<double(Tuple<double,N>)> f;
    };

    /** @brief A collection of boundary conditions for a mesh */
    template<int N, int P>
    struct BoundaryConditions
    {
        /** Construct empty boundary conditions for a mesh */
        BoundaryConditions(const Mesh<N,P>& mesh_) : mesh(&mesh_) {}

        /** Construct uniform boundary conditions for a mesh */
        BoundaryConditions(const Mesh<N,P>& mesh_, BoundaryCondition<N,P> bc) :
            mesh(&mesh_)
        {
            for (int bnd=0; bnd<mesh->ngb; ++bnd) {
                bcmap.emplace(bnd+1, bc);
            }
        }

        /** Construct zero Dirichlet conditions for a mesh */
        static BoundaryConditions<N,P> Dirichlet(const Mesh<N,P>& mesh_, double value = 0)
        {
            return BoundaryConditions<N,P>(mesh_, BoundaryCondition<N,P>(kDirichlet, value));
        }

        /** Construct given Dirichlet conditions for a mesh */
        static BoundaryConditions<N,P> Dirichlet(const Mesh<N,P>& mesh_, const std::function<double(Tuple<double,N>)>& f)
        {
            return BoundaryConditions<N,P>(mesh_, BoundaryCondition<N,P>(kDirichlet, f));
        }

        /** Construct zero Neumann conditions for a mesh */
        static BoundaryConditions<N,P> Neumann(const Mesh<N,P>& mesh_, double value = 0)
        {
            return BoundaryConditions<N,P>(mesh_, BoundaryCondition<N,P>(kNeumann, value));
        }

        /** Construct given Neumann conditions for a mesh */
        static BoundaryConditions<N,P> Neumann(const Mesh<N,P>& mesh_, const std::function<double(Tuple<double,N>)>& f)
        {
            return BoundaryConditions<N,P>(mesh_, BoundaryCondition<N,P>(kNeumann, f));
        }
        
        /** The mesh */
        const Mesh<N,P>* mesh;

        /** A mapping between the geometric boundary indices and the boundary
         *  condition to apply on each geometric boundary */
        std::unordered_map<int, BoundaryCondition<N,P>> bcmap;
    };
}

#endif

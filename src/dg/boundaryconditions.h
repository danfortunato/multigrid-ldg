#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <unordered_map>
#include <functional>
#include "mesh.h"

namespace DG
{
    enum BoundaryType
    {
        kDirichlet,
        kNeumann,
        kPeriodic
    };

    /** @brief A boundary condition
     *
     *  A boundary condition is a pair consisting of the type of the boundary
     *  condition and the boundary value */
    template<int P, int N>
    struct BoundaryCondition
    {
        BoundaryCondition(BoundaryType type_, double value = 0) : type(type_), f([value](Tuple<double,N>) { return value; }) {}
        BoundaryCondition(BoundaryType type_, const std::function<double(Tuple<double,N>)>& f_) : type(type_), f(f_) {}
        BoundaryType type;
        std::function<double(Tuple<double,N>)> f;
    };

    /** @brief A collection of boundary conditions for a mesh */
    template<int P, int N>
    struct BoundaryConditions
    {
        /** Construct empty boundary conditions for a mesh */
        BoundaryConditions(const Mesh<P,N>& mesh_) : mesh(&mesh_) {}

        /** Construct uniform boundary conditions for a mesh */
        BoundaryConditions(const Mesh<P,N>& mesh_, BoundaryCondition<P,N> bc) :
            mesh(&mesh_)
        {
            for (int bnd : mesh->boundaryIndices) {
                bcmap.emplace(bnd, bc);
            }
        }

        /** Construct zero Dirichlet conditions for a mesh */
        static BoundaryConditions<P,N> Dirichlet(const Mesh<P,N>& mesh_, double value = 0)
        {
            return BoundaryConditions<P,N>(mesh_, BoundaryCondition<P,N>(kDirichlet, value));
        }

        /** Construct given Dirichlet conditions for a mesh */
        static BoundaryConditions<P,N> Dirichlet(const Mesh<P,N>& mesh_, const std::function<double(Tuple<double,N>)>& f)
        {
            return BoundaryConditions<P,N>(mesh_, BoundaryCondition<P,N>(kDirichlet, f));
        }

        /** Construct zero Neumann conditions for a mesh */
        static BoundaryConditions<P,N> Neumann(const Mesh<P,N>& mesh_, double value = 0)
        {
            return BoundaryConditions<P,N>(mesh_, BoundaryCondition<P,N>(kNeumann, value));
        }

        /** Construct given Neumann conditions for a mesh */
        static BoundaryConditions<P,N> Neumann(const Mesh<P,N>& mesh_, const std::function<double(Tuple<double,N>)>& f)
        {
            return BoundaryConditions<P,N>(mesh_, BoundaryCondition<P,N>(kNeumann, f));
        }

        /** Construct periodic conditions for a mesh */
        static BoundaryConditions<P,N> Periodic(const Mesh<P,N>& mesh_)
        {
            return BoundaryConditions<P,N>(mesh_, BoundaryCondition<P,N>(kPeriodic));
        }
        
        /** The mesh */
        const Mesh<P,N>* mesh;

        /** A mapping between the geometric boundary indices and the boundary
         *  condition to apply on each geometric boundary */
        std::unordered_map<int, BoundaryCondition<P,N>> bcmap;
    };
}

#endif

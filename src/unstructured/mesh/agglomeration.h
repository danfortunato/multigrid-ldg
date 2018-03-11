#ifndef AGGLOMERATION_H
#define AGGLOMERATION_H

#include <vector>
#include <unordered_map>
#include "common.h"
#include "master.h"
#include "mesh.h"
#include "geometry.h"
extern "C" {
    #include "mgridgen.h"
}

namespace DG
{
    template<int N, int P>
    struct AgglomeratedMesh;
    template<int N, int P>
    struct AgglomeratedElement;

    template<int N, int P>
    struct Agglomeration
    {
        Agglomeration(const Mesh<N,P>& mesh_, int maxAgglom = 10) :
            mesh(&mesh_)
        {
            if (maxAgglom > 0) {
                aggloms.reserve(maxAgglom);
                aggloms.emplace_back(*mesh);
                for (int i=0; i<maxAgglom-1 && aggloms[i].ne>1; ++i) {
                    aggloms.emplace_back(aggloms[i]);
                }
            }
        }

        /** @brief The number of agglomerations */
        int numAgglom() const
        {
            return aggloms.size();
        }

        /** @brief The original mesh */
        const Mesh<N,P>* mesh;
        /** @brief The nested sequence of agglomerated meshes */
        std::vector<AgglomeratedMesh<N,P>> aggloms;
    };

    /** @brief An agglomerated mesh */
    template<int N, int P>
    struct AgglomeratedMesh
    {
        /** @brief Construct an agglomerated mesh from an initial mesh */
        AgglomeratedMesh(const Mesh<N,P>& mesh) :
            orig_mesh(&mesh),
            finer_mesh(nullptr)
        {
            // Create uncompressed adjacency list of original mesh
            std::unordered_map<int,std::vector<int>> t2f;
            for (int i=0; i<orig_mesh->nf; ++i) {
                auto f = orig_mesh->faces[i];
                t2f[f.left].push_back(i);
                if (f.right>=0) t2f[f.right].push_back(-i);
            }
            // Each element should now have an entry in t2f

            // Create the volume and surface area arrays
            std::vector<double> vvol(orig_mesh->ne,0), vsurf(orig_mesh->ne,0);
            for (int i=0; i<orig_mesh->ne; ++i) vvol[i] = orig_mesh->elements[i].volume();

            // Create the compressed adjacency list
            std::vector<int> adjacency, xadj;
            std::vector<double> adjweight;
            adjacency.resize(2*(orig_mesh->nf-orig_mesh->nb));
            adjweight.resize(2*(orig_mesh->nf-orig_mesh->nb));
            xadj.resize(orig_mesh->ne+1);
            xadj[0] = 0;
            for (int i=0; i<orig_mesh->ne; ++i) {
                const std::vector<int>& ff = t2f.at(i);
                assert(ff.size() == N+1);
                xadj[i+1] = xadj[i];
                for (int jj : ff) {
                    int j = jj>=0 ? jj : -jj;
                    int neighbor = jj>=0 ? orig_mesh->faces[j].right : orig_mesh->faces[j].left;
                    if (neighbor >= 0) {
                        adjacency[xadj[i+1]] = neighbor;
                        adjweight[xadj[i+1]] = orig_mesh->faces[j].area();
                        xadj[i+1]++;
                    } else {
                        // Add this boundary face's area for this element
                        vsurf[i] += orig_mesh->faces[j].area();
                    }
                }
            }

            // Initialize options and output data
            int minsize = 1;
            int maxsize = 1 << N;
            int options[] = {4, 6, 1, N};
            int nmoves;
            orig_partition.resize(orig_mesh->ne);

            // Call MGridGen to do the agglomeration
            MGridGen(
                orig_mesh->ne,
                xadj.data(),
                vvol.data(),
                vsurf.data(),
                adjacency.data(),
                adjweight.data(),
                minsize,
                maxsize,
                options,
                &nmoves,
                &ne,
                orig_partition.data()
            );

            // Create agglomerated elements from the partition of the original mesh
            elements.reserve(ne);
            for (int i=0; i<ne; ++i) elements.emplace_back(*this, i);
            for (int i=0; i<orig_mesh->ne; ++i) {
                int aggi = orig_partition[i];
                elements[aggi].addElement(i);
            }
        }

        /** @brief Construct an agglomerated mesh by agglomerating another
         *         agglomerated mesh */
        AgglomeratedMesh(const AgglomeratedMesh<N,P>& finer) :
            orig_mesh(finer.orig_mesh),
            finer_mesh(&finer)
        {
            // Create uncompressed adjacency list of the finer agglomerated mesh
            std::unordered_map<int,std::vector<int>> agg2f;
            for (int i=0; i<orig_mesh->nf; ++i) {
                auto f = orig_mesh->faces[i];
                if (f.right>=0) {
                    int aggl = finer_mesh->orig_partition[f.left];
                    int aggr = finer_mesh->orig_partition[f.right];
                    if (aggl != aggr) {
                        // This face is between two agglomerated elements
                        agg2f[aggl].push_back(i);
                        agg2f[aggr].push_back(-i);
                    }
                } else {
                    int aggl = finer_mesh->orig_partition[f.left];
                    agg2f[aggl].push_back(i);
                }
            }
            // Each agglomerated element should now have an entry in agg2f

            // Create the volume and surface area arrays
            std::vector<double> vvol(finer_mesh->ne,0), vsurf(finer_mesh->ne,0);
            for (int i=0; i<orig_mesh->ne; ++i) vvol[finer_mesh->orig_partition[i]] += orig_mesh->elements[i].volume();

            // Create the compressed adjacency list
            std::vector<int> adjacency, xadj;
            std::vector<double> adjweight;
            xadj.resize(finer_mesh->ne+1);
            xadj[0] = 0;
            for (int i=0; i<finer_mesh->ne; ++i) {
                const std::vector<int>& ff = agg2f.at(i);
                std::unordered_map<int,int> pos;
                xadj[i+1] = xadj[i];
                for (int jj : ff) {
                    int j = jj>=0 ? jj : -jj;
                    int orig_neighbor = jj>=0 ? orig_mesh->faces[j].right : orig_mesh->faces[j].left;
                    if (orig_neighbor >= 0) {
                        int agg_neighbor = finer_mesh->orig_partition[orig_neighbor];
                        if (pos.count(agg_neighbor) == 0) {
                            // This is a new neighbor
                            adjacency.push_back(agg_neighbor);
                            adjweight.push_back(orig_mesh->faces[j].area());
                            pos[agg_neighbor] = xadj[i+1];
                            xadj[i+1]++;
                        } else {
                            // We've seen this neighbor before
                            adjweight[pos[agg_neighbor]] += orig_mesh->faces[j].area();
                        }
                    } else {
                        // Add this boundary face's area for this element
                        vsurf[i] += orig_mesh->faces[j].area();
                    }
                }
            }

            // Initialize options and output data
            int minsize = 1;
            int maxsize = 1 << N;
            int options[] = {4, 6, 1, N};
            int nmoves;
            std::vector<int> partition;
            partition.resize(finer_mesh->ne);

            // Call MGridGen to do the agglomeration
            MGridGen(
                finer_mesh->ne,
                xadj.data(),
                vvol.data(),
                vsurf.data(),
                adjacency.data(),
                adjweight.data(),
                minsize,
                maxsize,
                options,
                &nmoves,
                &ne,
                partition.data()
            );

            // Compose the finer mesh's partition of the original mesh with this
            // mesh's partition of the finer mesh
            orig_partition.resize(orig_mesh->ne);
            for (int i=0; i<orig_mesh->ne; ++i) {
                orig_partition[i] = partition[finer_mesh->orig_partition[i]];
            }

            // Create agglomerated elements from the partition of the finer agglomerated mesh
            elements.reserve(ne);
            for (int i=0; i<ne; ++i) elements.emplace_back(*this, i);
            for (int i=0; i<finer_mesh->ne; ++i) {
                int aggi = partition[i];
                elements[aggi].addAgglomeratedElement(i);
            }
        }

        /** @brief Write the agglomerated mesh to a file */
        void write(const std::string& file)
        {
            std::ofstream ofs(file);
            ofs.precision(std::numeric_limits<double>::max_digits10);
            for (int aggi : orig_partition) ofs << aggi << std::endl;
            ofs.close();
        }

        /** @brief The number of agglomerated elements */
        int ne;
        /** @brief The partition of the original mesh defining the agglomeration */
        std::vector<int> orig_partition;
        /** @brief The agglomerated elements */
        std::vector<AgglomeratedElement<N,P>> elements;
        /** @brief The original mesh */
        const Mesh<N,P>* orig_mesh;
        /** @brief The finer mesh from which this mesh agglomerated */
        const AgglomeratedMesh<N,P>* finer_mesh;
    };

    /** @brief An agglomerated element */
    template<int N, int P>
    struct AgglomeratedElement
    {
        AgglomeratedElement(const AgglomeratedMesh<N,P>& aggmesh_, int lid_) :
            lid(lid_),
            vol(0),
            aggmesh(&aggmesh_)
        {}

        /** Add an element to the agglomeration */
        void addElement(int elem)
        {
            orig_elems.push_back(elem);
            vol += aggmesh->orig_mesh->elements[elem].volume();
            for (int j=0; j<N+1; ++j) {
                bounding_box.lower = bounding_box.lower.min(aggmesh->orig_mesh->elements[elem].simplex.p[j]);
                bounding_box.upper = bounding_box.upper.max(aggmesh->orig_mesh->elements[elem].simplex.p[j]);
            }
        }

        /** Add an agglomerated element to the agglomeration */
        void addAgglomeratedElement(int elem)
        {
            finer_elems.push_back(elem);
            vol += aggmesh->finer_mesh->elements[elem].volume();
            bounding_box.lower = bounding_box.lower.min(aggmesh->finer_mesh->elements[elem].bounding_box.lower);
            bounding_box.upper = bounding_box.upper.max(aggmesh->finer_mesh->elements[elem].bounding_box.upper);
        }

        /** @brief The volume of the agglomerated element */
        double volume() const
        {
            return vol;
        }

        /** @brief The index-th node in the element */
        Tuple<double,N> dgnodes(const Tuple<int,N>& index) const
        {
            return Cartesian::Master<N,P>::dgnodes(index, bounding_box);
        }

        /** @brief The local ID of this agglomerated element */
        int lid;
        /** @brief The element IDs in the original mesh defining this agglomerated element */
        std::vector<int> orig_elems;
        /** @brief The element IDs in the finer mesh defining this agglomerated element */
        std::vector<int> finer_elems;
        /** @brief The volume */
        double vol;
        /** @brief The bounding box */
        Cell<N> bounding_box;
        /** @brief The agglomerated mesh */
        const AgglomeratedMesh<N,P>* aggmesh;
    };
}

#endif

#ifndef WIREFRAME_H
#define WIREFRAME_H

#include <vector>
#include <fstream>
#include <sstream>
#include "common.h"

namespace DG
{
    /** @brief A wireframe for an unstructured mesh */
    template<int N>
    struct Wireframe
    {
        /** @brief Construct a wireframe from a set of points and simplices */
        Wireframe(const std::vector<Tuple<double,N>>& p_, const std::vector<Tuple<int,N+1>>& t_) :
            np(p_.size()),
            nt(t_.size()),
            p(p_),
            t(t_)
        {}

        /** @brief Read in a wireframe from a file */
        Wireframe(const std::string& file)
        {
            std::ifstream ifs(file);
            std::stringstream ss;
            std::string line;

            if (ifs.is_open()) {

                // Read headers
                std::getline(ifs, line);
                ss.clear();
                ss.str(line);
                ss >> np >> nt;
                p.reserve(np);
                t.reserve(nt);

                // Read p
                std::getline(ifs, line);
                while (std::getline(ifs, line) && !line.empty()) {
                    ss.clear();
                    ss.str(line);
                    Tuple<double,N> point;
                    double val;
                    int i=0;
                    while (ss >> val) point[i++] = val;
                    assert(i == N);
                    p.push_back(point);
                }

                // Read t
                while (std::getline(ifs, line) && !line.empty()) {
                    ss.clear();
                    ss.str(line);
                    Tuple<int,N+1> tri;
                    int val;
                    int i=0;
                    while (ss >> val) tri[i++] = val;
                    assert(i == N+1);
                    t.push_back(tri-1); // Shift 1-index to 0-index
                }

                ifs.close();
            }
        }

        /** @brief The number of points */
        int np;
        /** @brief The number of simplices */
        int nt;
        /** @brief The points (np x N) */
        std::vector<Tuple<double,N>> p;
        /** @brief The simplices (nt x N+1) */
        std::vector<Tuple<int,N+1>> t;
    };
}

#endif

//
// Created by Liam Curtis on 2024-06-09.
//

#ifndef METALFOAMS_MESH_PARAMETRIZATION_H
#define METALFOAMS_MESH_PARAMETRIZATION_H

#include <lf/uscalfe/uscalfe.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/io/io.h>

#include "linear_elasticity_assembler.h"
#include "parametric_matrix_computation.h"
#include "line_mapping.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <gmsh.h>

// This file will contain the structure of the primary class I will use to hold the lower dimensional mesh
// The class will consist of terminal points (3 per "branch"), a width for each branch, and a vector for the direction of each terminal point
// It will also contain helper functions to ensure that a given parametrization is valid
//      e.g. branches don't overlap anywhere except for at the center, the branches share points at the center
// It will also contain a helper function returning the necessary points in order to create the mesh using the Gmsh API

// TODO: IDEA, when splitting up the overall mesh into its constituent components, create a graph that represents the
//  "distance" to other parts of the mesh, and I can repeatedly check if this distance is respected, linked nodes may
//  have the compatibility condition between their _terminals, "non-linked but close" nodes will be checked for overlap
//  because it is localized this can be checked at all times, maybe?

// TODO: Make sure that the meshing algorithm has the entire center point if branches meet

// TODO: In general, when the NN is looking for an optimal solution, fix the amount by which points can move so it stays linearly elastic
// TODO: This class will likely include the energy difference calculator of two meshes

// TODO: Make sure that _widths is not an alterable parameter in the NN algorithm (this is subject to change)

// TODO: Ensure that during creation of each NN section, that the vector is approximately orthogonal to the mesh path

// TODO: Make sure that within the NN training algorithm, the compatibility conditions share the same points

// TODO: Clean up this code

using Tuple3i = std::tuple<int, int, int>;

// TODO: Refactor this code so that the meshparamdata is easily alterable on its own
// TODO: Consider calling meshParamValidator when initializing a MeshParametrizationData object
struct MeshParametrizationData {
    int numBranches;
    //TODO: Consider making _widths an alterable parameter, and let the compatibility conditions make sure the
    // remaining "constraints", such as with nodes, hold up, this might change the structure of widths to a matrix
    Eigen::VectorXd widths;
    Eigen::MatrixXd terminals;
    Eigen::MatrixXd vectors;

    // TODO: Decide if this default constructor should actually be here
    MeshParametrizationData() = default;

    // TODO: Perhaps include a few asserts to make sure that the shapes of all these matrices make sense
    MeshParametrizationData(int num, Eigen::VectorXd w, Eigen::MatrixXd t, Eigen::MatrixXd v)
            : numBranches(num), widths(std::move(w)), terminals(std::move(t)), vectors(std::move(v)) {
        // TODO: Might need to change this if widths ends up changing
        LF_ASSERT_MSG(num == w.size(),
                      "Number of branches is not equal to the amount of _widths given in MeshParam");
        LF_ASSERT_MSG(t.rows() / 2 == num and t.cols() == 3,
                      "Size of _terminals is invalid in MeshParam");
        LF_ASSERT_MSG(v.rows() / 2 == num and v.cols() == 3, "Size of _vectors is invalid in MeshParam");
    }
};

namespace MeshParametrization {

    std::string getPointKey(double x, double y, double z, double tolerance);

    void normalizeVectors(Eigen::MatrixXd& vectors);
    Eigen::MatrixXd polynomialPoints(MeshParametrizationData &param);
    void angleVectors(int numBranch, Eigen::MatrixXd& poly_points);
    void intersectionVectors(int numBranch, Eigen::MatrixXd& poly_points);
    bool intersectionBranches(Eigen::MatrixXd& poly_points, int first, int second);
    bool meshParamValidator(const int num, Eigen::VectorXd& width, Eigen::MatrixXd &terminal, Eigen::MatrixXd &vector);
    std::pair<bool, double> displacementEnergy(MeshParametrizationData &first, MeshParametrizationData &second);

    //TODO: Change the signature of this function to work off of displacement energy
    bool elasticRegion(MeshParametrizationData &first, MeshParametrizationData &second);

    Eigen::MatrixXd connectionPoints(MeshParametrizationData &multiBranch);
    void generateMesh(MeshParametrizationData &parametrization, const std::string& mesh_name);

    Eigen::Vector2d displacementBC(Eigen::MatrixXd firstBranch, Eigen::MatrixXd secondBranch, Eigen::Vector2d point);

    void fixFlaggedSolutionComponentsLE(std::vector<std::pair<bool, Eigen::Vector2d>>& selectvals,
                                        lf::assemble::COOMatrix<double>& A, Eigen::Matrix<double, Eigen::Dynamic, 1>& phi);

    // TODO: Decide if this actually has to be private or not, likely not imo

};

#endif //METALFOAMS_MESH_PARAMETRIZATION_H
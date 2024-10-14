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
#include <future>
#include <cassert>
#include <filesystem>
#include <gmsh.h>


// This file will contain the structure of the primary class I will use to hold the lower dimensional mesh
// The class will consist of terminal points (3 per "branch"), a width for each branch, and a vector for the direction of each terminal point
// It will also contain helper functions to ensure that a given parametrization is valid
//      e.g. branches don't overlap anywhere except for at the center, the branches share points at the center
// It will also contain a helper function returning the necessary points in order to create the mesh using the Gmsh API

struct calculationParams {
    double yieldStrength;
    double youngsModulus;
    double poissonRatio;
    double meshSize;
    int order;

    calculationParams(double yStren, double E, double nu, double ms, int o)
        : yieldStrength(yStren), youngsModulus(E), poissonRatio(nu), meshSize(ms), order(o)
    {
        LF_ASSERT_MSG(o == 1 or o == 2, "Value of order must be 1 or 2, but is instead " << o);
    }
};

struct MeshParametrizationData {
    int numBranches{};
    Eigen::MatrixXd widths;
    Eigen::MatrixXd terminals;
    Eigen::MatrixXd vectors;

    MeshParametrizationData() = default;

    MeshParametrizationData(int num, Eigen::MatrixXd w, Eigen::MatrixXd t, Eigen::MatrixXd v)
            : numBranches(num), widths(std::move(w)), terminals(std::move(t)), vectors(std::move(v)) {
        LF_ASSERT_MSG(num == widths.rows() and widths.cols() == 3,
                      "Number of branches is not equal to the amount of _widths given in MeshParam");
        LF_ASSERT_MSG(terminals.rows() / 2 == num and terminals.cols() == 3,
                      "Size of _terminals is invalid in MeshParam");
        LF_ASSERT_MSG(vectors.rows() / 2 == num and vectors.cols() == 3, "Size of _vectors is invalid in MeshParam");
    }
};

namespace MeshParametrization {

    bool checkVectorLengths(const Eigen::MatrixXd &vectors);
    void normalizeVectors(Eigen::MatrixXd& vectors);
    Eigen::MatrixXd polynomialPoints(MeshParametrizationData &param);
    bool angleVectors(MeshParametrizationData &param);
    bool intersectionVectors(MeshParametrizationData &param);
    bool intersectionBranches(const Eigen::MatrixXd &poly_points_1, const Eigen::MatrixXd &poly_points_2);
    bool selfIntersection(const Eigen::MatrixXd &poly_points);
    std::optional<Eigen::VectorXi> connectionPoints(MeshParametrizationData &multiBranch);
    MeshParametrizationData pointToParametrization(const Eigen::MatrixXd &poly_points);


    bool meshParamValidator(MeshParametrizationData &param);

    void generateMesh(MeshParametrizationData &parametrization, const std::string& mesh_name,
        double mesh_size, int order);

    Eigen::Vector2d displacementBC(const Eigen::MatrixXd &branch, const Eigen::MatrixXd &displacement,
                                                     const Eigen::Vector2d &point);
    void fixFlaggedSolutionComponentsLE(std::vector<std::pair<bool, Eigen::Vector2d>>& select_vals,
                                        lf::assemble::COOMatrix<double>& A, Eigen::Matrix<double, Eigen::Dynamic, 1>& phi);

    bool elasticRegion(const Eigen::MatrixXd &stresses, double yieldStrength);
    std::pair<bool, double> displacementEnergy(MeshParametrizationData &param,
        Eigen::MatrixXd &displacement, const calculationParams &calc_params);

};

#endif //METALFOAMS_MESH_PARAMETRIZATION_H
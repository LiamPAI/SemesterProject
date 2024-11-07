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

/**
 * @brief Parameters for finite element calculations and mesh generation
 * @details Contains material properties and mesh configuration parameters to be used for calculations with
 * individual parametrizations and the overall mesh
 */
struct calculationParams {
    double yieldStrength; ///< Material yield strength
    double youngsModulus; ///< Young's modulus (E)
    double poissonRatio;  ///< Poisson's ratio (ν)
    double meshSize;      ///< Mesh element size parameter
    int order;          ///< Finite element order (1 or 2)

    /**
     * @brief Constructs calculation parameters
     * @param yStren Yield strength
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @param ms Mesh size
     * @param o Element order (must be 1 or 2)
     * @throws LFException if order is not 1 or 2
     */
    calculationParams(double yStren, double E, double nu, double ms, int o)
        : yieldStrength(yStren), youngsModulus(E), poissonRatio(nu), meshSize(ms), order(o)
    {
        LF_ASSERT_MSG(o == 1 or o == 2, "Value of order must be 1 or 2, but is instead " << o);
    }
};

/**
 * @brief Container for mesh parametrization geometric data
 * @details Stores branch information including widths, terminal points, and direction vectors
 *          in a locally ordered format
 *
 * Matrix Ordering Convention:
 * For a parametrization with n branches, widths will have shape n by 3, each row representing a new branch,
 * and each column corresponding to the same columns for the terminals and the vectors
 * Both vectors and terminals will have shape 2n by 3, given the 2D nature of the terminal points and vectors
 *
 * Local Ordering Convention:
 * For multi-branch parametrizations, column 0 contains the points related to the "center" shape, such as a triangle
 * a 3 branch multi parametrization
 */
struct MeshParametrizationData {
    int numBranches{};          ///< Number of branches in parametrization
    Eigen::MatrixXd widths;     ///< Matrix of branch widths (numBranches × 3)
    Eigen::MatrixXd terminals;  ///< Matrix of terminal points (2*numBranches × 3)
    Eigen::MatrixXd vectors;    ///< Matrix of direction vectors (2*numBranches × 3)

    /** @brief Default constructor */
    MeshParametrizationData() = default;

    /**
     * @brief Constructs mesh parametrization with brief validation
     * @param num Number of branches
     * @param w Width matrix
     * @param t Terminal points matrix
     * @param v Direction vectors matrix
     * @throws LFException if matrix dimensions don't match branch count
     */
    MeshParametrizationData(int num, Eigen::MatrixXd w, Eigen::MatrixXd t, Eigen::MatrixXd v)
            : numBranches(num), widths(std::move(w)), terminals(std::move(t)), vectors(std::move(v)) {
        LF_ASSERT_MSG(num == widths.rows() and widths.cols() == 3,
                      "Number of branches is not equal to the amount of _widths given in MeshParam");
        LF_ASSERT_MSG(terminals.rows() / 2 == num and terminals.cols() == 3,
                      "Size of _terminals is invalid in MeshParam");
        LF_ASSERT_MSG(vectors.rows() / 2 == num and vectors.cols() == 3, "Size of _vectors is invalid in MeshParam");
    }
};

/**
 * @brief Namespace for mesh parametrization operations
 * @details Provides functionality for mesh generation, validation, and analysis
 */
namespace MeshParametrization {

    /**
     * @brief Validates vector lengths in direction matrix
     * @param vectors Matrix of direction vectors
     * @return true if all vectors have non-zero lengths
     */
    bool checkVectorLengths(const Eigen::MatrixXd &vectors);

    /**
     * @brief Normalizes direction vectors to unit length
     * @param vectors Matrix of direction vectors (modified in-place)
     */
    void normalizeVectors(Eigen::MatrixXd& vectors);

    /**
     * @brief Generates polynomial points from parametrization
     * @details uses @link normalizeVectors @endlink to ensure normalized vectors
     * @param param Mesh parametrization data
     * @return Matrix of polynomial points, equivalent to points field of @link ParametrizationPoints @endlink
     */
    Eigen::MatrixXd polynomialPoints(MeshParametrizationData &param);

    /**
     * @brief Checks angles between branch vectors
     * @param param Mesh parametrization data
     * @return true if all angles are under 45 degrees
     */
    bool angleVectors(MeshParametrizationData &param);

    /**
     * @brief Checks for intersections between vectors
     * @param param Mesh parametrization data
     * @return true if no intersections found
     */
    bool intersectionVectors(MeshParametrizationData &param);

    /**
     * @brief Checks for intersections between nearby branches
     * @details Does not include overlap at end points of the branch
     *
     * @param poly_points_1 Polynomial points of first branch
     * @param poly_points_2 Polynomial points of second branch
     * @return true if branches intersect
     */
    bool intersectionBranches(const Eigen::MatrixXd &poly_points_1, const Eigen::MatrixXd &poly_points_2);

   /**
    * @brief Checks for self-intersections in branch
    * @param poly_points Polynomial points of branch
    * @return true if self-intersection found
    */
    bool selfIntersection(const Eigen::MatrixXd &poly_points);

   /**
    * @brief Identifies connection points in multi-branch parametrization
    * @details As mentioned in the local ordering of @link MeshParametrizationData @endlink, these connection points
    * must be found in the first columns of the relevant fields
    * @param multiBranch Multi-branch parametrization data
    * @return Optional vector of connection indices, std::nullopt if they aren't found
    */
    std::optional<Eigen::VectorXi> connectionPoints(MeshParametrizationData &multiBranch);

   /**
     * @brief Validates complete mesh parametrization
     * @param param Mesh parametrization to validate
     * @return true if parametrization is valid
     */
    bool meshParamValidator(MeshParametrizationData &param);

   /**
    * @brief Converts polynomial points to mesh parametrization
    * @details These polynomial points would be identical to the points field of a
    * @link ParametrizationPoints @endlink object
    * @param poly_points Matrix of polynomial points
    * @return Mesh parametrization data
    */
    MeshParametrizationData pointToParametrization(const Eigen::MatrixXd &poly_points);

    /**
      * @brief Generates mesh from a @link MeshParametrizationData @endlink object
      * @param parametrization Mesh parametrization data
      * @param mesh_name Output mesh file name
      * @param mesh_size Mesh element size (no .msh extension required)
      * @param order Finite element order
      */
    void generateMesh(MeshParametrizationData &parametrization, const std::string& mesh_name,
        double mesh_size, int order);

    /**
     * @brief Calculates displacement boundary condition at point
     * @details There might be multiple nodes on the edge for which displacement boundary conditions are defined.
     * The displacement boundary conditions are provided at the end points of a line segment, this functions help to
     * provide the interpolated displacement BCs for the "internal" nodes of the line segment
     * @param branch Branch geometry
     * @param displacement Displacement boundary conditions
     * @param point Point to calculate displacement at
     * @return 2D displacement vector
     */
    Eigen::Vector2d displacementBC(const Eigen::MatrixXd &branch, const Eigen::MatrixXd &displacement,
                                                     const Eigen::Vector2d &point);

   /**
    * @brief Fixes matrix to format 2.7.6.15 of NUMPDE lecture document
    * @details Given that some displacement boundary conditions are prescribed on the boundary, this function ensures
    * that the corresponding rows of the Galerkin matrix for these rows are altered to the identity, and the
    * corresponding elements in the solution vector are modified as well to reflect this
    * @param select_vals Vector of flagged components and values
    * @param A Galerkin matrix
    * @param phi Solution vector
    */
    void fixFlaggedSolutionComponentsLE(std::vector<std::pair<bool, Eigen::Vector2d>>& select_vals,
                                        lf::assemble::COOMatrix<double>& A, Eigen::Matrix<double, Eigen::Dynamic, 1>& phi);


    /**
     * @brief Checks if stress state is within elastic region of the material
     * @param stresses Matrix of stress components
     * @param yieldStrength Material yield strength
     * @return true if within elastic region
     */
    bool elasticRegion(const Eigen::MatrixXd &stresses, double yieldStrength);

    /**
     * @brief Calculates displacement energy and checks if in the linear elastic region
     * @details Calls @link elasticRegion @endlink for the linear elastic check
     * @param param Mesh parametrization
     * @param displacement Displacement boundary conditions
     * @param calc_params Calculation parameters
     * @return Pair of (is_valid, energy)
     */
    std::pair<bool, double> displacementEnergy(MeshParametrizationData &param,
        Eigen::MatrixXd &displacement, const calculationParams &calc_params);

};

#endif //METALFOAMS_MESH_PARAMETRIZATION_H
//
// Created by Liam Curtis on 2024-05-06.
//

#ifndef GETTINGSTARTED_PARAMETRIC_MATRIX_COMPUTATION_H
#define GETTINGSTARTED_PARAMETRIC_MATRIX_COMPUTATION_H

#include <lf/uscalfe/uscalfe.h>
#include <Eigen/Core>

//The implementation of this class is in parametric_matrix_computation
//This implementation of this class is very similar to that of LinearMatrixComputation, but
// the cells in the mesh are allowed to be 6 node triangles, and 9 node quadrangles with curvilinear edges
//In order to do this, the edges are approximated by a quadratic polynomial
/**
 * @brief Namespace for parametric finite element matrix computations of order 2
 * @details Contains classes for computing stiffness matrices and load vectors
 *          for parametric finite elements with second-order shape functions
 */
namespace ParametricMatrixComputation {

    /**
     * @brief Class for computing parametric finite element stiffness matrices
     * @details Implements stiffness matrix computation for second-order
     *          elements using either plane stress or plane strain formulation
     */
    class ParametricFEElementMatrix {
    private:
        //Declare our parameters for the calculation of the stiffness matrix, using Young's modulus, nu, and
        //either the plane stress of plane strain matrix
        const double youngmodulus{}; ///< Young's modulus (E)
        const double nu{};          ///< Poisson's ratio (ν)
        Eigen::Matrix<double, 3, 3> D; ///< Constitutive matrix (plane stress/strain)

    public:

        //The default constructor is not allowed, as the parameters (Young's modulus, and nu) are required
        //for the computation of the stiffness matrix
        /** @brief Default constructor disabled - material parameters required */
        ParametricFEElementMatrix() = delete;

        //Constructor I intend to use
        /**
         * @brief Constructs element matrix calculator with material properties
         * @param young Young's modulus
         * @param nu Poisson's ratio
         * @param planeStrain true for plane strain, false for plane stress
         * @throws LFException if nu = 0.5 (plane strain) or nu = 1.0 (plane stress)
         */
        ParametricFEElementMatrix(double young, double nu, bool planeStrain) : youngmodulus {young}, nu {nu}
        {
            if (planeStrain) {
                //D is our plane strain matrix, which we initialize here
                LF_ASSERT_MSG(std::abs(nu - 0.5) >= (0.0001), "Value of nu (0.5) will lead to a divide by zero");
                D << (1 - nu), nu, 0.0,
                        nu, (1- nu), 0.0,
                        0.0, 0.0, (1 - 2 * nu) / 2.0;
                D *= youngmodulus / ((1 + nu) * (1 - 2 * nu));
            }
            else {
                //Initialize D as a plane stress matrix
                LF_ASSERT_MSG((std::abs(nu - 1.0) >= (0.0001)), "Value of nu (1.0) will lead to a divide by zero");
                D << 1.0, nu, 0.0,
                    nu, 1.0, 0.0,
                    0.0, 0.0, (1 - nu) / 2.0;
                D *= youngmodulus / (1 - nu * nu);
            }
        }

        // The isActive and Eval function required to send to AssembleMatrixLocally later on
        /**
         * @brief Indicates if element should be processed
         * @return Always returns true as all cells are integrated
         */
        bool isActive (const lf::mesh::Entity&) {return true;} //All cells will be integrated over

        /**
         * @brief Computes element stiffness matrix for given cell
         * @details The returned matrix is of size 18x18 as a quadrilateral cell will have 9 nodes, so 18 entries due
         * to the 2D nature of linear elasticity. For triangular cells, the final 6 rows and 6 columns are simply 0
         * since second-order triangular cells have 6 nodes
         *
         * @param cell Mesh entity (element) to compute matrix for
         * @return 18x18 element stiffness matrix for parametric elements
         */
        Eigen::Matrix<double, 18, 18> Eval(const lf::mesh::Entity &cell);

        // This method will be used for post-processing, taking in the displacement vector, and return matrices
        // for the stress and strains at the element's respective nodes
        // This method is virtually identical to the one in linear_matrix computation
        /**
         * @brief Computes stress and strain at element nodes
         * @param cell Mesh entity to compute for
         * @param disp Global displacement vector
         * @param dofh Degree of freedom handler
         * @return Tuple of (stress matrix, strain matrix, nodal coordinates)
         */
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> stressStrain(const lf::mesh::Entity &cell, Eigen::VectorXd &disp, const lf::assemble::DofHandler &dofh);

        // The purpose of this method is to calculate the energy of a cell once we have obtained the displacement vector
        // Note this method will use stressStrain as part of its calculation
        /**
         * @brief Calculates strain energy for an element
         * @param cell Mesh entity to compute energy for
         * @param disp Global displacement vector
         * @param dofh Degree of freedom handler
         * @return Strain energy of the element
         */
        double energyCalc(const lf::mesh::Entity &cell, Eigen::VectorXd &disp, const lf::assemble::DofHandler &dofh);
    };

    /**
     * @brief Class for computing parametric finite element load vectors (of order 2)
    * @details Handles computation of load vectors including body forces and surface tractions for second-order
    * elements. This was added for completeness in the case that traction and body forces were to be implemented,
    * but is not used within the implementation
     */
    class ParametricFELoadVector {
    private:
        //Body force, boundary flags, and a traction function will be required for the computation of the load vector
        lf::mesh::utils::CodimMeshDataSet<bool> bd_flags;   ///< Boundary flags
        std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction_; ///< Surface traction function
        std::function<Eigen::Vector2d (const Eigen::Vector2d &)> body_f;    ///< Body force function

    public:
        //No default constructor, I want the bd_flags at least to be present when initialized
        /** @brief Default constructor disabled - boundary flags required */
        ParametricFELoadVector() = delete;

        //The case where we don't have traction, 0 Neumann condition, the rhs vector will be 0
        /**
         * @brief Constructs with boundary flags only (zero loads)
         * @param bd_flags Boundary condition flags
         */
        explicit ParametricFELoadVector(lf::mesh::utils::CodimMeshDataSet<bool> bd_flags)
        : bd_flags(std::move(bd_flags)), traction_{}, body_f{} {}

        //The case where there is traction but no body force
        /**
         * @brief Constructs with boundary flags and surface traction
         * @param bd_flags Boundary condition flags
         * @param traction Surface traction function
         */
        ParametricFELoadVector(lf::mesh::utils::CodimMeshDataSet<bool> bd_flags, std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction)
        : bd_flags(std::move(bd_flags)), traction_(std::move(traction)), body_f{} {}

        //Complete constructor, when there is all three
        /**
         * @brief Constructs with boundary flags, traction, and body forces
         * @param bd_flags Boundary condition flags
         * @param traction Surface traction function
         * @param body Body force function
         */
        ParametricFELoadVector(lf::mesh::utils::CodimMeshDataSet<bool> bd_flags, std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction,
                std::function<Eigen::Vector2d (const Eigen::Vector2d &)> body)
        : bd_flags(std::move(bd_flags)), traction_(std::move(traction)), body_f(std::move(body)) {}


        //Return the value at bd_flags for edges, and if there is no body force, don't loop over cells
        /**
         * @brief Determines if entity should be processed
         * @param entity Mesh entity to check
         * @return true if entity should be included in computation
         * @throws LFException for invalid entity types
         */
        bool isActive (const lf::mesh::Entity &entity) {
            const lf::base::RefEl ref_el (entity.RefEl());
            switch (ref_el) {
                case lf::base::RefEl::kSegment(): {
                    return bd_flags(entity);
                }
                case lf::base::RefEl::kTria(): case lf::base::RefEl::kQuad(): {
                    if (!body_f) {
                        return false;
                    }
                    return true;
                }
                default: {
                    LF_ASSERT_MSG(false, "Illegal cell type sent to isActive LinearFELoadVector");
                }
            }
        }

        //Declaration for Eval, this should be able to take both cells and edges, hence the name entity
        //Has size up to 16 since we are now dealing with 8 nodes on a quadrilateral
        /**
         * @brief Computes load vector contribution for entity
         * @details Similar to the Eval function for @link ParametricFEElementMatrix @endlink, the vector is of size 18 for
         * quadrilateral elements. For trianglular cells, the last 6 entries will simply be zero
         * @param entity Mesh entity (element or edge)
         * @return 18-dimensional local load vector
         */
        Eigen::Vector<double, 18> Eval(const lf::mesh::Entity &entity);
    };
}

#endif //GETTINGSTARTED_PARAMETRIC_MATRIX_COMPUTATION_H

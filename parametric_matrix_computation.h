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
namespace ParametricMatrixComputation {

    class ParametricFEElementMatrix {

    private:
        //Declare our parameters for the calculation of the stiffness matrix, using Young's modulus, nu, and
        //either the plane stress of plane strain matrix
        const double youngmodulus{};
        const double nu{};
        Eigen::Matrix<double, 3, 3> D;

    public:

        //The default constructor is not allowed, as the parameters (Young's modulus, and nu) are required
        //for the computation of the stiffness matrix
        ParametricFEElementMatrix() = delete;

        //Constructor I intend to use
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
        bool isActive (const lf::mesh::Entity&) {return true;} //All cells will be integrated over
        Eigen::Matrix<double, 18, 18> Eval(const lf::mesh::Entity &cell);

        // This method will be used for post-processing, taking in the displacement vector, and return matrices
        // for the stress and strains at the element's respective nodes
        // This method is virtually identical to the one in linear_matrix computation
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> stressStrain(const lf::mesh::Entity &cell, Eigen::VectorXd &disp, const lf::assemble::DofHandler &dofh);

        // TODO: Implement this method
        // The purpose of this method is to calculate the energy of a cell once we have obtained the displacement vector
        // Note this method will use stressStrain as part of its calculation
        double energyCalc(const lf::mesh::Entity &cell, Eigen::VectorXd &disp, const lf::assemble::DofHandler &dofh);
    };

    class ParametricFELoadVector {

    private:
        //Body force, boundary flags, and a traction function will be required for the computation of the load vector
        lf::mesh::utils::CodimMeshDataSet<bool> bd_flags;
        std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction_;
        std::function<Eigen::Vector2d (const Eigen::Vector2d &)> body_f;

    public:
        //No default constructor, I want the bd_flags at least to be present when initialized
        ParametricFELoadVector() = delete;

        //The case where we don't have traction, 0 Neumann condition, the rhs vector will be 0
        ParametricFELoadVector(lf::mesh::utils::CodimMeshDataSet<bool> bd_flags)
        : bd_flags(std::move(bd_flags)), traction_{}, body_f{} {}

        //The case where there is traction but no body force
        ParametricFELoadVector(lf::mesh::utils::CodimMeshDataSet<bool> bd_flags, std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction)
        : bd_flags(std::move(bd_flags)), traction_(std::move(traction)), body_f{} {}

        //Complete constructor, when there is all three
        ParametricFELoadVector(lf::mesh::utils::CodimMeshDataSet<bool> bd_flags, std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction,
                std::function<Eigen::Vector2d (const Eigen::Vector2d &)> body)
        : bd_flags(std::move(bd_flags)), traction_(std::move(traction)), body_f(std::move(body)) {}


        //TODO: implement the capability for there to be a traction BC and displacement BC on the same node
        //Return the value at bd_flags for edges, and if there is no body force, don't loop over cells
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
        Eigen::Vector<double, 18> Eval(const lf::mesh::Entity &entity);
    };
}

#endif //GETTINGSTARTED_PARAMETRIC_MATRIX_COMPUTATION_H

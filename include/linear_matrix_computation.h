//
// Created by Liam Curtis on 2024-05-01.
//
#ifndef GETTINGSTARTED_LINEAR_MATRIX_COMPUTATION_H
#define GETTINGSTARTED_LINEAR_MATRIX_COMPUTATION_H

#include <lf/uscalfe/uscalfe.h>
#include <Eigen/Core>

namespace LinearMatrixComputation {

    class LinearFEElementMatrix {

    private:
        //Young's modulus, nu, and D are parameters we require for the computation of the stiffness matrix
        //D is either the plane-strain matrix or plane-stress matrix depending on the mesh
        const double youngmodulus {};
        const double nu {};
        Eigen::Matrix<double, 3, 3> D;

    public:
        //The default constructor is not allowed, as the parameters (Young's modulus, and nu) are required
        //for the computation of the stiffness matrix
        LinearFEElementMatrix() = delete;

        //Constructor I intend to use
        LinearFEElementMatrix(double young, double nu, bool planeStrain) : youngmodulus {young}, nu {nu}
        {
            if (planeStrain) {
                 //D is our plane strain matrix, which we initialize here
                LF_ASSERT_MSG(std::abs(nu - 0.5) >= (0.0001), "Value of nu (0.5) will lead to a divide by zero");
                D << (1 - nu), nu, 0.0,
                    nu, (1- nu), 0.0,
                    0.0, 0.0, (1 - 2 * nu) / 2.0;
                D *= youngmodulus / ((1 + nu) * (1 - 2 * nu));
            }
            else
                //Initialize D as a plane stress matrix
                LF_ASSERT_MSG((std::abs(nu - 1.0) >= (0.0001)), "Value of nu (1.0) will lead to a divide by zero");
                D << 1.0, nu, 0.0,
                    nu, 1.0, 0.0,
                    0.0, 0.0, (1 - nu) / 2.0;
                D *= youngmodulus / (1 - nu * nu); {

            }
        }

        //The isActive and Eval function required to send to AssembleMatrixLocally later on
        bool isActive (const lf::mesh::Entity&) {return true;} //All cells will be integrated over
        Eigen::Matrix<double, 8, 8> Eval(const lf::mesh::Entity &cell);

        //This method will be used for post-processing, taking in the displacement vector, and return matrices
        //for the stress and strains at the element's respective nodes
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> stressStrain(const lf::mesh::Entity &cell, Eigen::VectorXd &disp, const lf::assemble::DofHandler &dofh);

        // TODO: Implement this method
        // The purpose of this method is to calculate the energy of a cell once we have obtained the displacement vector
        // Note this method will use stressStrain as part of its calculation
        double energyCalc(const lf::mesh::Entity &cell, Eigen::VectorXd &disp, const lf::assemble::DofHandler &dofh);
    };



    class LinearFELoadVector {
    private:
        //Body force, boundary flags, and a traction function will be required for the computation of the load vector
        lf::mesh::utils::CodimMeshDataSet<bool> bd_flags;
        std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction_;
        std::function<Eigen::Vector2d (const Eigen::Vector2d &)> body_f;

    public:
        //No default constructor, I want the bd_flags at least to be present when initialized
        LinearFELoadVector() = delete;

        //I allow a constructor for the case where there is no body force or traction, in this case the implementation
        //will just return a zero vector, which it should already be set to
        explicit LinearFELoadVector(lf::mesh::utils::CodimMeshDataSet<bool> bd_flags)
        : bd_flags(std::move(bd_flags)), traction_{}, body_f{} {}

        //The case where there is traction but no body force
        LinearFELoadVector(lf::mesh::utils::CodimMeshDataSet<bool> bd_flags, std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction)
        : bd_flags(std::move(bd_flags)), traction_(std::move(traction)), body_f{} {}

        //Complete constructor
        LinearFELoadVector(lf::mesh::utils::CodimMeshDataSet<bool> bd_flags, std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction,
                           std::function<Eigen::Vector2d (const Eigen::Vector2d &)> body)
        : bd_flags(std::move(bd_flags)), traction_(std::move(traction)), body_f(std::move(body)) {}

        //Additionally, I have not included the possibility that we compute the traction in just one direction instead of both
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
        Eigen::Vector<double, 8> Eval(const lf::mesh::Entity &entity);
    };
}


#endif //GETTINGSTARTED_LINEAR_MATRIX_COMPUTATION_H


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
        const double youngmodulus{};
        const double nu{};

    public:

        //The default constructor is not allowed, as the parameters (Young's modulus, and nu) are required
        //for the computation of the stiffness matrix
        ParametricFEElementMatrix() = delete;

        //Constructor I intend to use
        ParametricFEElementMatrix(double young, double nu) : youngmodulus {young}, nu {nu}
        {}

        //The isActive and Eval function required to send to AssembleMatrixLocally later on
        bool isActive (const lf::mesh::Entity&) {return true;} //All cells will be integrated over
        Eigen::Matrix<double, 8, 8> Eval(const lf::mesh::Entity &cell);
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

        //The case where there is traction but no body force
        ParametricFELoadVector(lf::mesh::utils::CodimMeshDataSet<bool> bd_flags, std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction)
        : bd_flags(std::move(bd_flags)), traction_(std::move(traction)), body_f{} {}

        //Complete constructor
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
        Eigen::Vector<double, 8> Eval(const lf::mesh::Entity &entity);
    };
}

#endif //GETTINGSTARTED_PARAMETRIC_MATRIX_COMPUTATION_H

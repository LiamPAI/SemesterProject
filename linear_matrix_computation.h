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
        //Young's modulus and nu are parameters we require for the computation of the stiffness matrix
        const double youngmodulus {};
        const double nu {};

    public:
        //The default constructor is not allowed, as the parameters (Young's modulus, and nu) are required
        //for the computation of the stiffness matrix
        LinearFEElementMatrix() = delete;

        //Constructor I intend to use
        LinearFEElementMatrix(double young, double nu) : youngmodulus {young}, nu {nu}
        {}

        //The isActive and Eval function required to send to AssembleMatrixLocally later on
        bool isActive (const lf::mesh::Entity&) {return true;} //All cells will be integrated over
        Eigen::Matrix<double, 8, 8> Eval(const lf::mesh::Entity &cell);
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
        //will just return a zero vector
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


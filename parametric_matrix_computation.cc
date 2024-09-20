//
// Created by Liam Curtis on 2024-05-06.
//

# include "parametric_matrix_computation.h"

#include <lf/base/base.h>
#include <lf/geometry/geometry.h>
#include <lf/mesh/mesh.h>
#include <lf/uscalfe/uscalfe.h>
#include <Eigen/Core>

// TODO: test if LF_ASSERT wants true or false in order for the condition to be asserted

//The following code can take in parametrized or non-parametrized triangles and quadrangles to compute the element
//stiffness matrices and load vector on a given mesh
namespace ParametricMatrixComputation {

    Eigen::Matrix<double, 18, 18> ParametricFEElementMatrix::Eval(const lf::mesh::Entity &cell) {

        // Obtain reference element, geometry, vertices, and initialize our return matrix
        const lf::base::RefEl ref_el {cell.RefEl()};
        const lf::geometry::Geometry *geo_ptr {cell.Geometry()};
        Eigen::Matrix<double, 18, 18> elem_mat;
        elem_mat.setZero();

        // Check if the reference element is a triangle/quadrilateral
        // We are solving the integral of BT * D * B
        // B is the matrix of gradients, which is the matrix of strains (symmetric gradient * displacement)

        //Since the mesh is of order 2, the cells are of type Tria02() and QuadO2(), so they are already parametrized
        //We can use this to slightly alter the computation of element stiffness matrices, especially for triangles
        //The implementation for cells of type kQuad() is virtually the same as in the completely linear case
        //The implementation for cells of type kTria() is different since the result is no longer analytical
        switch(ref_el) {
            case lf::base::RefEl::kTria(): {

                lf::uscalfe::FeLagrangeO2Tria<double> element;
                //Make an order 4 quadrature rule, to keep the order the same with quads and triangles
                lf::quad::QuadRule qr = lf::quad::make_TriaQR_P6O4();

                //Jacobian Inverse Gramian will give a 2x12 matrix (2x2 for each of the quadrature points)
                auto JinvT = geo_ptr->JacobianInverseGramian(qr.Points());

                //Integration elements will give the 6 determinants of the Jacobian, one for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 6x12 matrix, for the gradients of the 6 basis functions at the 6 different points
                auto precomp = element.GradientsReferenceShapeFunctions(qr.Points());

                //Declare our matrix B, which we will reinitialize at every quadrature point
                //The 3 by 12 comes from the symmetric gradients and the 6 basis functions
                Eigen::Matrix<double, 3, 18> B;

                //basis will contain the gradients of the basis functions on the cell transformed to the reference element
                //There are 6 basis functions for an O2Tria, so basis is 2x6
                Eigen::Matrix<double, 2, 6> basis;

                //We will have to initialize B for every quadrature point
                for (int i = 0; i < qr.NumPoints(); ++i) {
                    const double w = qr.Weights()[i] * determinants[i];

                    //Initialize our basis functions at this quadrature point, transformed to reference element
                    basis << JinvT.block<2, 2>(0,2*i) * (precomp.block<6, 2>(0, 2*i)).transpose();

                    //Initialize our matrix B depending on the basis matrix, zeros are on the end since this is a triangle
                    B << basis(0,0), 0.0, basis(0,1), 0.0, basis(0,2), 0.0,
                    basis(0, 3), 0.0, basis(0, 4), 0.0, basis(0, 5), 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, basis(1,0), 0.0, basis(1,1), 0.0, basis(1,2),
                    0.0, basis(1,3), 0.0, basis(1,4), 0.0, basis(1,5),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    basis(1,0), basis(0,0), basis(1,1), basis(0,1), basis(1,2), basis(0,2),
                    basis(1,3), basis(0,3), basis(1,4), basis(0,4), basis(1,5), basis(0,5),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

                    //Add product to our element stiffness matrix
                    elem_mat += w * B.transpose() * D * B;
                }
                break;
            }

            case lf::base::RefEl::kQuad(): {

                //I choose to use a second order quadrangle, to be match the 2nd order parametrization of the shape
                lf::uscalfe::FeLagrangeO2Quad<double> element;
                lf::quad::QuadRule qr = lf::quad::make_QuadQR_P4O4();

                //Jacobian Inverse Gramian will give a 2x8 matrix (2x2 for each of the quadrature points)
                auto JinvT = geo_ptr->JacobianInverseGramian(qr.Points());

                //Integration elements will give the 4 determinants of the Jacobian, one for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 9x8 matrix, for the gradients of the 4 basis functions at the 4 different points
                auto precomp = element.GradientsReferenceShapeFunctions(qr.Points());

                //Declare our matrix B, which we will reinitialize at every quadrature point
                Eigen::Matrix<double, 3, 18> B;

                //basis will contain the gradients of the basis functions on the cell transformed to the reference element
                Eigen::Matrix<double, 2, 9> basis;

                //We will have to initialize B for every quadrature point
                for (int i = 0; i < qr.NumPoints(); ++i) {
                    const double w = qr.Weights()[i] * determinants[i];

                    //Initialize our basis functions at this quadrature point, transformed to reference element
                    basis << JinvT.block<2, 2>(0,2*i) * (precomp.block<9, 2>(0, 2*i)).transpose();

                    //Initialize our matrix B depending on the basis matrix
                    B << basis(0,0), 0.0, basis(0,1), 0.0, basis(0,2), 0.0,
                    basis(0,3), 0.0, basis(0,4), 0.0, basis(0,5), 0.0,
                    basis(0,6), 0.0, basis(0,7), 0.0, basis(0,8), 0.0,
                    0.0, basis(1,0), 0.0, basis(1,1), 0.0, basis(1,2),
                    0.0, basis(1,3), 0.0, basis(1,4), 0.0, basis(1,5),
                    0.0, basis(1,6), 0.0, basis(1,7), 0.0, basis(1,8),
                    basis(1,0), basis(0,0), basis(1,1), basis(0,1), basis(1,2), basis(0,2),
                    basis(1,3), basis(0,3), basis(1,4), basis(0,4), basis(1,5), basis(0,5),
                    basis(1,6), basis(0,6), basis(1,7), basis(0,7), basis(1,8), basis(0,8);

                    //Add product to our element stiffness matrix
                    elem_mat += w * B.transpose() * D * B;
                }
                break;
            }
            default: {
                LF_ASSERT_MSG(false, "Illegal cell type sent to ParametricFEElementMatrix");
            }
        }
        return elem_mat;
    }


    Eigen::Vector<double, 18> ParametricFELoadVector::Eval(const lf::mesh::Entity &entity) {

        const lf::base::RefEl ref_el {entity.RefEl()};
        const lf::geometry::Geometry *geo_ptr {entity.Geometry()};
        Eigen::Vector<double, 18> elem_vec;
        elem_vec.setZero();

        //Due to the implementation of isActive, we know that if the entity is a triangle or quadrilateral, that body_f
        //is implemented, so we can safely use the function

        //If the entity is an edge, since isActive allowed us to use this Eval, we assume traction_ is well implemented

        //If there were no traction in our problem, we expect bd_flags to always return false, hence Eval not to be used
        //with an invalid traction_ function

        switch(ref_el) {
            case lf::base::RefEl::kSegment(): {

                //Since this is a segment, we only need to evaluate the traction integral
                lf::uscalfe::FeLagrangeO2Segment<double> segment;
                lf::quad::QuadRuleCache qr_cache;
                const lf::quad::QuadRule& qr = qr_cache.Get(ref_el, 4);
                //Integration elements will give the determinants of the Jacobian, one for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 3xNumPoints matrix, for the values of the 2 basis functions at the different points
                auto precomp = segment.EvalReferenceShapeFunctions(qr.Points());

                //Loop over quadrature points
                for (int i = 0; i < qr.NumPoints(); i++) {
                    const double w = qr.Weights()[i] * determinants[i];

                    //Compute traction at that quadrature point
                    Eigen::Vector2d trac = traction_(geo_ptr->Global(qr.Points().col(i)));

                    //Add contributions to load vector using quadrature formula
                    elem_vec(0) += w * trac(0) * precomp(0, i);
                    elem_vec(1) += w * trac(1) * precomp(0, i);
                    elem_vec(2) += w * trac(0) * precomp(1, i);
                    elem_vec(3) += w * trac(1) * precomp(1, i);
                    elem_vec(4) += w * trac(0) * precomp(2, i);
                    elem_vec(5) += w * trac(1) * precomp(2, i);
                }
                break;
            }

            case lf::base::RefEl::kTria(): {
                //Must evaluate the body force integral
                //Declare quadrature rule and element type needed for basis function evaluations
                lf::uscalfe::FeLagrangeO2Tria<double> tria;
                lf::quad::QuadRule qr = lf::quad::make_TriaQR_P6O4();

                //Obtain determinant for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 6x6 matrix, for each of the 6 basis functions and 6 quadrature points
                //1 entire column per basis function
                auto precomp = tria.EvalReferenceShapeFunctions(qr.Points());

                //Loop over quadrature points
                for (int i= 0; i < qr.NumPoints(); i++) {
                    const double w = qr.Weights()[i] * determinants[i];

                    //Compute body force at this quadrature point
                    Eigen::Vector2d body = body_f(geo_ptr->Global(qr.Points().col(i)));

                    //Add contributions to load vector
                    elem_vec(0) += w * body(0) * precomp(0, i);
                    elem_vec(1) += w * body(1) * precomp(0, i);
                    elem_vec(2) += w * body(0) * precomp(1, i);
                    elem_vec(3) += w * body(1) * precomp(1, i);
                    elem_vec(4) += w * body(0) * precomp(2, i);
                    elem_vec(5) += w * body(1) * precomp(2, i);
                    elem_vec(6) += w * body(0) * precomp(3, i);
                    elem_vec(7) += w * body(1) * precomp(3, i);
                    elem_vec(8) += w * body(0) * precomp(4, i);
                    elem_vec(9) += w * body(1) * precomp(4, i);
                    elem_vec(10) += w * body(0) * precomp(5, i);
                    elem_vec(11) += w * body(1) * precomp(5, i);
                }
                break;
            }

            case lf::base::RefEl::kQuad(): {
                //Must evaluate the body force integral
                //Declare quadrature rule and element type needed for basis function evaluations
                lf::uscalfe::FeLagrangeO2Quad<double> quad;
                lf::quad::QuadRule qr = lf::quad::make_QuadQR_P4O4();

                //Obtain determinant for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 9x4 matrix, for each of the 4 basis functions and 4 quadrature points
                auto precomp = quad.EvalReferenceShapeFunctions(qr.Points());

                for (int i = 0; i < qr.NumPoints(); i++) {
                    const double w = qr.Weights()[i] * determinants[i];

                    //Compute body force at this quadrature point
                    Eigen::Vector2d body = body_f(geo_ptr->Global(qr.Points().col(i)));

                    //Add contributions to load vector
                    elem_vec(0) += w * body(0) * precomp(0, i);
                    elem_vec(1) += w * body(1) * precomp(0, i);
                    elem_vec(2) += w * body(0) * precomp(1, i);
                    elem_vec(3) += w * body(1) * precomp(1, i);
                    elem_vec(4) += w * body(0) * precomp(2, i);
                    elem_vec(5) += w * body(1) * precomp(2, i);
                    elem_vec(6) += w * body(0) * precomp(3, i);
                    elem_vec(7) += w * body(1) * precomp(3, i);
                    elem_vec(8) += w * body(0) * precomp(4, i);
                    elem_vec(9) += w * body(1) * precomp(4, i);
                    elem_vec(10) += w * body(0) * precomp(5, i);
                    elem_vec(11) += w * body(1) * precomp(5, i);
                    elem_vec(12) += w * body(0) * precomp(6, i);
                    elem_vec(13) += w * body(1) * precomp(6, i);
                    elem_vec(14) += w * body(0) * precomp(7, i);
                    elem_vec(15) += w * body(1) * precomp(7, i);
                    elem_vec(16) += w * body(0) * precomp(8, i);
                    elem_vec(17) += w * body(1) * precomp(8, i);
                }
                break;
            }
            default: {
                LF_ASSERT_MSG(false, "Illegal cell type sent to ParametricFELoadVector");
            }
        }
        return elem_vec;
    }


    //This function implements post-processing to allow for the calculation of stresses and strains at various
    //points on the mesh (in this case the quadrature points)
    //The strains and stress are made up each of 3 components (xx, yy, xy) for each quadrature point, they are stored
    //as column _vectors
    // TODO: check that this is correctly implemented
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ParametricFEElementMatrix::stressStrain
    (const lf::mesh::Entity &cell, Eigen::VectorXd &disp, const lf::assemble::DofHandler &dofh) {

        const lf::base::RefEl ref_el {cell.RefEl()};
        const lf::geometry::Geometry *geo_ptr {cell.Geometry()};

        switch (ref_el) {

            case lf::base::RefEl::kTria() : {

                //Declare the type of element, the quadrature rule, and the return matrices
                lf::uscalfe::FeLagrangeO2Tria<double> element;
                lf::quad::QuadRule qr = lf::quad::make_TriaQR_P6O4();

                //3x6, 3 for the strain vector and 6 for the 6 quadrature points
                Eigen::Matrix<double, 3, 6> stress;
                Eigen::Matrix<double, 3, 6> strain;
                stress.setZero();
                strain.setZero();

                //Initialize the vector of displacements for this particular cell
                Eigen::VectorXd localu(12);
                std::span<const lf::assemble::gdof_idx_t> indices (dofh.GlobalDofIndices(cell));
                localu[0] = disp(indices[0] * 2);
                localu[1] = disp(indices[0] * 2 + 1);
                localu[2] = disp(indices[1] * 2);
                localu[3] = disp(indices[1] * 2 + 1);
                localu[4] = disp(indices[2] * 2);
                localu[5] = disp(indices[2] * 2 + 1);
                localu[6] = disp(indices[3] * 2);
                localu[7] = disp(indices[3] * 2 + 1);
                localu[8] = disp(indices[4] * 2);
                localu[9] = disp(indices[4] * 2 + 1);
                localu[10] = disp(indices[5] * 2);
                localu[11] = disp(indices[5] * 2 + 1);

                //This will be a 6x12 matrix for the 6 basis functions and the 6 quadrature points
                auto precomp = element.GradientsReferenceShapeFunctions(qr.Points());

                //Jacobian Inverse Gramian will give a 2x12 matrix (2x2 for each of the quadrature points)
                auto JinvT = geo_ptr->JacobianInverseGramian(qr.Points());

                //Declare our matrix B, which we will reinitialize at every quadrature point
                Eigen::Matrix<double, 3, 12> B;

                //basis will contain the gradients of the basis functions on the cell transformed to the reference element
                Eigen::Matrix<double, 2, 6> basis;

                //Loop over all quadrature points
                for (int i = 0; i < qr.NumPoints(); ++i) {
                    //Initialize our basis functions at this quadrature point, transformed to reference element
                    basis << JinvT.block<2, 2>(0,2*i) * (precomp.block<6, 2>(0, 2*i)).transpose();

                    //Initialize our matrix B depending on the basis matrix
                    B << basis(0,0), 0.0, basis(0,1), 0.0, basis(0,2), 0.0,
                            basis(0, 3), 0.0, basis(0, 4), 0.0, basis(0, 5), 0.0,
                            0.0, basis(1,0), 0.0, basis(1,1), 0.0, basis(1,2),
                            0.0, basis(1,3), 0.0, basis(1,4), 0.0, basis(1,5),
                            basis(1,0), basis(0,0), basis(1,1), basis(0,1), basis(1,2), basis(0,2),
                            basis(1,3), basis(0,3), basis(1,4), basis(0,4), basis(1,5), basis(0,5);

                    strain.block<3, 1>(0, i) << B * localu;
                    stress.block<3, 1>(0, i) << D * strain.block<3, 1>(0, i);
                }

                return std::tuple(stress, strain, geo_ptr->Global(qr.Points()));
            }

            case lf::base::RefEl::kQuad() : {

                //Declare the type of element, the quadrature rule, and the return matrices
                lf::uscalfe::FeLagrangeO2Quad<double> element;
                lf::quad::QuadRule qr = lf::quad::make_QuadQR_P4O4();

                //3x4, 3 for the dimension of the vector, 4 for each of the quadrature points
                Eigen::Matrix<double, 3, 4> stress;
                Eigen::Matrix<double, 3, 4> strain;
                stress.setZero();
                strain.setZero();

                //Initialize the vector of displacements for this particular cell
                Eigen::VectorXd localu(18);
                std::span<const lf::assemble::gdof_idx_t> indices (dofh.GlobalDofIndices(cell));
                localu[0] = disp(indices[0] * 2);
                localu[1] = disp(indices[0] * 2 + 1);
                localu[2] = disp(indices[1] * 2);
                localu[3] = disp(indices[1] * 2 + 1);
                localu[4] = disp(indices[2] * 2);
                localu[5] = disp(indices[2] * 2 + 1);
                localu[6] = disp(indices[3] * 2);
                localu[7] = disp(indices[3] * 2 + 1);
                localu[8] = disp(indices[4] * 2);
                localu[9] = disp(indices[4] * 2 + 1);
                localu[10] = disp(indices[5] * 2);
                localu[11] = disp(indices[5] * 2 + 1);
                localu[12] = disp(indices[6] * 2);
                localu[13] = disp(indices[6] * 2 + 1);
                localu[14] = disp(indices[7] * 2);
                localu[15] = disp(indices[7] * 2 + 1);
                localu[16] = disp(indices[8] * 2);
                localu[17] = disp(indices[8] * 2 + 1);

                //This will be a 9x8 matrix, for the 9 basis functions and the 4 quadrature points
                auto precomp = element.GradientsReferenceShapeFunctions(qr.Points());

                //Jacobian Inverse Gramian will give a 2x8 matrix (2x2 for each of the quadrature points)
                auto JinvT = geo_ptr->JacobianInverseGramian(qr.Points());

                //Declare our matrix B, which we will reinitialize at every quadrature point
                Eigen::Matrix<double, 3, 18> B;

                //basis will contain the gradients of the basis functions on the cell transformed to the reference element
                Eigen::Matrix<double, 2, 9> basis;

                //Loop over all quadrature points
                for (int i = 0; i < qr.NumPoints(); ++i) {
                    //Initialize our basis functions at this quadrature point, transformed to reference element
                    basis << JinvT.block<2, 2>(0,2*i) * (precomp.block<9, 2>(0, 2*i)).transpose();

                    //Initialize our matrix B depending on the basis matrix
                    B << basis(0,0), 0.0, basis(0,1), 0.0, basis(0,2), 0.0,
                    basis(0,3), 0.0, basis(0,4), 0.0, basis(0,5), 0.0,
                    basis(0,6), 0.0, basis(0,7), 0.0, basis(0,8), 0.0,
                    0.0, basis(1,0), 0.0, basis(1,1), 0.0, basis(1,2),
                    0.0, basis(1,3), 0.0, basis(1,4), 0.0, basis(1,5),
                    0.0, basis(1,6), 0.0, basis(1,7), 0.0, basis(1,8),
                    basis(1,0), basis(0,0), basis(1,1), basis(0,1), basis(1,2), basis(0,2),
                    basis(1,3), basis(0,3), basis(1,4), basis(0,4), basis(1,5), basis(0,5),
                    basis(1,6), basis(0,6), basis(1,7), basis(0,7), basis(1,8), basis(0,8);

                    strain.block<3, 1>(0, i) << B * localu;
                    stress.block<3, 1>(0, i) << D * strain.block<3, 1>(0, i);
                }

                return std::tuple(stress, strain, geo_ptr->Global(qr.Points()));
            }

            default: {
                LF_ASSERT_MSG(false, "Illegal cell type sent to stressStrain");
            }
        }
    }

    // TODO: Test this method
    // This method takes in a cell, the overall displacement vector for the mesh, and the Dofhandler and outputs the
    // energy functional over that cell. Note that due to the parameters of the research problem, I do not include the
    // capability of external forces or non-zero traction BCs, though this could be implemented using functionality
    // found in other methods
    // This function is identical to the one in linear_matrix_computation.cc, but it is implied the cell sent to this
    // is of order 2
    double ParametricFEElementMatrix::energyCalc(const lf::mesh::Entity &cell, Eigen::VectorXd &disp,
                                             const lf::assemble::DofHandler &dofh) {

        // Obtain the cell type and geometry pointed
        const lf::base::RefEl ref_el {cell.RefEl()};
        const lf::geometry::Geometry *geo_ptr {cell.Geometry()};

        // Call the stressStrain functions to obtain the stresses and strains for this cell
        auto stressAndStrains = stressStrain(cell, disp, dofh);
        auto stresses = std::get<0>(stressAndStrains);
        auto strains = std::get<1>(stressAndStrains);

        // Initialize the energy for this cell, which will later be summed up for quad rule
        double energy = 0.0;

        switch (ref_el) {

            case lf::base::RefEl::kTria() : {
                // Declare the quadrature rule for the element, note that this is the same
                // quadrature rule as in the method stressStrain, which is essential, for tria type
                lf::quad::QuadRule qr = lf::quad::make_TriaQR_P6O4();

                // Obtain the determinants at the quadrature points for easy summation
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                for (int i = 0; i < qr.NumPoints(); ++i) {
                    // Sum the energy for this quadrature point
                    energy += 0.5 * qr.Weights()[i] * determinants[i] * stresses.col(i).transpose() * strains.col(i);
                }
            }

            case lf::base::RefEl::kQuad() : {
                // Declare the quadrature rule for the element, note that this is the same
                // quadrature rule as in the method stressStrain, which is essential, for quad type
                lf::quad::QuadRule qr = lf::quad::make_QuadQR_P4O4();

                // Obtain the determinants at the quadrature points for easy summation
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                for (int i = 0; i < qr.NumPoints(); ++i) {
                    // Sum the energy for this quadrature point
                    energy += 0.5 * qr.Weights()[i] * determinants[i] * stresses.col(i).transpose() * strains.col(i);
                }
            }

            default : {
                LF_ASSERT_MSG(true, "Illegal cell type sent to stressStrain");
            }

        }
        return energy;
    }

}

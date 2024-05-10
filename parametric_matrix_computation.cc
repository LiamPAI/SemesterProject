//
// Created by Liam Curtis on 2024-05-06.
//

# include "parametric_matrix_computation.h"

#include <lf/base/base.h>
#include <lf/geometry/geometry.h>
#include <lf/mesh/mesh.h>
#include <lf/uscalfe/uscalfe.h>
#include <Eigen/Core>

//The following code can take in parametrized or non-parametrized triangles and quadrangles to compute the element
//stiffness matrices and load vector on a given mesh
namespace ParametricMatrixComputation {

    Eigen::Matrix<double, 8, 8> ParametricFEElementMatrix::Eval(const lf::mesh::Entity &cell) {

        // Obtain reference element, geometry, vertices, and initialize our return matrix
        const lf::base::RefEl ref_el {cell.RefEl()};
        const lf::geometry::Geometry *geo_ptr {cell.Geometry()};
        Eigen::Matrix<double, 8, 8> elem_mat;
        elem_mat.setZero();

        // Check if the reference element is a triangle/quadrilateral
        // We are solving the integral of BT * D * B
        // B is the matrix of gradients, which is the matrix of strains (symmetric gradient * displacement)

        // D is our plane strain matrix, which we initialize here
//        LF_ASSERT_MSG(std::abs(nu - 0.5) >= (0.0001), "Value of nu (0.5) will lead to a divide by zero");
//        Eigen::Matrix<double, 3, 3> D;
//        D << (1 - nu), nu, 0.0,
//            nu, (1- nu), 0.0,
//            0.0, 0.0, (1 - 2 * nu) / 2.0;
//        D *= youngmodulus / ((1 + nu) * (1 - 2 * nu));

        //Initialize D as a plane stress matrix, likely not needed
        LF_ASSERT_MSG((std::abs(nu - 1.0) >= (0.0001)), "Value of nu (1.0) will lead to a divide by zero");
        Eigen::Matrix<double, 3, 3> D;
        D << 1.0, nu, 0.0,
                nu, 1.0, 0.0,
                0.0, 0.0, (1 - nu) / 2.0;
        D *= youngmodulus / (1 - nu * nu);

        //Since the mesh is of order 2, the cells are of type Tria02() and QuadO2(), so they are already parametrized
        //We can use this to slightly alter the computation of element stiffness matrices, especially for triangles
        //The implementation for cells of type kQuad() is virtually the same as in the completely linear case
        //The implementation for cells of type kTria() is different since the result is no longer analytical
        switch(ref_el) {
            case lf::base::RefEl::kTria(): {

                lf::uscalfe::FeLagrangeO1Tria<double> element;
                //Make an order 4 quadrature rule, to keep the order the same with quads and triangles
                lf::quad::QuadRule qr = lf::quad::make_TriaQR_P6O4();

                //Jacobian Inverse Gramian will give a 2x12 matrix (2x2 for each of the quadrature points)
                auto JinvT = geo_ptr->JacobianInverseGramian(qr.Points());

                //Integration elements will give the 6 determinants of the Jacobian, one for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 3x12 matrix, for the gradients of the 3 basis functions at the 6 different points
                auto precomp = element.GradientsReferenceShapeFunctions(qr.Points());

                //Declare our matrix B, which we will reinitialize at every quadrature point
                Eigen::Matrix<double, 3, 8> B;

                //basis will contain the gradients of the basis functions on the cell transformed to the reference element
                Eigen::Matrix<double, 2, 3> basis;

                //We will have to initialize B for every quadrature point
                for (int i = 0; i < qr.NumPoints(); ++i) {
                    const double w = qr.Weights()[i] * determinants[i];

                    //Initialize our basis functions at this quadrature point, transformed to reference element
                    basis << JinvT.block<2, 2>(0,2*i) * (precomp.block<3, 2>(0, 2*i)).transpose();

                    //Initialize our matrix B depending on the basis matrix, zeros are on the end since this is a triangle
                    B << basis(0,0), 0.0, basis(0,1), 0.0, basis(0,2), 0.0, 0.0, 0.0,
                            0.0, basis(1,0), 0.0, basis(1,1), 0.0, basis(1,2), 0.0, 0.0,
                            basis(1,0), basis(0,0), basis(1,1), basis(0,1), basis(1,2), basis(0,2), 0.0, 0.0;

                    //Add product to our element stiffness matrix
                    elem_mat += w * B.transpose() * D * B;
                }
                break;
            }

            case lf::base::RefEl::kQuad(): {

                //I choose to still use only 4 nodal functions on a quadrangle, and quadrature rule of order 4
                lf::uscalfe::FeLagrangeO1Quad<double> element;
                lf::quad::QuadRule qr = lf::quad::make_QuadQR_P4O4();

                //Jacobian Inverse Gramian will give a 2x8 matrix (2x2 for each of the quadrature points)
                auto JinvT = geo_ptr->JacobianInverseGramian(qr.Points());

                //Integration elements will give the 4 determinants of the Jacobian, one for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 4x8 matrix, for the gradients of the 4 basis functions at the 4 different points
                auto precomp = element.GradientsReferenceShapeFunctions(qr.Points());

                //Declare our matrix B, which we will reinitialize at every quadrature point
                Eigen::Matrix<double, 3, 8> B;

                //basis will contain the gradients of the basis functions on the cell transformed to the reference element
                Eigen::Matrix<double, 2, 4> basis;

                //We will have to initialize B for every quadrature point
                for (int i = 0; i < qr.NumPoints(); ++i) {
                    const double w = qr.Weights()[i] * determinants[i];

                    //Initialize our basis functions at this quadrature point, transformed to reference element
                    basis << JinvT.block<2, 2>(0,2*i) * (precomp.block<4, 2>(0, 2*i)).transpose();

                    //Initialize our matrix B depending on the basis matrix
                    B << basis(0,0), 0.0, basis(0,1), 0.0, basis(0,2), 0.0, basis(0,3), 0.0,
                            0.0, basis(1,0), 0.0, basis(1,1), 0.0, basis(1,2), 0.0, basis(1,3),
                            basis(1,0), basis(0,0), basis(1,1), basis(0,1), basis(1,2), basis(0,2), basis(1,3), basis(0,3);

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


    Eigen::Vector<double, 8> ParametricFELoadVector::Eval(const lf::mesh::Entity &entity) {

        const lf::base::RefEl ref_el {entity.RefEl()};
        const lf::geometry::Geometry *geo_ptr {entity.Geometry()};
        Eigen::Vector<double, 8> elem_vec;
        elem_vec.setZero();

        //Due to the implementation of isActive, we know that if the entity is a triangle or quadrilateral, that body_f
        //is implemented, so we can safely use the function

        //If the entity is an edge, since isActive allowed us to use this Eval, we assume traction_ is well implemented

        //If there were no traction in our problem, we expect bd_flags to always return false, hence Eval not to be used
        //with an invalid traction_ function

        switch(ref_el) {
            case lf::base::RefEl::kSegment(): {

                //Since this is a segment, we only need to evaluate the traction integral
                lf::uscalfe::FeLagrangeO1Segment<double> segment;
                lf::quad::QuadRuleCache qr_cache;
                const lf::quad::QuadRule& qr = qr_cache.Get(ref_el, 4);

                //Integration elements will give the determinants of the Jacobian, one for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 2xNumPoints matrix, for the values of the 2 basis functions at the different points
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
                 }
                break;
            }

            case lf::base::RefEl::kTria(): {
                //Must evaluate the body force integral
                //Declare quadrature rule and element type needed for basis function evaluations
                lf::uscalfe::FeLagrangeO1Tria<double> tria;
                lf::quad::QuadRule qr = lf::quad::make_TriaQR_P6O4();

                //Obtain determinant for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 3x6 matrix, for each of the 3 basis functions and 6 quadrature points
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
                }
                break;
            }

            case lf::base::RefEl::kQuad(): {
                //Must evaluate the body force integral
                //Declare quadrature rule and element type needed for basis function evaluations
                lf::uscalfe::FeLagrangeO1Quad<double> quad;
                lf::quad::QuadRule qr = lf::quad::make_QuadQR_P4O4();

                //Obtain determinant for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 4x4 matrix, for each of the 4 basis functions and 4 quadrature points
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
                }
                break;
            }
            default: {
                LF_ASSERT_MSG(false, "Illegal cell type sent to ParametricFELoadVector");
            }
        }
        return elem_vec;
    }

}

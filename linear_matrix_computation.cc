//
// Created by Liam Curtis on 2024-05-01.
//
#include "linear_matrix_computation.h"

#include <lf/base/base.h>
#include <lf/geometry/geometry.h>
#include <lf/mesh/mesh.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/QR>

// I assume non-parametrized triangular cells, as well as general quadrilateral elements for the computation
// of the stiffness matrix (i.e. no non-linear parametrized boundaries/cell types)
namespace LinearMatrixComputation {

    Eigen::Matrix<double, 8, 8> LinearFEElementMatrix::Eval(const lf::mesh::Entity &cell) {

        // Obtain reference element, geometry, vertices, and initialize our return matrix
        const lf::base::RefEl ref_el {cell.RefEl()};
        const lf::geometry::Geometry *geo_ptr {cell.Geometry()};
        auto vertices {geo_ptr->Global(ref_el.NodeCoords())};
        Eigen::Matrix<double, 8, 8> elem_mat;
        elem_mat.setZero();

        // Check if the reference element is a triangle/quadrilateral
        // We are solving the integral of BT * D * B
        // B is the matrix of gradients, which is the matrix of strains (symmetric gradient * displacement)

        switch (ref_el) {
            case lf::base::RefEl::kTria(): {
                const double area = lf::geometry::Volume(*(cell.Geometry()));

                // Implementation of gradbary coordinates on a triangle
                Eigen::Matrix<double, 3, 3> X;
                Eigen::Matrix<double, 2, 3> grads;
                Eigen::Matrix<double, 3, 8> B;

                X.block<3, 1> (0, 0) = Eigen::Vector3d::Ones();
                X.block<3, 2> (0, 1) = vertices.transpose();
                grads = (X.completeOrthogonalDecomposition().pseudoInverse()).block<2, 3>(1, 0);

                // B is our matrix of gradients for the calculation of the stiffness matrix
                // It is usually size 3x6 for triangles, although here it is 3x8 to account for return type of 8x8

                B << grads(0, 0), 0.0, grads(0, 1), 0.0, grads(0, 2), 0.0, 0.0, 0.0,
                    0.0, grads(1, 0), 0.0, grads(1, 1), 0.0, grads(1, 2), 0.0, 0.0,
                    grads(1, 0), grads(0, 0), grads(1, 1), grads(0, 1), grads(1, 2), grads(0, 2), 0.0, 0.0;

                // Calculate our element matrix
                elem_mat += area * B.transpose() * D * B;
                break;
            }

            case lf::base::RefEl::kQuad(): {

                lf::uscalfe::FeLagrangeO1Quad<double> element;
                lf::quad::QuadRule qr = lf::quad::make_QuadQR_P4O4();

                //Jacobian Inverse Gramian will give a 2x8 matrix (2x2 for each of the quadrature points)
                auto JinvT = geo_ptr->JacobianInverseGramian(qr.Points());

                //Integration elements will give the 4 determinants of the Jacobian, one for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This will give a 4x8 matrix, for the gradients of 4 basis functions are 4 different points
                auto precomp = element.GradientsReferenceShapeFunctions(qr.Points());

                //Declare our matrix B, which we will reinitialize at every quadrature point
                Eigen::Matrix<double, 3, 8> B;

                //basis will contain the gradients of the basis functions on the cell transformed to the reference element
                Eigen::Matrix<double, 2, 4> basis;

                //We will have to initialize B for every quadrature point
                for (int i = 0; i < qr.NumPoints(); ++i) {
                    const double w = qr.Weights()[i] * determinants[i];

                    //Initialize our basis function derivatives at this quadrature point, transformed to reference element
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
                LF_ASSERT_MSG(false, "Illegal cell type sent to LinearFEElementMatrix");
            }
        }
        return elem_mat;
    }



    Eigen::Vector<double, 8> LinearFELoadVector::Eval(const lf::mesh::Entity &entity) {

        const lf::base::RefEl ref_el {entity.RefEl()};
        const lf::geometry::Geometry *geo_ptr {entity.Geometry()};
        //auto vertices {geo_ptr->Global(ref_el.NodeCoords())};
        Eigen::Vector<double, 8> elem_vec;
        elem_vec.setZero();

        //Due to the implementation of isActive, we know that if the entity is a triangle or quadrilateral, that body_f
        //is implemented, so we can safely use the function

        //If the entity is an edge, since isActive allowed us to use this Eval, we assume traction_ is well implemented

        //If there were no traction in our problem, we expect bd_flags to always return false, hence Eval not to be used
        //with an invalid traction_ function

        switch(ref_el) {
            //For a segment, I only need to evaluate the traction integral, where I approximate the traction function numerically
            case lf::base::RefEl::kSegment(): {

                auto nodes_of_edge = entity.SubEntities(1);
                auto n0_xy = nodes_of_edge[0]->Geometry()->Global(nodes_of_edge[0]->RefEl().NodeCoords());
                auto n1_xy = nodes_of_edge[1]->Geometry()->Global(nodes_of_edge[1]->RefEl().NodeCoords());
                double edge_length = std::sqrt((n0_xy(0) - n1_xy(0)) * (n0_xy(0) - n1_xy(0)) +
                        (n0_xy(1) - n1_xy(1)) * (n0_xy(1) - n1_xy(1)));

                Eigen::Vector2d t0_xy = traction_(n0_xy);
                Eigen::Vector2d t1_xy = traction_(n1_xy);

                elem_vec(0) += (edge_length / 6) * (2 * t0_xy(0) + t1_xy(0));
                elem_vec(1) += (edge_length / 6) * (2 * t0_xy(1) + t1_xy(1));
                elem_vec(2) += (edge_length / 6) * (t0_xy(0) + 2 * t1_xy(0));
                elem_vec(3) += (edge_length / 6) * (t0_xy(1) + 2 * t1_xy(1));
                break;
            }

            //For a triangle, I need to evaluate the body force integral, which I approximate by a linear function in the fe_space
            //with values equal to the function at the nodes of the cell
            case lf::base::RefEl::kTria(): {

                const double area = lf::geometry::Volume(*(entity.Geometry()));
                auto nodes_of_tri = entity.SubEntities(2);
                auto n0_xy = nodes_of_tri[0]->Geometry()->Global(nodes_of_tri[0]->RefEl().NodeCoords());
                auto n1_xy = nodes_of_tri[1]->Geometry()->Global(nodes_of_tri[1]->RefEl().NodeCoords());
                auto n2_xy = nodes_of_tri[2]->Geometry()->Global(nodes_of_tri[2]->RefEl().NodeCoords());

                Eigen::Vector2d b0_xy = body_f(n0_xy);
                Eigen::Vector2d b1_xy = body_f(n1_xy);
                Eigen::Vector2d b2_xy = body_f(n2_xy);

                elem_vec(0) += (area / 12) * (2 * b0_xy(0) + b1_xy(0) + b2_xy(0));
                elem_vec(1) += (area / 12) * (2 * b0_xy(1) + b1_xy(1) + b2_xy(1));
                elem_vec(2) += (area / 12) * (b0_xy(0) + 2 * b1_xy(0) + b2_xy(0));
                elem_vec(3) += (area / 12) * (b0_xy(1) + 2 * b1_xy(1) + b2_xy(1));
                elem_vec(4) += (area / 12) * (b0_xy(0) + b1_xy(0) + 2 * b2_xy(0));
                elem_vec(5) += (area / 12) * (b0_xy(1) + b1_xy(1) + 2 * b2_xy(1));
                break;
            }

            //For a quadrilateral, I use the same method where I approximate the body force using the values at the nodes
            case lf::base::RefEl::kQuad(): {

                lf::uscalfe::FeLagrangeO1Quad<double> element;
                //lf::quad::QuadRule qr = lf::quad::make_QuadQR_P4O2();
                lf::quad::QuadRule qr = lf::quad::make_QuadQR_P4O4();

                //Integration elements will give the 4 determinants of the Jacobian, one for each quadrature point
                auto determinants = geo_ptr->IntegrationElement(qr.Points());

                //This vector will hold the values of the basis functions at the quadrature points in the loop
                //It should have shape 4x4 (for 4 shape functions and 4 points to evaluate them)
                auto refVals = element.EvalReferenceShapeFunctions(qr.Points());

                //This vector will be reinitialized at every quadrature point, and is used to "reshape" refVals
                Eigen::Vector<double, 8> basis;
                auto global_coords = geo_ptr->Global(qr.Points());

                for (int i = 0; i < qr.NumPoints(); ++i) {

                    //Calculate the value of the body force at this quadrature point, using global coordinates
                    Eigen::Vector2d vec = body_f(global_coords.block<2, 1>(0,i));

                    //Initialize the value of the basis vector times the body force, staggered
                    basis << vec(0) * refVals(i, 0), vec(1) * refVals(i, 0),
                     vec(0) * refVals(i, 1), vec(1) * refVals(i, 1),
                    vec(0) * refVals(i, 2), vec(1) * refVals(i, 2),
                    vec(0) * refVals(i, 3), vec(1) * refVals(i, 3);

                    //Add values to elem_vec for this quadrature point
                    elem_vec += qr.Weights()[i] * determinants[i] * basis;
                }
                break;
            }
            default: {
                LF_ASSERT_MSG(false, "Illegal cell type sent to LinearFELoadVector");
            }
        }
        return elem_vec;
    }

    // This function implements post-processing to allow for the calculation of stresses and strains at various
    // points on the mesh (in this case the quadrature points)
    // The strains and stress are made up each of 3 components (xx, yy, xy) for each quadrature points, they are stored
    // as column _vectors
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> LinearFEElementMatrix::stressStrain
    (const lf::mesh::Entity &cell, Eigen::VectorXd &disp, const lf::assemble::DofHandler &dofh) {

        const lf::base::RefEl ref_el {cell.RefEl()};
        const lf::geometry::Geometry *geo_ptr {cell.Geometry()};

        switch (ref_el) {

            case lf::base::RefEl::kTria() : {

                //Declare the type of element, the quadrature rule, and the return matrices
                lf::uscalfe::FeLagrangeO1Tria<double> element;
                lf::quad::QuadRule qr = lf::quad::make_TriaQR_P6O4();
                Eigen::Matrix<double, 3, 6> stress;
                Eigen::Matrix<double, 3, 6> strain;
                stress.setZero();
                strain.setZero();

                //Initialize the vector of displacements for this particular cell
                Eigen::VectorXd localu(6);
                std::span<const lf::assemble::gdof_idx_t> indices (dofh.GlobalDofIndices(cell));
                localu[0] = disp(indices[0] * 2);
                localu[1] = disp(indices[0] * 2 + 1);
                localu[2] = disp(indices[1] * 2);
                localu[3] = disp(indices[1] * 2 + 1);
                localu[4] = disp(indices[2] * 2);
                localu[5] = disp(indices[2] * 2 + 1);

                //This will be a 3x12 matrix for the 3 basis functions and the 6 quadrature points
                auto precomp = element.GradientsReferenceShapeFunctions(qr.Points());

                //Jacobian Inverse Gramian will give a 2x12 matrix (2x2 for each of the quadrature points)
                auto JinvT = geo_ptr->JacobianInverseGramian(qr.Points());

                //Declare our matrix B, which we will reinitialize at every quadrature point
                Eigen::Matrix<double, 3, 6> B;

                //basis will contain the gradients of the basis functions on the cell transformed to the reference element
                Eigen::Matrix<double, 2, 3> basis;

                //Loop over all quadrature points
                for (int i = 0; i < qr.NumPoints(); ++i) {
                    //Initialize our basis functions at this quadrature point, transformed to reference element
                    basis << JinvT.block<2, 2>(0,2*i) * (precomp.block<3, 2>(0, 2*i)).transpose();

                    //Initialize our matrix B depending on the basis matrix
                    B << basis(0,0), 0.0, basis(0,1), 0.0, basis(0,2), 0.0,
                            0.0, basis(1,0), 0.0, basis(1,1), 0.0, basis(1,2),
                            basis(1,0), basis(0,0), basis(1,1), basis(0,1), basis(1,2), basis(0,2);

                    strain.block<3, 1>(0, i) << B * localu;
                    stress.block<3, 1>(0, i) << D * strain.block<3, 1>(0, i);
                }

                return std::tuple(stress, strain, geo_ptr->Global(qr.Points()));
            }

            case lf::base::RefEl::kQuad() : {

                //Declare the type of element, the quadrature rule, and the return matrices
                lf::uscalfe::FeLagrangeO1Quad<double> element;
                lf::quad::QuadRule qr = lf::quad::make_QuadQR_P4O4();
                Eigen::Matrix<double, 3, 4> stress;
                Eigen::Matrix<double, 3, 4> strain;
                stress.setZero();
                strain.setZero();

                //Initialize the vector of displacements for this particular cell
                Eigen::VectorXd localu(8);
                std::span<const lf::assemble::gdof_idx_t> indices (dofh.GlobalDofIndices(cell));
                localu[0] = disp(indices[0] * 2);
                localu[1] = disp(indices[0] * 2 + 1);
                localu[2] = disp(indices[1] * 2);
                localu[3] = disp(indices[1] * 2 + 1);
                localu[4] = disp(indices[2] * 2);
                localu[5] = disp(indices[2] * 2 + 1);
                localu[6] = disp(indices[3] * 2);
                localu[7] = disp(indices[3] * 2 + 1);

                //This will be a 4x8 matrix, for the 4 basis functions and the 4 quadrature points
                auto precomp = element.GradientsReferenceShapeFunctions(qr.Points());

                //Jacobian Inverse Gramian will give a 2x8 matrix (2x2 for each of the quadrature points)
                auto JinvT = geo_ptr->JacobianInverseGramian(qr.Points());

                //Declare our matrix B, which we will reinitialize at every quadrature point
                Eigen::Matrix<double, 3, 8> B;

                //basis will contain the gradients of the basis functions on the cell transformed to the reference element
                Eigen::Matrix<double, 2, 4> basis;

                //Loop over all quadrature points
                for (int i = 0; i < qr.NumPoints(); ++i) {
                    //Initialize our basis functions at this quadrature point, transformed to reference element
                    basis << JinvT.block<2, 2>(0,2*i) * (precomp.block<4, 2>(0, 2*i)).transpose();

                    //Initialize our matrix B depending on the basis matrix
                    B << basis(0,0), 0.0, basis(0,1), 0.0, basis(0,2), 0.0, basis(0,3), 0.0,
                            0.0, basis(1,0), 0.0, basis(1,1), 0.0, basis(1,2), 0.0, basis(1,3),
                            basis(1,0), basis(0,0), basis(1,1), basis(0,1), basis(1,2), basis(0,2), basis(1,3), basis(0,3);

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
    // capability of external forces or non-zero traction BCs
    // This function is identical to the one in parametric_matrix_computation.cc, but it is implied that the cell
    // sent to this is of order 1
    double LinearFEElementMatrix::energyCalc(const lf::mesh::Entity &cell, Eigen::VectorXd &disp,
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
                // Declare the element type and quadrature rule for the element, note that this is the same
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
                LF_ASSERT_MSG(false, "Illegal cell type sent to stressStrain");
            }

        }
        return energy;
    }

}

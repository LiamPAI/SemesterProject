//
// Created by Liam Curtis on 2024-05-02.
//
#ifndef GETTINGSTARTED_LINEAR_ELASTICITY_ASSEMBLER_H
#define GETTINGSTARTED_LINEAR_ELASTICITY_ASSEMBLER_H

#include <lf/assemble/assemble_concepts.h>

/**
 * @brief Namespace for linear elasticity assembly operations
 * @details Provides functions for assembling global Galerkin matrices and
 *          vectors for the linear elasticity problem
 */
namespace LinearElasticityAssembler {

    /**
     * @brief Assembles global stiffness matrix from local contributions
     * @details This uses the Eval functions of the @link LinearFEElementMatrix @endlink or @link
     * ParametricFEElementMatrix @endlink and uses the local ordering of the shape functions to add to the global
     * Galerkin matrix
     * @tparam TMPMATRIX Matrix type for assembly
     * @tparam ENTITY_MATRIX_PROVIDER Provider of element matrices
     * @param codim Codimension of entities to process
     * @param dof_handler_trial Trial space DOF handler
     * @param dof_handler_test Test space DOF handler
     * @param entity_matrix_provider Provider of element matrices
     * @param matrix Global matrix to assemble into
     */
    template<typename TMPMATRIX, lf::assemble::EntityMatrixProvider ENTITY_MATRIX_PROVIDER>
    void AssembleMatrixLocally (lf::base::dim_t codim, const lf::assemble::DofHandler &dof_handler_trial,
                                const lf::assemble::DofHandler &dof_handler_test, ENTITY_MATRIX_PROVIDER &entity_matrix_provider,
                                TMPMATRIX &matrix) {

        //Get pointer to underlying mesh
        auto mesh = dof_handler_trial.Mesh();
        LF_ASSERT_MSG(mesh == dof_handler_test.Mesh(), "Trial and test space must be defined on the same mesh");

        //Loop over all the entities of the specified codimension
        for (const lf::mesh::Entity *entity : mesh->Entities(codim)) {

            //Only calculate for active cells (should include all cells)
            if(entity_matrix_provider.isActive(*entity)) {

                //Obtain size of element matrix, multiply by 2 due to displacements in x and y
                const lf::assemble::size_type nrows_local = dof_handler_test.NumLocalDofs(*entity) * 2;
                const lf::assemble::size_type ncols_local = dof_handler_trial.NumLocalDofs(*entity) * 2;

                //Obtain row and column indices for contributions to global matrix
                std::span<const lf::assemble::gdof_idx_t> row_idx (dof_handler_test.GlobalDofIndices(*entity));
                std::span<const lf::assemble::gdof_idx_t> col_idx (dof_handler_trial.GlobalDofIndices(*entity));

                //Obtain element matrix
                const auto elem_mat {entity_matrix_provider.Eval(*entity)};

                LF_ASSERT_MSG(elem_mat.rows() >= nrows_local, "nrows mismatch " << elem_mat.rows() << " <-> " << nrows_local
                                                << ", entity " << mesh->Index(*entity));
                LF_ASSERT_MSG(elem_mat.cols() >= ncols_local, "ncols mismatch " << elem_mat.cols() << " <-> " << nrows_local
                                                << ", entity " << mesh->Index(*entity));


                //Loop over all elements of the element matrix
                for (int i = 0; i < nrows_local; i++) {
                    for (int j = 0; j < ncols_local; j++) {

                        //We divide by two because there are components in both the x and y direction
                        if (i % 2 == 0 and j % 2 == 0) {
                            matrix.AddToEntry(row_idx[i / 2] * 2, col_idx[j / 2] * 2, elem_mat(i, j));
                        }
                        else if (i % 2 == 0) {
                            matrix.AddToEntry(row_idx[i / 2] * 2, col_idx[j / 2] * 2 + 1, elem_mat(i, j));
                        }
                        else if (j % 2 == 0) {
                            matrix.AddToEntry(row_idx[i / 2] * 2 + 1, col_idx[j / 2] * 2, elem_mat(i, j));
                        }
                        else {
                            matrix.AddToEntry(row_idx[i / 2] * 2 + 1, col_idx[j / 2] * 2 + 1, elem_mat(i, j));
                        }
                    }
                }
            }
        }
    }

    //This function will need to be called multiple times in case there is a body force
    /**
     * @brief Assembles global load vector from local contributions
     * @details This uses the Eval functions of the @link LinearFELoadVector @endlink or @link
     * ParametricFELoadVector @endlink and uses the local ordering of the shape functions to add to the global
     * load vector. Note that similar to these classes, this was added for completely in the case where traction
     * conditions or body forces are added, but ended up not being used in the implementation
     * @tparam VECTOR Vector type for assembly
     * @tparam ENTITY_VECTOR_PROVIDER Provider of element vectors
     * @param codim Codimension of entities to process
     * @param dof_handler DOF handler
     * @param entity_vector_provider Provider of element vectors
     * @param resultvector Global vector to assemble into
     */
    template <typename VECTOR, lf::assemble::EntityVectorProvider ENTITY_VECTOR_PROVIDER>
    void AssembleVectorLocally(lf::base::dim_t codim, const lf::assemble::DofHandler &dof_handler,
                               ENTITY_VECTOR_PROVIDER &entity_vector_provider, VECTOR &resultvector) {

        auto mesh = dof_handler.Mesh();

        for (const lf::mesh::Entity *entity: mesh->Entities(codim)) {

            if (entity_vector_provider.isActive(*entity)) {

                const lf::base::RefEl refEl {entity->RefEl()};

                //We multiply by two due to the x and y components of displacement
                const lf::assemble::size_type veclen = dof_handler.NumLocalDofs(*entity) * 2;

                //Obtain global indices for contributions of the entity
                const std::span<const lf::assemble::gdof_idx_t> dof_idx(dof_handler.GlobalDofIndices(*entity));

                //Obtain local vector from entity_vector_provider object
                const auto elem_vec {entity_vector_provider.Eval(*entity)};
                LF_ASSERT_MSG(elem_vec.size() >= veclen, "length mismatch " << elem_vec.size() << " <-> " << veclen
                                                 << ", entity " << mesh->Index(*entity));

                //Assembly loop, needs to be different if the entity is a cell or edge
                for (int i = 0; i < veclen; i++) {
                    // We use the same method as in AssembleMatrixLocally, due to the presence of both x and y displacements
                    if (i % 2 == 0) {
                        resultvector[dof_idx[i / 2] * 2] += elem_vec[i];
                    } else {
                        resultvector[dof_idx[i / 2] * 2 + 1] += elem_vec[i];
                    }
                }
            }
        }
    }

    // This function was first created to generate training data for a PINN, but has been repurposed to generate the
    // matrix of stresses and strains for a given FEM calculation
    // This consists of reshaping the displacement vector to (2, num_nodes) where each column represents ux and uy
    // Along with the displacement vector, the function will return the corresponding original node coordinates
    // in the same shape, (2, num_nodes)
    // This also includes the strain matrix, which will have shape (3, num_quad_points)
    // The stress matrix will have the same shape as the strain matrix
    // The stress and strain will be paired with their vector of coordinates, with shape (2, num_quad_points)
    /**
     * @brief Computes stress and strain fields from displacement solution
     * @details This uses the stressStrain functions of the @link LinearFEElementMatrix @endlink or @link
     * ParametricFEElementMatrix @endlink classes, and concatenates all the results
     * @tparam ENTITY_MATRIX_PROVIDER Provider of element matrices
     * @param mesh_ptr Pointer to mesh
     * @param disp_vec Global displacement vector
     * @param assembler Matrix assembler
     * @param degree FE space degree (1 or 2)
     * @return Tuple of (displacement matrix, node coordinates, stress matrix,
     *         strain matrix, quadrature point coordinates)
     */
    template<lf::assemble::EntityMatrixProvider ENTITY_MATRIX_PROVIDER>
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> stressStrainLoader
            (const std::shared_ptr<lf::mesh::Mesh> &mesh_ptr, Eigen::VectorXd &disp_vec, ENTITY_MATRIX_PROVIDER &assembler, int degree) {

        // Declare the mesh and the dof_handler for future use
        std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space;
        if (degree == 1){
            fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
        }
        else {
            fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);
        }
        const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};
        const lf::uscalfe::size_type N_dofs {dofh.NumDofs()};
        auto mesh = dofh.Mesh();
        // Build the matrix of node coordinates
        Eigen::MatrixXd node_coords(2, N_dofs);
        for (const lf::mesh::Entity *node : mesh->Entities(2)) {
            node_coords.block<2, 1>(0, dofh.GlobalDofIndices(*node)[0])
                    << node->Geometry()->Global(node->Geometry()->RefEl().NodeCoords());
        }

        // Build the matrix of displacements
        Eigen::MatrixXd disp_mat(2, N_dofs);
        disp_mat << disp_vec.reshaped(2, N_dofs);

        // Build the matrix of stresses, strains, and quadrature point coordinates
        Eigen::Matrix<double, 3, Eigen::Dynamic> stresses;
        Eigen::Matrix<double, 3, Eigen::Dynamic> strains;
        Eigen::Matrix<double, 2, Eigen::Dynamic> qr_coords;

        for (const lf::mesh::Entity *cell : mesh->Entities(0)) {

            const lf::base::RefEl ref_el {cell->RefEl()};
            auto trio = assembler.stressStrain(*cell, disp_vec, dofh);

            switch (ref_el) {

                // In this case we must expand the matrix by 6 columns to fit the new points
                case lf::base::RefEl::kTria() : {

                    auto cols = stresses.cols();
                    stresses.conservativeResize(Eigen::NoChange_t(), cols + 6);
                    strains.conservativeResize(Eigen::NoChange_t(), cols + 6);
                    qr_coords.conservativeResize(Eigen::NoChange_t(), cols + 6);

                    stresses.block<3, 6>(0, cols) << std::get<0>(trio);
                    strains.block<3, 6>(0, cols) << std::get<1>(trio);
                    qr_coords.block<2, 6>(0, cols) << std::get<2>(trio);

                    break;
                }

                // In this case we must expand the matrix by 4 columns to fit the new points
                case lf::base::RefEl::kQuad() : {

                    auto cols = stresses.cols();
                    stresses.conservativeResize(Eigen::NoChange_t(), cols + 4);
                    strains.conservativeResize(Eigen::NoChange_t(), cols + 4);
                    qr_coords.conservativeResize(Eigen::NoChange_t(), cols + 4);

                    stresses.block<3, 4>(0, cols) << std::get<0>(trio);
                    strains.block<3, 4>(0, cols) << std::get<1>(trio);
                    qr_coords.block<2, 4>(0, cols) << std::get<2>(trio);

                    break;
                }

                default: {
                    LF_ASSERT_MSG(false, "Illegal Entity Type sent to pinnDataLoader");
                }
            }
        }

        return std::tuple(disp_mat, node_coords, stresses, strains, qr_coords);
    }

    // This function iterates through all cells of the mesh, and sums up the energies from each cell
    /**
     * @brief Calculates total strain energy of the system
     * @details This uses the energyCalc functions of the @link LinearFEElementMatrix @endlink or @link
     * ParametricFEElementMatrix @endlink and sums over the results of all the cells
     * @tparam ENTITY_MATRIX_PROVIDER Provider of element matrices
     * @param mesh_ptr Pointer to mesh
     * @param disp_vec Global displacement vector
     * @param assembler Matrix assembler
     * @param degree FE space degree
     * @return Total strain energy
     */
    template<lf::assemble::EntityMatrixProvider ENTITY_MATRIX_PROVIDER>
    double energyCalculator(const std::shared_ptr<lf::mesh::Mesh> &mesh_ptr,
        Eigen::VectorXd &disp_vec, ENTITY_MATRIX_PROVIDER &assembler, int degree) {

        // Declare the mesh and the dof_handler for future use
        std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space;
        if (degree == 1){
            fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
        }
        else {
            fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);
        }
        const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};
        auto mesh = dofh.Mesh();
        double energy = 0;

        for (const lf::mesh::Entity *cell : mesh->Entities(0)) {
            energy += assembler.energyCalc(*cell, disp_vec, dofh);
        }

        return energy;
    }

}

#endif //GETTINGSTARTED_LINEAR_ELASTICITY_ASSEMBLER_H

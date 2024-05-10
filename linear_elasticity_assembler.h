//
// Created by Liam Curtis on 2024-05-02.
//
#ifndef GETTINGSTARTED_LINEAR_ELASTICITY_ASSEMBLER_H
#define GETTINGSTARTED_LINEAR_ELASTICITY_ASSEMBLER_H

#include <lf/assemble/assemble.h>
#include <lf/assemble/assemble_concepts.h>


namespace LinearElasticityAssembler {

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
                    //We use the same method as in AssembleMatrixLocally, due to the presence of both x and y displacements
                    if (i % 2 == 0) {
                        resultvector[dof_idx[i / 2] * 2] += elem_vec[i];
                    } else {
                        resultvector[dof_idx[i / 2] * 2 + 1] += elem_vec[i];
                    }
                }
            }
        }
    }

}

#endif //GETTINGSTARTED_LINEAR_ELASTICITY_ASSEMBLER_H

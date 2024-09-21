//
// Created by Liam Curtis on 2024-05-15.
//
#ifndef METALFOAMS_MESH_H
#define METALFOAMS_MESH_H

#include <lf/mesh/utils/utils.h>
#include <lf/base/base.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/io/io.h>
#include <lf/uscalfe/uscalfe.h>
#include <filesystem>
#include <memory>
#include <typeinfo>
#include "linear_matrix_computation.h"
#include "linear_elasticity_assembler.h"
#include "parametric_matrix_computation.h"

// This file include various methods to aid in solving the linear elasticity problem on various meshes
// As such, the primary functionality is to take in a mesh, given by a path string,
// and return the corresponding solution vector for that mesh

//In addition, this file will contain helper functions to allow for the training of the PINN
//Its purpose will be to return the collocation point data as well as the necessary training points for the
//boundary conditions and body force (if present)

std::vector<long> dist0nodes;
std::vector<long> trac0nodes;
std::vector<long> trac1nodes;

//This method uses sol_vec and dist0nodes to build the full solution vector, which will include the prescribed
//displacement BCs
Eigen::VectorXd adjustDisplacementVector (Eigen::VectorXd sol_vec, unsigned int N_dofs) {

    //Initialize intended return vector of displacements
    Eigen::VectorXd u_vec = Eigen::VectorXd::Zero(2 * N_dofs);

    //Ensure that dist0nodes is nonempty
    int j = -1;
    if (!dist0nodes.empty()) {
        j = 0;
    }
    else {
        return sol_vec;
    }

    //Loops over the indices of all nodes
    bool check = true;
    int k = 0;
    for (int i = 0; i < N_dofs; i++) {
        if (check and i == dist0nodes[j]) {
            j++;
            if (j >= dist0nodes.size()) {
                check = false;
                //j = dist0nodes.size() - 1;
            }
        }
        else {
            u_vec[2 * i] = sol_vec(2 * k);
            u_vec[2 * i + 1] = sol_vec(2 * k + 1);
            k++;
        }
    }
    return u_vec;
}

// Only the method adjustSolution is allowed to modify dist0nodes, which contains the indices of the points
// to be removed in the solution vector
// The method adjustSolution removes the rows and columns in the stiffness matrix that don't play a part in
// calculating the solution due to the given distance boundary conditions
std::pair<Eigen::SparseMatrix<double>, Eigen::Matrix<double, Eigen::Dynamic, 1>> adjustSolution
        (const lf::io::GmshReader &reader, lf::assemble::COOMatrix<double> &A
                , Eigen::Matrix<double, Eigen::Dynamic, 1> &phi, int physicalNum) {

    //Vector of nodes that will need to be "removed" from stiffness matrix as they have a 0 displacement BC
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(reader.mesh());
    const lf::mesh::Mesh &mesh {*(reader.mesh())};
    const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs {dofh.NumDofs()};
    int edcnt = 0;

    //Iterate through all edges on the mesh to find the nodes with a 0 displacement BC
    for(const lf::mesh::Entity *edge : mesh.Entities(1)) {

        LF_ASSERT_MSG(edge->RefEl() == lf::base::RefEl::kSegment(), " edge must be a SEGMENT!");

        //Check if edge has the physical entity number of the 0 displacement BC
        if (reader.IsPhysicalEntity(*edge, physicalNum)) {
            edcnt++;

            //Get indices of nodes through call to GlobalDofIndices
            const std::span<const lf::assemble::gdof_idx_t> dof_idx(dofh.GlobalDofIndices(*edge));

            //If the node is not already present in dist0nodes, we add it
            if (std::find(dist0nodes.begin(), dist0nodes.end(), dof_idx[0]) == dist0nodes.end()) {
                dist0nodes.insert(dist0nodes.begin(), dof_idx[0]);
            }
            if (std::find(dist0nodes.begin(), dist0nodes.end(), dof_idx[1]) == dist0nodes.end()) {
                dist0nodes.insert(dist0nodes.begin(), dof_idx[1]);
            }
        }
    }

    std::sort(dist0nodes.begin(), dist0nodes.end());

    std::vector<long> indx(2*N_dofs);
    //Vector indx stores all the indices of the complete stiffness matrix
    for (int i = 0; i < 2*N_dofs; ++i) {
        indx[i] = i;
    }

    int cnt = 0;
    //We eliminate the indices present in dist0nodes, two elements are erased per node
    for(long elem: dist0nodes) {
        indx.erase(indx.begin() + 2 * (elem - cnt));
        indx.erase(indx.begin() + 2 * (elem - cnt));
        cnt++;
    }

    auto A_dense = A.makeDense();
    auto A_reduced = A_dense(indx, indx);

    auto phi_reduced = phi(indx);

    return std::pair(A_reduced.sparseView(), phi_reduced);
}

// The method mesh_assembler takes in a mesh number, along with its Gmsh reader and returns the reduced stiffness matrix
// and load vector to be solved
std::tuple<Eigen::SparseMatrix<double>, Eigen::Matrix<double, Eigen::Dynamic, 1>, double, double, bool> mesh_assembler
(int num , lf::io::GmshReader &reader) {

    const std::shared_ptr<lf::mesh::Mesh> mesh_ptr = reader.mesh();
    std::cout << "Num entities: " << mesh_ptr->NumEntities(2) << std::endl;
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
    const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};

    const lf::uscalfe::size_type N_dofs {dofh.NumDofs()};
    const lf::mesh::Mesh &mesh {*mesh_ptr};
    lf::assemble::COOMatrix<double> A(N_dofs*2, N_dofs*2);
    Eigen::Matrix<double, Eigen::Dynamic, 1> phi(N_dofs*2);
    std::pair<Eigen::SparseMatrix<double>, Eigen::Matrix<double, Eigen::Dynamic, 1>> result;
    std::tuple<Eigen::SparseMatrix<double>, Eigen::Matrix<double, Eigen::Dynamic, 1>, double, double, bool> my_result;
    phi.setZero();

    if (num == 0 or num == 1 or num == 2 or num == 3 or num == 4) {

        lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{mesh_ptr, 1, false};

        for(const lf::mesh::Entity *edge : mesh.Entities(1)) {
            LF_ASSERT_MSG(edge->RefEl() == lf::base::RefEl::kSegment(), " edge must be a SEGMENT!");
            if (reader.IsPhysicalEntity(*edge, 9)) {
                bd_flags(*edge) = true;
            }
        }

        std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction = [] (const Eigen::Vector2d &coords) {
            double eps = 1.0e-5;
            Eigen::Vector2d trac;
            if (coords(0) < (2 + eps) and coords(0) > (-eps) and abs(coords(1) - 1) < eps) {
                trac << 0.0, -20.0;
            }
            else {
                trac << 0.0, 0.0;
            }
            return trac;
        };

        double E = 30000000.0;
        double v = 0.3;
        //False in our constructor means we are doing a plane-stress calculation
        LinearMatrixComputation::LinearFEElementMatrix assemble {E, v, false};
        LinearMatrixComputation::LinearFELoadVector load {bd_flags, traction};

        LinearElasticityAssembler::AssembleMatrixLocally(0, dofh, dofh, assemble, A);
        LinearElasticityAssembler::AssembleVectorLocally(1, dofh, load, phi);

        result = adjustSolution(reader, A, phi, 10);
        my_result = std::tuple(result.first, result.second, E, v, false);
    }



    else if (num == 10 or num == 11 or num == 12 or num == 13 or num == 14 or num == 15) {
        //Get the locations of the entities on the boundary
        std::vector<std::pair<lf::base::size_type, std::string>> phys_ent_list {reader.PhysicalEntities(1)};

        //Initialize to false since there is only a body force here
        lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{mesh_ptr, 1, false};

        //Traction function will always return 0
        std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction = [] (const Eigen::Vector2d &coords) {
            Eigen::Vector2d trac;
            trac.setZero();
            return trac;
        };

        std::function<Eigen::Vector2d (const Eigen::Vector2d &)> body = [] (const Eigen::Vector2d &coords) {
            Eigen::Vector2d force;
            force << 0.0, -0.001;
            return force;
        };

        double E = 100000.0;
        double v = 0.3;
        //False in our constructor means we are doing a plane-stress calculation
        LinearMatrixComputation::LinearFEElementMatrix assemble {E, v, false};
        LinearMatrixComputation::LinearFELoadVector load {bd_flags, traction, body};

        LinearElasticityAssembler::AssembleMatrixLocally(0, dofh, dofh, assemble, A);
        LinearElasticityAssembler::AssembleVectorLocally(0, dofh, load, phi);

        result = adjustSolution(reader, A, phi, 8);
        my_result = std::tuple(result.first, result.second, E, v, false);
    }

    else if (num == 20 or num == 21 or num == 22 or num == 23 or num == 24 or num == 25) {

        lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{mesh_ptr, 1, false};

        for(const lf::mesh::Entity *edge : mesh.Entities(1)) {
            LF_ASSERT_MSG(edge->RefEl() == lf::base::RefEl::kSegment(), " edge must be a SEGMENT!");
            if (reader.IsPhysicalEntity(*edge, 7)) {
                bd_flags(*edge) = true;
            }
        }

        std::function<Eigen::Vector2d (const Eigen::Vector2d &)> traction = [] (const Eigen::Vector2d &coords) {
            double eps = 1.0e-5;
            Eigen::Vector2d trac;
            if (coords(0) < (2 + eps) and coords(0) > (-eps) and abs(coords(1) - 2) < eps) {
                trac << 0.0, -20.0;
            }
            else {
                trac << 0.0, 0.0;
            }
            return trac;
        };

        double E = 30000000.0;
        double v = 0.3;


        if (num == 20 or num == 22 or num == 24) {
            //False in our constructor means we are doing a plane-stress calculation
            LinearMatrixComputation::LinearFEElementMatrix assemble{E, v, false};
            LinearMatrixComputation::LinearFELoadVector load{bd_flags, traction};

            LinearElasticityAssembler::AssembleMatrixLocally(0, dofh, dofh, assemble, A);

            LinearElasticityAssembler::AssembleVectorLocally(1, dofh, load, phi);
            LinearElasticityAssembler::AssembleVectorLocally(0, dofh, load, phi);

            result = adjustSolution(reader, A, phi, 6);
            my_result = std::tuple(result.first, result.second, E, v, false);
        }
        else {
            ParametricMatrixComputation::ParametricFEElementMatrix assemble{E, v, false};
            ParametricMatrixComputation::ParametricFELoadVector load{bd_flags, traction};

            LinearElasticityAssembler::AssembleMatrixLocally(0, dofh, dofh, assemble, A);

            LinearElasticityAssembler::AssembleVectorLocally(1, dofh, load, phi);
            LinearElasticityAssembler::AssembleVectorLocally(0, dofh, load, phi);

            result = adjustSolution(reader, A, phi, 6);
            my_result = std::tuple(result.first, result.second, E, v, false);
        }
    }
    return my_result;
}

//This function's purpose is to initialize the vector trac0nodes and trac1nodes with the indices of the nodes
//trac0nodes will contain the indices for all the nodes that have a 0 traction boundary condition
//trac1nodes will contain the indices for all the nodes that have a non-zero traction boundary condition
void tractionNodes(int trac1, int trac0, lf::io::GmshReader &reader) {

    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(reader.mesh());
    const lf::mesh::Mesh &mesh {*(reader.mesh())};
    const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs {dofh.NumDofs()};

    for (const lf::mesh::Entity *edge : mesh.Entities(1)) {

        LF_ASSERT_MSG(edge->RefEl() == lf::base::RefEl::kSegment(), " edge must be a SEGMENT!");

        if (trac1 != -1) {

            if (reader.IsPhysicalEntity(*edge, trac1)) {

                //Get indices of nodes through call to GlobalDofIndices
                const std::span<const lf::assemble::gdof_idx_t> dof_idx(dofh.GlobalDofIndices(*edge));

                //If the node is not already present in trac1nodes, we add it
                if (std::find(trac1nodes.begin(), trac1nodes.end(), dof_idx[0]) == trac1nodes.end()) {
                    trac1nodes.insert(trac1nodes.begin(), dof_idx[0]);
                }
                if (std::find(trac1nodes.begin(), trac1nodes.end(), dof_idx[1]) == trac1nodes.end()) {
                    trac1nodes.insert(trac1nodes.begin(), dof_idx[1]);
                }
            }
        }
        if (trac0 != -1) {

            if (reader.IsPhysicalEntity(*edge, trac0)) {

                //Get indices of nodes through call to GlobalDofIndices
                const std::span<const lf::assemble::gdof_idx_t> dof_idx(dofh.GlobalDofIndices(*edge));

                //If the node is not already present in trac0nodes, we add it
                if (std::find(trac0nodes.begin(), trac0nodes.end(), dof_idx[0]) == trac0nodes.end()) {
                    trac0nodes.insert(trac0nodes.begin(), dof_idx[0]);
                }
                if (std::find(trac0nodes.begin(), trac0nodes.end(), dof_idx[1]) == trac0nodes.end()) {
                    trac0nodes.insert(trac0nodes.begin(), dof_idx[1]);
                }
            }
        }
    }
}

// This function takes in a mesh number and returns traction _vectors, normal _vectors, node coords, displacement _vectors
// its respective node coordinates, body force, and its respective node coordinates
// These will be separated into 3 matrices, one for traction, one for displacement, and one for body force
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> boundaryAssembler(int num, lf::io::GmshReader &reader) {

    const std::shared_ptr<lf::mesh::Mesh> mesh_ptr = reader.mesh();
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
    const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};

    const lf::uscalfe::size_type N_dofs {dofh.NumDofs()};
    const lf::mesh::Mesh &mesh {*mesh_ptr};

    //These matrices have 4 rows, the top 2 will contain values, and the bottom 2 will contain coordinates
    Eigen::MatrixXd body_coords(4, N_dofs);
    Eigen::MatrixXd disp_BC_coords(4, dist0nodes.size());

    //The top 2 rows will contain values, the middle 2 will contain the normal vector, and the bottom 2 will have coords
    Eigen::Matrix<double, 6, Eigen::Dynamic> trac_BC_coords;

    //We have a clamped left end, no body force, traction on top, and no traction everywhere else
    if (num == 0 or num == 1 or num == 2 or num == 3 or num == 4) {

        tractionNodes(9, 11, reader);
        trac_BC_coords.resize(6,trac1nodes.size() + trac0nodes.size());
        int i = 0, j = 0, k = 0;
        double eps = 1.0e-5;

        for (const lf::mesh::Entity *node : mesh.Entities(2)) {

            LF_ASSERT_MSG(node->RefEl() == lf::base::RefEl::kPoint(), " edge must be a NODE!");

            auto coords = node->Geometry()->Global(node->RefEl().NodeCoords());

            body_coords.block<4, 1>(0, i) << 0.0, 0.0, coords(0), coords(1);
            i++;

            if (abs(coords(1) - 1) < eps) {
                trac_BC_coords.block<6, 1>(0, j) << 0.0, -20.0, 0.0, 1.0, coords(0), coords(1);
                j++;
            }
            if (coords(0) < eps) {
                disp_BC_coords.block<4, 1>(0, k) << 0.0, 0.0, coords(0), coords(1);
                k++;
            }
            if (abs(coords(1) - 0.25 * (coords(0))) < eps) {
                trac_BC_coords.block<6, 1>(0, j) << 0.0, 0.0, 0.9701, -0.2425, coords(0), coords(1);
                j++;
            }
            else if (coords(0) > (2 - eps)) {
                trac_BC_coords.block<6, 1>(0, j) << 0.0, 0.0, 1.0, 0.0, coords(0), coords(1);
                j++;
            }
        }
    }


    //Now we have a body force, clamped left end, and 0 traction on the rest of the boundary
    else if (num == 10 or num == 11 or num == 12 or num == 13 or num == 14 or num == 15) {

        tractionNodes(-1, 9, reader);
        trac_BC_coords.resize(6,trac1nodes.size() + trac0nodes.size());
        int i = 0, j = 0, k = 0;
        double eps = 1.0e-5;

        for (const lf::mesh::Entity *node : mesh.Entities(2)) {

            LF_ASSERT_MSG(node->RefEl() == lf::base::RefEl::kPoint(), " edge must be a NODE!");
            auto coords = node->Geometry()->Global(node->RefEl().NodeCoords());

            body_coords.block<4, 1>(0, i) << 0.0, -0.001, coords(0), coords(1);
            i++;

            if (coords(0) < eps) {
                disp_BC_coords.block<4, 1>(0, k) << 0.0, 0.0, coords(0), coords(1);
                k++;
            }
            if (coords(1) < eps) {
                trac_BC_coords.block<6, 1>(0, j) << 0.0, 0.0, 0.0, -1.0, coords(0), coords(1);
                j++;
            }
            else if (coords(0) > (25.0 - eps)) {
                trac_BC_coords.block<6, 1>(0, j) << 0.0, 0.0, 1.0, 0.0, coords(0), coords(1);
                j++;
            }
            else if (coords(1) > (1 - eps)) {
                trac_BC_coords.block<6, 1>(0, j) << 0.0, 0.0, 0.0, 1.0, coords(0), coords(1);
                j++;
            }

        }
    }


    //We have a clamped end, a traction force, and no traction or body force on the remainder of the mesh
    else if (num == 20 or num == 21 or num == 22 or num == 23 or num == 24 or num == 25) {

        tractionNodes(7, 8, reader);
        trac_BC_coords.resize(6,trac1nodes.size() + trac0nodes.size());
        int i = 0, j = 0, k = 0;
        double eps = 1.0e-5;

        for (const lf::mesh::Entity *node : mesh.Entities(2)) {
            LF_ASSERT_MSG(node->RefEl() == lf::base::RefEl::kPoint(), " edge must be a NODE!");
            auto coords = node->Geometry()->Global(node->RefEl().NodeCoords());

            body_coords.block<4, 1>(0, i) << 0.0, 0.0, coords(0), coords(1);
            i++;

            if (coords(0) < eps) {
                disp_BC_coords.block<4, 1>(0, k) << 0.0, 0.0, coords(0), coords(1);
                k++;
            }
            if (coords(1) > (2 - eps)) {
                trac_BC_coords.block<6, 1>(0, j) << 0.0, -20.0, 0.0, 1.0, coords(0), coords(1);
                j++;
            }
            if(coords(0) > (2 - eps)) {
                trac_BC_coords.block<6, 1>(0, j) << 0.0, 0.0, 1.0, 0.0, coords(0), coords(1);
                j++;
            }
            else if (((coords(0) - 2) * (coords(0) - 2) + coords(1) * coords(1) - 1.5 * 1.5) < eps) {
                trac_BC_coords.block<6, 1>(0, j) << 0.0, 0.0, (2 - coords(0)) / 1.5, -coords(1) / 1.5,
                        coords(0), coords(1);
                j++;
            }
            else if (coords(1) < eps) {
                trac_BC_coords.block<6, 1>(0, j) << 0.0, 0.0, 0.0, -1.0, coords(0), coords(1);
                j++;
            }
        }
    }
    return std::tuple(disp_BC_coords, trac_BC_coords, body_coords);
}


void test_mesh(std::string path) {

    auto factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    const std::filesystem::path here = __FILE__;
    auto mesh_file = here.parent_path();
    mesh_file += path;
    lf::io::GmshReader reader(std::move(factory), mesh_file);

    const std::shared_ptr<lf::mesh::Mesh> mesh_ptr = reader.mesh();

    const lf::mesh::Mesh &mesh {*mesh_ptr};
    std::cout << "Mesh from file " << mesh_file.string() << ": [" << mesh.DimMesh() << ", " << mesh.DimWorld() << "] dim:" << "\n";
    std::cout << mesh.NumEntities(0) << " cells, " << mesh.NumEntities(1) << " edges, " << mesh.NumEntities(2) << " nodes \n";

    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
    const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs {dofh.NumDofs()};


    //Obtain the mesh number in the file name
    std::string delimiter = "test";
    size_t pos = 0;
    std::string number;
    if ((pos = path.find(delimiter)) != std::string::npos) {
        path.erase(0, pos + delimiter.length());
    }
    delimiter = ".msh";
    if ((pos = path.find(delimiter)) != std::string::npos) {
        path.erase(pos, pos + delimiter.length());
    }

    int mesh_num = stoi(path);

    auto result = mesh_assembler(mesh_num, reader);

    Eigen::VectorXd sol_vec = Eigen::VectorXd::Zero(2 * N_dofs);

    //Solve the system of equations to obtain the displacements
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(std::get<0>(result));

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Could not decompose the matrix");
    }
    sol_vec = solver.solve(std::get<1>(result));

    auto true_sol_vec = adjustDisplacementVector(sol_vec, N_dofs);

    //I call the linear version as the implementation for both the parametric and linear cases are the same
    LinearMatrixComputation::LinearFEElementMatrix assemble{std::get<2>(result), std::get<3>(result), std::get<4>(result)};

    //We now obtain the collocation points required to train the PINN
    auto collocationPoints = LinearElasticityAssembler::pinnDataLoader(mesh_ptr, true_sol_vec, assemble);

    auto boundaryPoints = boundaryAssembler(mesh_num, reader);

    std::cout << true_sol_vec << std::endl;
//    std::cout << std::get<0>(boundaryPoints) << "\n\n" << std::endl;
//    std::cout << std::get<1>(boundaryPoints) << "\n\n" << std::endl;
//    std::cout << std::get<2>(boundaryPoints) << std::endl;


}

#endif //METALFOAMS_MESH_H

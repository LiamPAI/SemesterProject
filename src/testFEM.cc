//
// Created by Liam Curtis on 2024-06-02.
//

#include <lf/mesh/utils/utils.h>
#include <lf/base/base.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/io/io.h>
#include <lf/uscalfe/uscalfe.h>
#include <filesystem>
#include <memory>
#include <typeinfo>
#include "../include/linear_matrix_computation.h"
#include "../include/linear_elasticity_assembler.h"
#include "../include/parametric_matrix_computation.h"
#include "../include/graph_mesh.h"


// This file contains various methods to ensure the correctness of linear_matrix_computation and parametric_matrix_computation

//This method uses sol_vec and dist0nodes to build the full solution vector, which will include the prescribed
//displacement BCs
Eigen::VectorXd adjustDisplacementVector (Eigen::VectorXd sol_vec, unsigned int N_dofs, std::vector<long> &dist0nodes) {

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

//Only the method adjustSolution is allowed to modify dist0nodes, which contains the indices of the points
//to be removed in the solution vector
std::pair<Eigen::SparseMatrix<double>, Eigen::Matrix<double, Eigen::Dynamic, 1>> adjustSolution
        (const lf::io::GmshReader &reader, lf::assemble::COOMatrix<double> &A
                , Eigen::Matrix<double, Eigen::Dynamic, 1> &phi, int physicalNum, int degree, std::vector<long> &dist0nodes) {

    //Vector of nodes that will need to be "removed" from stiffness matrix as they have a 0 displacement BC

    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space;

    if (degree == 1) {
        fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(reader.mesh());
    }
    else {
        fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(reader.mesh());
    }

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

            //If we are doing degree 2 Lagrangian FEM, we add the middle node of the edge every time
            if (degree != 1) {
                dist0nodes.insert(dist0nodes.begin(), dof_idx[2]);
            }

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

std::pair<Eigen::SparseMatrix<double>, Eigen::Matrix<double, Eigen::Dynamic, 1>> mesh_assembler(int num , lf::io::GmshReader &reader, int degree, std::vector<long> &dist0nodes) {

    const std::shared_ptr<lf::mesh::Mesh> mesh_ptr = reader.mesh();
    std::cout << "Num entities: " << mesh_ptr->NumEntities(2) << std::endl;

    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space;

    if (degree == 1) {
        fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
    }
    else {
        fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);
    }

    const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs {dofh.NumDofs()};
    const lf::mesh::Mesh &mesh {*mesh_ptr};
    std::pair<Eigen::SparseMatrix<double>, Eigen::Matrix<double, Eigen::Dynamic, 1>> result;

    if (num == 0 or num == 1 or num == 2 or num == 3 or num == 4 or num == 5 or num == 6) {

        lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{mesh_ptr, 1, false};

        int edcnt = 0;
        for (const lf::mesh::Entity *edge: mesh.Entities(1)) {
            LF_ASSERT_MSG(edge->RefEl() == lf::base::RefEl::kSegment(), " edge must be a SEGMENT!");
            if (reader.IsPhysicalEntity(*edge, 9)) {
                edcnt++;
                bd_flags(*edge) = true;
            }
        }

        std::function<Eigen::Vector2d(const Eigen::Vector2d &)> traction = [](const Eigen::Vector2d &coords) {
            double eps = 1.0e-5;
            Eigen::Vector2d trac;
            if (abs(coords(1) - 1) < eps) {
                trac << 0.0, -20.0;
            } else {
                trac << 0.0, 0.0;
            }
            return trac;
        };

        double E = 30000000.0;
        double v = 0.3;

        lf::assemble::COOMatrix<double> A(N_dofs*2, N_dofs*2);

        Eigen::Matrix<double, Eigen::Dynamic, 1> phi(N_dofs*2);
        phi.setZero();


        //False in our constructor means we are doing a plane-stress calculation
        if (degree == 1) {
            LinearMatrixComputation::LinearFEElementMatrix assemble{E, v, false};
            LinearMatrixComputation::LinearFELoadVector load{bd_flags, traction};
            LinearElasticityAssembler::AssembleMatrixLocally(0, dofh, dofh, assemble, A);
            LinearElasticityAssembler::AssembleVectorLocally(1, dofh, load, phi);
        }
        else {
            ParametricMatrixComputation::ParametricFEElementMatrix assemble{E, v, false};
            ParametricMatrixComputation::ParametricFELoadVector load{bd_flags, traction};
            LinearElasticityAssembler::AssembleMatrixLocally(0, dofh, dofh, assemble, A);
            LinearElasticityAssembler::AssembleVectorLocally(1, dofh, load, phi);
        }
        std::cout << "Galerkin matrix: \n" << A.makeDense() << std::endl;
        result = adjustSolution(reader, A, phi, 10, degree, dist0nodes);
    }

    return result;

}

void test_mesh(std::string path, int degree) {

    std::vector<long> dist0nodes;

    auto factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    const std::filesystem::path here = __FILE__;
    auto mesh_file = here.parent_path().parent_path();
    mesh_file += path;
    lf::io::GmshReader reader(std::move(factory), mesh_file);

    const std::shared_ptr<lf::mesh::Mesh> mesh_ptr = reader.mesh();

    const lf::mesh::Mesh &mesh {*mesh_ptr};
    std::cout << "Mesh from file " << mesh_file.string() << ": [" << mesh.DimMesh() << ", " << mesh.DimWorld() << "] dim:" << "\n";
    std::cout << mesh.NumEntities(0) << " cells, " << mesh.NumEntities(1) << " edges, " << mesh.NumEntities(2) << " nodes \n";

    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space;

    if (degree == 1) {
        fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
    }
    else {
        fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);
    }

    const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs {dofh.NumDofs()};
    std::cout << "There are " << N_dofs << " nodes" << std::endl;
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

    auto result = mesh_assembler(mesh_num, reader, degree, dist0nodes);

    Eigen::VectorXd sol_vec = Eigen::VectorXd::Zero(2 * N_dofs);

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(result.first);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Could not decompose the matrix");
    }
    sol_vec = solver.solve(result.second);

    std::cout << "Max distance is " << sol_vec.lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << "sol_vec is \n" << sol_vec << std::endl;

    auto true_sol_vec = adjustDisplacementVector(sol_vec, N_dofs, dist0nodes);


    double E = 30000000.0;
    double v = 0.3;

    LinearMatrixComputation::LinearFEElementMatrix assemble {E, v, false};
    ParametricMatrixComputation::ParametricFEElementMatrix assemblePar{E, v, false};

    if (degree == 1) {
        auto linearPinn = LinearElasticityAssembler::stressStrainLoader(mesh_ptr, true_sol_vec, assemble, 1);
        std::cout << "Value of stresses: \n" << std::get<2>(linearPinn) << std::endl;
        std::cout << "Value of strains: \n" << std::get<3>(linearPinn) << std::endl;
        std::cout << std::get<4>(linearPinn) << std::endl;
    }
    else {
        auto paraPinn = LinearElasticityAssembler::stressStrainLoader(mesh_ptr, true_sol_vec, assemblePar, 2);
        std::cout << std::get<2>(paraPinn) << std::endl;
        std::cout << std::get<3>(paraPinn) << std::endl;
        std::cout << std::get<4>(paraPinn) << std::endl;
    }

}

int main() {
    test_mesh("/meshes/test0.msh", 1);
//    test_mesh("/meshes/test4.msh", 2);


    // GraphMesh mesh;
    //
    // try
    // {
    //     mesh.buildSplitAndPrintMesh("testNE1.geo", 0.5, 0.1);
    // }
    // catch (const std::exception& e) {
    //     std::cerr << "An error occurred: " << e.what() << std::endl;
    //     return 1;
    // }
    //
    // return 0;

}
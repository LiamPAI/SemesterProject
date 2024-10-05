//
// Created by Liam Curtis on 26.09.2024.
//

#include "../include/mesh_parametrization.h"


void test_checkVectorLengths() {
    Eigen::MatrixXd test_vectors(2, 3);

    test_vectors << 0.0, 0.0, -1.0, 0.01, 0.0000001, 1.0;

    bool check = MeshParametrization::checkVectorLengths(test_vectors);

    assert(!check);
}

// The function creates a 3-branch parametrization (though for the purposes of this function it doesn't have to be a
// "valid" parametrization) and normalizes its vectors
void test_normalizeVectors() {
    // Initialize the necessary matrices
    Eigen::MatrixXd unnormalized_vectors(6,3);
    Eigen::MatrixXd terminals(6, 3);
    Eigen::MatrixXd widths(3, 3);

    terminals << 0.0, 3.0, 6.0, // 1st branch
                1.5, 1.5, 1.5,
                -2.0, -2.0, -2.0, // 2nd branch
                0.0, -4.0, -8.0,
                -2.0, -2.0, -2.0, // 3rd branch
                1.5, 6.5, 11.5;

    unnormalized_vectors << 0.0, 0.0, 0.0, // 1st branch
                            2.0, 5.71, -4.3,
                            3.0, 1.0, -1.0, // 2nd branch
                            0.0, 0.0, 0.0,
                            0.8, 1.6, 0.01,  // 3rd branch
                            0.6, 1.2, 0.0;

    widths << 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0;

    MeshParametrizationData multi_branch{3, widths, terminals, unnormalized_vectors};

    MeshParametrization::normalizeVectors(multi_branch.vectors);

    Eigen::MatrixXd correct_vectors(6,3);
    correct_vectors << 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.0, 0.6, 0.6, 0.0;

    assert(correct_vectors.isApprox(multi_branch.vectors));
}

// This function creates a single branch and a multi-branch parametrization to make sure the polynomial points for
// the parametrizations are correct
void test_polynomialPoints() {
    // Initialize the necessary matrices for the single branch and test it
    Eigen::MatrixXd vectors(2,3);
    Eigen::MatrixXd terminals(2, 3);
    Eigen::MatrixXd widths(1, 3);

    terminals << 0.0, 1.0, 1.0,
                 0.0, 0.0, -1.0;
    widths << 1.0, 0.9, 0.8;
    vectors <<  0.0, -1.0, -3.0,
                1.0, -1.0, 0.0;

    MeshParametrizationData single_branch{1, widths, terminals, vectors};

    Eigen::MatrixXd poly_points = MeshParametrization::polynomialPoints(single_branch);

    Eigen::MatrixXd correct_points(4, 3);
    correct_points <<  0.0, 1.3182, 1.4, 0.5, 0.318198, -1.0, 0.0, 0.681802, 0.6, -0.5, -0.318198, -1.0;

    assert(correct_points.isApprox(poly_points, 1e-4));

    // Initialize the necessary matrices for the multi branch and test it
    Eigen::MatrixXd multi_vectors(8,3);
    Eigen::MatrixXd multi_terminals(8, 3);
    Eigen::MatrixXd multi_widths(4, 3);

    multi_widths << 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8;
    multi_terminals << 1.0, 2.0, 3.0, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.0, -1.0, -2.0,
                        0.5, 0.5, 0.5, 1.0, 2.0, 3.0,
                        0.0, -1.0, -2.0, 0.5, 0.5, 0.5;
    multi_vectors << 0.0, 0.0, 0.0, 2.0, -2.0, 2.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 2.0, -2.0, 2.0;

    MeshParametrizationData multi_branch {4, multi_widths, multi_terminals, multi_vectors};

    Eigen::MatrixXd multi_points = MeshParametrization::polynomialPoints(multi_branch);

    Eigen::MatrixXd correct_multi_points(16, 3);

    correct_multi_points << 1, 2, 3, 1, 0.95, 0.9, 1, 2, 3, 0, 0.05, 0.1,
                            1, 0.95, 0.9, 0, -1, -2, 0, 0.05, 0.1, 0, -1, -2,
                            1, 0.95, 0.9, 1, 2, 3, 0, 0.05, 0.1, 1, 2, 3,
                            0, -1, -2, 1, 0.95, 0.9, 0, -1, -2, 0, 0.05, 0.1;

    assert(correct_multi_points.isApprox(multi_points, 1e-4));
}

// This function creates a couple single branches to test angleVectors, the first branch uses angles of exactly 45
// degrees, and the second branches has invalid angles
void test_angleVectors() {
    // Initialize the necessary matrices for the single branch and test it
    Eigen::MatrixXd vectors(2,3);
    Eigen::MatrixXd terminals(2, 3);
    Eigen::MatrixXd widths(1, 3);

    terminals << 0.0, 1.0, 1.0, 0.0, 0.0, -1.0;
    vectors << 0.0, -1.0, 1.0,
               1.0, -1.0, 0.0;
    widths << 1.0, 0.9, 0.8;

    MeshParametrizationData single_branch{1, widths, terminals, vectors};

    bool check1 = MeshParametrization::angleVectors(single_branch);

    assert(check1);

    Eigen::MatrixXd vectors2(2,3);
    Eigen::MatrixXd terminals2(2, 3);
    Eigen::MatrixXd widths2(1, 3);

    terminals2 << 0.0, 1.0, 2.0, 0.0, 0.0, -1.0;
    vectors2 << 0.0, -1.0, -3.0, 3.0, -2.0, 0.0;
    widths2 << 1.0, 0.9, 0.8;

    MeshParametrizationData single_branch2{1, widths2, terminals2, vectors2};

    bool check2 = MeshParametrization::angleVectors(single_branch2);

    assert(!check2);
}

// This function tests whether the vectors in a parametrization overlap
void test_intersectionVectors() {
    // Initialize the necessary matrices for the single branch and test it
    Eigen::MatrixXd vectors(2,3);
    Eigen::MatrixXd terminals(2, 3);
    Eigen::MatrixXd widths(1, 3);

    terminals << 0.0, 1.0, 2.0, 0.0, 0.0, 0.0;
    vectors << 0.0, 1.0, 0.0, 1.0, 0.0, 1.0;
    widths << 1.0, 1.99, 1.0;

    MeshParametrizationData single_branch{1, widths, terminals, vectors};

    bool check1 = MeshParametrization::intersectionVectors(single_branch);
    assert(check1);

    // In this branch, the vectors just barely overlap
    Eigen::MatrixXd vectors2(2,3);
    Eigen::MatrixXd terminals2(2, 3);
    Eigen::MatrixXd widths2(1, 3);

    terminals2 << 0.0, 1.0, 2.0, 0.0, 0.0, 0.0;
    vectors2 << 0.0, 1.0, 0.0, 1.0, 0.0, 1.0;
    widths2 << 1.0, 2.0, 1.0;

    MeshParametrizationData single_branch2{1, widths2, terminals2, vectors2};

    bool check2 = MeshParametrization::intersectionVectors(single_branch2);
    assert(!check2);
}

// This function tests whether two nearby branches overlap or not, useful for multi-branch parametrizations
void test_intersectionBranches() {
    // Initialize the necessary matrices for the single branch and test it
    Eigen::MatrixXd vectors1(2,3);
    Eigen::MatrixXd terminals1(2, 3);
    Eigen::MatrixXd widths1(1, 3);

    terminals1 << 0.0, 1.0, 1.0, 0.0, 0.0, -1.0;
    vectors1 << 0.0, 1.0, 1.0, 1.0, 1.0, 0.0;
    widths1 << 1.0, 1.0, 1.0;

    Eigen::MatrixXd vectors2(2,3);
    Eigen::MatrixXd terminals2(2, 3);
    Eigen::MatrixXd widths2(1, 3);

    terminals2 << 3.0, 2.0, 2.0, 0.0, 0.0, -1.0;
    vectors2 << 0.0, -1.0, 1.0, 1.0, 1.0, 0.0;
    widths2 << 1.0, 1.0, 0.99;

    MeshParametrizationData single_branch1{1, widths1, terminals1, vectors1};
    MeshParametrizationData single_branch2{1, widths2, terminals2, vectors2};

    Eigen::MatrixXd poly_points1 = MeshParametrization::polynomialPoints(single_branch1);
    Eigen::MatrixXd poly_points2 = MeshParametrization::polynomialPoints(single_branch2);

    bool check = MeshParametrization::intersectionBranches(poly_points1, poly_points2);

    assert(check);

    // Initialize the necessary matrices for the multi-branch and test it
    Eigen::MatrixXd multi_vectors(8,3);
    Eigen::MatrixXd multi_terminals(8, 3);
    Eigen::MatrixXd multi_widths(4, 3);

    multi_widths << 1.0, 1.0, 0.8, 1.01, 1.0, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8;
    multi_terminals << 1.0, 2.0, 3.0, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.0, -1.0, -2.0,
                        0.5, 0.5, 0.5, 1.0, 2.0, 3.0,
                        0.0, -1.0, -2.0, 0.5, 0.5, 0.5;
    multi_vectors << 0.0, 0.0, 0.0, 2.0, -2.0, 2.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 2.0, -2.0, 2.0;

    MeshParametrizationData multi_branch {4, multi_widths, multi_terminals, multi_vectors};

    Eigen::MatrixXd poly_points = MeshParametrization::polynomialPoints(multi_branch);

    Eigen::MatrixXd branch_1 = poly_points.block<4, 3>(0, 0);
    Eigen::MatrixXd branch_2 = poly_points.block<4, 3>(4, 0);

    bool check_2 = MeshParametrization::intersectionBranches(branch_1, branch_2);

    assert(!check_2);
}

// This function tests whether a branch has intersecting lines with itself
void test_selfIntersection() {
    // Initialize the necessary matrices for the single branch and test it
    Eigen::MatrixXd vectors1(2,3);
    Eigen::MatrixXd terminals1(2, 3);
    Eigen::MatrixXd widths1(1, 3);

    terminals1 << 0.0, 1.0, 1.0, 0.0, 0.0, -1.0;
    vectors1 << 0.0, 1.0, 1.0, 1.0, 1.0, 0.0;
    widths1 << 1.0, 1.0, 1.0;

    // Initialize the necessary matrices for the single branch and test it
    Eigen::MatrixXd vectors2(2,3);
    Eigen::MatrixXd terminals2(2, 3);
    Eigen::MatrixXd widths2(1, 3);

    terminals2 << 0.0, 1.0, 1.0, 0.0, 0.0, -1.0;
    vectors2 << 0.0, -1.0, 1.0, 1.0, 1.0, 0.0;
    widths2 << 1.0, 1.0, 1.0;

    MeshParametrizationData single_branch1{1, widths1, terminals1, vectors1};
    MeshParametrizationData single_branch2{1, widths2, terminals2, vectors2};

    Eigen::MatrixXd poly_points1 = MeshParametrization::polynomialPoints(single_branch1);
    Eigen::MatrixXd poly_points2 = MeshParametrization::polynomialPoints(single_branch2);

    bool check1 = MeshParametrization::selfIntersection(poly_points1);
    bool check2 = MeshParametrization::selfIntersection(poly_points2);

    assert(check1);
    assert(!check2);
}

void test_connectionPoints() {
    // Initialize the necessary matrices for the multi branch and test it
    Eigen::MatrixXd multi_vectors(8,3);
    Eigen::MatrixXd multi_terminals(8, 3);
    Eigen::MatrixXd multi_widths(4, 3);

    multi_widths << 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8;
    multi_terminals << 1.0, 2.0, 3.0, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.0, -1.0, -2.0,
                        0.5, 0.5, 0.5, 1.0, 2.0, 3.0,
                        0.0, -1.0, -2.0, 0.5, 0.5, 0.5;
    multi_vectors << 0.0, 0.0, 0.0, 2.0, -2.0, 2.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 2.0, -2.0, 2.0;

    MeshParametrizationData multi_branch {4, multi_widths, multi_terminals, multi_vectors};

    Eigen::MatrixXd multi_points = MeshParametrization::polynomialPoints(multi_branch);

    std::optional<Eigen::VectorXi> connections = MeshParametrization::connectionPoints(multi_branch);

    Eigen::VectorXi correct_connections(8);
    correct_connections << 4, 2, 1, 7, 0, 6, 5, 3;


    assert(correct_connections.isApprox(*connections));

    Eigen::MatrixXd multi_vectors2(8,3);
    Eigen::MatrixXd multi_terminals2(8, 3);
    Eigen::MatrixXd multi_widths2(4, 3);

    multi_widths2 << 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 0.99, 0.9, 0.8;
    multi_terminals2 << 1.0, 2.0, 3.0, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.0, -1.0, -2.0,
                        0.5, 0.5, 0.5, 1.0, 2.0, 3.0,
                        0.0, -1.0, -2.0, 0.5, 0.5, 0.5;
    multi_vectors2 << 0.0, 0.0, 0.0, 2.0, -2.0, 2.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 2.0, -2.0, 2.0;

    MeshParametrizationData multi_branch2 {4, multi_widths2, multi_terminals2, multi_vectors2};

    Eigen::MatrixXd multi_points2 = MeshParametrization::polynomialPoints(multi_branch2);

    std::optional<Eigen::VectorXi> connections2 = MeshParametrization::connectionPoints(multi_branch2);

    assert(!connections2.has_value());
}

void test_generateMesh() {
    // Initialize the necessary matrices for the single branch and test it
    Eigen::MatrixXd single_vectors(2,3);
    Eigen::MatrixXd single_terminals(2, 3);
    Eigen::MatrixXd single_widths(1, 3);

    single_terminals << 0.0, 1.0, 1.0, 0.0, 0.0, -1.0;
    single_vectors << 0.0, 1.0, 1.0, 1.0, 1.0, 0.0;
    single_widths << 1.0, 1.0, 1.0;

    MeshParametrizationData single_branch {1, single_widths, single_terminals, single_vectors};

    MeshParametrization::generateMesh(single_branch, "test_generateMesh_single", 1.0, 2);

    // Initialize the necessary matrices for the multi-branch and test it
    Eigen::MatrixXd multi_vectors(8,3);
    Eigen::MatrixXd multi_terminals(8, 3);
    Eigen::MatrixXd multi_widths(4, 3);

    multi_widths << 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8;
    multi_terminals << 1.0, 2.0, 3.0, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.0, -1.0, -2.0,
                        0.5, 0.5, 0.5, 1.0, 2.0, 3.0,
                        0.0, -1.0, -2.0, 0.5, 0.5, 0.5;
    multi_vectors << 0.0, 0.0, 0.0, 2.0, -2.0, 2.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 2.0, -2.0, 2.0;

    MeshParametrizationData multi_branch {4, multi_widths, multi_terminals, multi_vectors};

    MeshParametrization::generateMesh(multi_branch, "test_generateMesh_multi", 1.0, 1);
}

void test_displacementBC() {

    // Initialize the necessary matrices for the single branch and test it
    Eigen::MatrixXd single_vectors(2,3);
    Eigen::MatrixXd single_terminals(2, 3);
    Eigen::MatrixXd single_widths(1, 3);

    single_terminals << 0.0, 1.0, 1.0, 0.0, 0.0, -1.0;
    single_vectors << 0.0, 1.0, 1.0, 1.0, 1.0, 0.0;
    single_widths << 1.0, 1.0, 1.0;

    MeshParametrizationData single_branch {1, single_widths, single_terminals, single_vectors};

    Eigen::MatrixXd poly_points_single = MeshParametrization::polynomialPoints(single_branch);

    Eigen::MatrixXd displacements_single (4, 2);

    displacements_single << 0.0, 0.0,
                            0.0, -1.0,
                            0.0, 0.0,
                            0.0, 0.0;

    Eigen::Vector2d single_point1;
    Eigen::Vector2d single_point2;
    Eigen::Vector2d single_point3;
    Eigen::Vector2d single_point4;

    single_point1 << 0.75, -1.0;
    single_point2 << 1.25, -1.0;
    single_point3 << 0.0, 0.25;
    single_point4 << 0.5, -1.0;

    Eigen::Vector2d displaced_point1 = MeshParametrization::displacementBC(poly_points_single,
        displacements_single, single_point1) + single_point1;
    Eigen::Vector2d displaced_point2 = MeshParametrization::displacementBC(poly_points_single,
        displacements_single, single_point2) + single_point2;
    Eigen::Vector2d displaced_point3 = MeshParametrization::displacementBC(poly_points_single,
        displacements_single, single_point3) + single_point3;
    Eigen::Vector2d displaced_point4 = MeshParametrization::displacementBC(poly_points_single,
        displacements_single, single_point4) + single_point4;

    Eigen::Vector2d correct_point1, correct_point2, correct_point3, correct_point4;

    correct_point1 << 0.75, -1.25;
    correct_point2 << 1.25, -1.75;
    correct_point3 << 0.0, 0.25;
    correct_point4 << 0.5, -1.0;

    assert(displaced_point1.isApprox(correct_point1) and displaced_point2.isApprox(correct_point2) and
        displaced_point3.isApprox(correct_point3) and displaced_point4.isApprox(correct_point4));

    // Initialize the necessary matrices for the multi-branch and test it
    Eigen::MatrixXd multi_vectors(8,3);
    Eigen::MatrixXd multi_terminals(8, 3);
    Eigen::MatrixXd multi_widths(4, 3);

    multi_widths << 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8;
    multi_terminals << 1.0, 2.0, 3.0, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.0, -1.0, -2.0,
                        0.5, 0.5, 0.5, 1.0, 2.0, 3.0,
                        0.0, -1.0, -2.0, 0.5, 0.5, 0.5;
    multi_vectors << 0.0, 0.0, 0.0, 2.0, -2.0, 2.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    1.5, -1.5, 1.5, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 2.0, -2.0, 2.0;

    MeshParametrizationData multi_branch {4, multi_widths, multi_terminals, multi_vectors};

    Eigen::MatrixXd poly_points_multi = MeshParametrization::polynomialPoints(multi_branch);


    Eigen::MatrixXd displacements_multi(16, 1);
    displacements_multi << 0.1, 0.1, -0.1, -0.1, 0, 0, 0, 0, 0.1, -0.5, 0.1, 0.3, 0, 0, 0, 0;


    Eigen::Vector2d multi_point1, multi_point2, multi_point3, multi_point4, multi_point5, multi_point6;
    multi_point1 << 3, 0.8;
    multi_point2 << 3, 0.1;
    multi_point3 << 0.7, -2;
    multi_point4 << 0.1, 3;
    multi_point5 << 0.8, 3;
    multi_point6 << -2, 0.5;

    Eigen::Vector2d displaced_multi1 = MeshParametrization::displacementBC(poly_points_multi.block<4, 3>(0, 0),
        displacements_multi.block<4, 1>(0, 0), multi_point1) + multi_point1;
    Eigen::Vector2d displaced_multi2 = MeshParametrization::displacementBC(poly_points_multi.block<4, 3>(0, 0),
        displacements_multi.block<4, 1>(0, 0), multi_point2) + multi_point2;
    Eigen::Vector2d displaced_multi3 = MeshParametrization::displacementBC(poly_points_multi.block<4, 3>(4, 0),
        displacements_multi.block<4, 1>(4, 0), multi_point3) + multi_point3;
    Eigen::Vector2d displaced_multi4 = MeshParametrization::displacementBC(poly_points_multi.block<4, 3>(8, 0),
        displacements_multi.block<4, 1>(8, 0), multi_point4) + multi_point4;
    Eigen::Vector2d displaced_multi5 = MeshParametrization::displacementBC(poly_points_multi.block<4, 3>(8, 0),
        displacements_multi.block<4, 1>(8, 0), multi_point5) + multi_point5;
    Eigen::Vector2d displaced_multi6 = MeshParametrization::displacementBC(poly_points_multi.block<4, 3>(12, 0),
        displacements_multi.block<4, 1>(12, 0), multi_point6) + multi_point6;

    Eigen::Vector2d correct_multi1, correct_multi2, correct_multi3, correct_multi4, correct_multi5, correct_multi6;

    correct_multi1 << 3.075, 0.875;
    correct_multi2 << 2.9, 0;
    correct_multi3 << 0.7, -2;
    correct_multi4 << 0.2, 3.3;
    correct_multi5 << 0.9, 2.6;
    correct_multi6 << -2, 0.5;

    assert(displaced_multi1.isApprox(correct_multi1) and displaced_multi2.isApprox(correct_multi2) and
        displaced_multi3.isApprox(correct_multi3) and displaced_multi4.isApprox(correct_multi4) and
        displaced_multi5.isApprox(correct_multi5) and displaced_multi6.isApprox(correct_multi6));
}

void test_fixFlaggedSolutionComponentsLE() {

    Eigen::Matrix<double, 8, 8> A_eigen;
    A_eigen << 1.07777e+07, 3.2967e+06, -6.6568e+06, 2.47253e+06, -5.64243e+06, -4.12088e+06, 1.52156e+06, -1.64835e+06,
         3.2967e+06, 2.60249e+07, 1.64835e+06, -2.45826e+07, -4.12088e+06, -1.53265e+07, -824176, 1.38842e+07,
         -6.6568e+06, 1.64835e+06, 1.48986e+07, -7.41758e+06, 1.52156e+06, -824176, -9.76331e+06, 6.59341e+06,
         2.47253e+06, -2.45826e+07, -7.41758e+06, 2.74672e+07, -1.64835e+06, 1.38842e+07, 6.59341e+06, -1.67688e+07,
         -5.64243e+06, -4.12088e+06, 1.52156e+06, -1.64835e+06, 1.59129e+07, 2.47253e+06, -1.17921e+07, 3.2967e+06,
         -4.12088e+06, -1.53265e+07, -824176, 1.38842e+07, 2.47253e+06, 3.67234e+07, 2.47253e+06, -3.52811e+07,
         1.52156e+06, -824176, -9.76331e+06, 6.59341e+06, -1.17921e+07, 2.47253e+06, 2.00338e+07, -8.24176e+06,
         -1.64835e+06, 1.38842e+07, 6.59341e+06, -1.67688e+07, 3.2967e+06, -3.52811e+07, -8.24176e+06, 3.81657e+07;

    lf::assemble::COOMatrix<double> A(8, 8);
    for (int i = 0; i < A_eigen.rows(); ++i) {
        for (int j = 0; j < A_eigen.cols(); ++j) {
            A.AddToEntry(i, j, A_eigen(i, j));
        }
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> phi(8);
    phi.setZero();

    std::vector<std::pair<bool, Eigen::Vector2d>> select_vals;

    Eigen::Vector2d zeros = Eigen::Vector2d::Zero();
    Eigen::Vector2d displacement;
    displacement << 1e-6, 1e-6;

    select_vals.emplace_back(true, displacement);
    select_vals.emplace_back(true, zeros);
    select_vals.emplace_back(false, zeros);
    select_vals.emplace_back(false, zeros);

    MeshParametrization::fixFlaggedSolutionComponentsLE(select_vals, A, phi);

    Eigen::SparseMatrix A_crs = A.makeSparse();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_crs);
    Eigen::VectorXd sol_vec = solver.solve(phi) * 1e6;

    Eigen::VectorXd correct_solution(8);

    correct_solution << 1, 1, 0, 0, 0.342138, 2.3984, 0.700897, 2.01833;

    assert((correct_solution - sol_vec).norm() < 1e-5);

}

void test_elasticRegion() {

    Eigen::MatrixXd test0_stresses (4, 3);
    test0_stresses << -42.0205,  18.5063, -12.5328,  28.4571,
                    -22.9815, -4.82341, -5.64255,  6.65441,
                    2.55292,  1.09229, -45.4656, -46.4547;

    double yield_strength = 1000000;

    bool check = MeshParametrization::elasticRegion(test0_stresses, yield_strength);

    assert(check);
}

void test_displacementEnergy() {
    calculationParams calc_params {1000000000, 200000000000, 0.3, 0.01, 2};

    // Initialize the necessary matrices for the single branch and test it, this is a beam with fixed left end
    Eigen::MatrixXd single_vectors(2,3);
    Eigen::MatrixXd single_terminals(2, 3);
    Eigen::MatrixXd single_widths(1, 3);

    single_terminals << 0.0, 0.5, 1.0, 0.0, 0.0, 0.0;
    single_vectors << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0; // All point upward
    single_widths << 0.1, 0.1, 0.1;

    MeshParametrizationData single_branch {1, single_widths, single_terminals, single_vectors};


    // Fixed on the left end, and a displacement of 1mm in the x-direction on the right end
    Eigen::MatrixXd displacement_BCs(4, 2);

    displacement_BCs << 0.0, 0.001,
                        0.0, 0.00,
                        0.0, 0.001,
                        0.0, 0.00;

    std::pair<bool, double> energy = MeshParametrization::displacementEnergy(single_branch,
        displacement_BCs, calc_params);

    std::cout << "Is it in the linear elastic region? " << energy.first << "\nValue of energy: " << energy.second << "\n";

    // Initialize the necessary matrices for the single branch and test it, this is a beam with fixed left end
    Eigen::MatrixXd single_vectors_2(2,3);
    Eigen::MatrixXd single_terminals_2(2, 3);
    Eigen::MatrixXd single_widths_2(1, 3);

    single_terminals_2 << 0.0, 0.5, 1.0, 0.0, -0.025, -0.05;
    single_vectors_2 << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0; // All point upward
    single_widths_2 << 0.2, 0.15, 0.1;

    MeshParametrizationData single_branch_2 {1, single_widths_2, single_terminals_2, single_vectors_2};

    energy = MeshParametrization::displacementEnergy(single_branch_2,
        displacement_BCs, calc_params);

    std::cout << "Is it in the linear elastic region? " << energy.first << "\nValue of energy: " << energy.second << "\n";
}


int main()
{
    test_checkVectorLengths();
    test_normalizeVectors();
    test_polynomialPoints();
    test_angleVectors();
    test_intersectionVectors();
    test_intersectionBranches();
    test_selfIntersection();
    test_connectionPoints();
    test_generateMesh();
    test_displacementBC();
    test_fixFlaggedSolutionComponentsLE();
    test_elasticRegion();
    test_displacementEnergy();

    return 0;
}
//
// Created by Liam Curtis on 2024-06-09.
//

#ifndef METALFOAMS_MESH_PARAMETRIZATION_H
#define METALFOAMS_MESH_PARAMETRIZATION_H

#include <lf/uscalfe/uscalfe.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <gmsh.h>

// This file will contain the structure of the primary class I will use to hold the lower dimensional mesh
// The class will consist of terminal points (3 per "branch"), a width for each branch, and a vector for the direction of each terminal point
// It will also contain helper functions to ensure that a given parametrization is valid
//      e.g. branches don't overlap anywhere except for at the center, the branches share points at the center
// It will also contain a helper function returning the necessary points in order to create the mesh using the Gmsh API

// TODO: IDEA, when splitting up the overall mesh into its constituent components, create a graph that represents the
//  "distance" to other parts of the mesh, and I can repeatedly check if this distance is respected, linked nodes may
//  have the compatibility condition between their _terminals, "non-linked but close" nodes will be checked for overlap
//  because it is localized this can be checked at all times, maybe?

// TODO: Make sure that the meshing algorithm has the entire center point if branches meet

// TODO: In general, when the NN is looking for an optimal solution, fix the amount by which points can move so it stays linearly elastic
// TODO: This class will likely include the energy difference calculator of two meshes

// TODO: For the NN algorithm, ensure that the multi-branch case compatibility conditions are held when using gradient descent

// TODO: Make sure that _widths is not an alterable parameter in the NN algorithm

// TODO: Ensure that during creation of each NN section, that the vector is approximately orthogonal to the mesh path

using Tuple3i = std::tuple<int, int, int>;
using TupleMap = std::map<Tuple3i, Tuple3i>;

namespace MeshParam {

    class MeshParametrization {

    public:
        //These are public as the training of the NN will change the values of the _terminals (maybe _widths as well???)
        // numBranches is the number of branches in our little "cut-out" of the overall mesh
        // _widths contains the width of each branch, #_widths == numBranches
        // _terminals contain the center terminal points of each branch, and each branch requires 3 terminal points
        // The width in row 1 of _widths matches the width of the of the first terminal (i.e. the first two rows)
        // to be defined, to make it quadratic, shape of _terminals will be 2 * numbranches x 3 for x and y coords
        // Shape of _vectors is also 2 * numbranches x 2 (each branch requires two _vectors)
        const int numBranches;
        const Eigen::VectorXd _widths;
        Eigen::MatrixXd _terminals;
        Eigen::MatrixXd _vectors;

        //Empty constructor is not allowed, the parametrization must be present
        MeshParametrization() = delete;

        //Constructor I intend to use, it verifies that the parametrization sent in is valid
        MeshParametrization(int num, Eigen::VectorXd widths, Eigen::MatrixXd terminals, Eigen::MatrixXd vectors) :
                numBranches {num}, _widths {widths}, _terminals{terminals}, _vectors{vectors}
        {
            // Ensure the dimensions match up correctly
            LF_ASSERT_MSG(num == widths.size(), "Number of branches is not equal to the amount of _widths given in MeshParam");
            LF_ASSERT_MSG(terminals.rows() / 2 == num and terminals.cols() == 3, "Size of _terminals is invalid in MeshParam");
            LF_ASSERT_MSG(vectors.rows() / 2 == num and vectors.cols() == 3, "Size of _vectors is invalid in MeshParam");
            // TODO: Make sure this is returning a bool or delete it from the constructor
            LF_ASSERT_MSG(meshParamValidator(num, widths, terminals, vectors), "Given MeshParam is invalid according to validator");
        }

        // Function to ensure that a given parametrization does not overlap on itself and that compatibility conditions
        // holds if there are multiple branches
        bool meshParamValidator(int num, Eigen::VectorXd &width, Eigen::MatrixXd &terminal, Eigen::MatrixXd &vector);

        // Function to generate the polynomial points and ensure they are in the correct local ordering
        Eigen::MatrixXd polynomialPoints(const Eigen::VectorXd &widths, Eigen::MatrixXd &terminals, Eigen::MatrixXd &vectors);

        //This function takes in a matrix of vectors for the parametrization and changes them to unit vectors
        void normalizeVectors(Eigen::MatrixXd &vectors);

        // This function takes in the vectors for a parametrization throws an exception is there is too much curvature
        // TODO: Consider putting a constraint on the minimum distance between two terminal points and the angle
        //  between vectors as the spline may start to diverge from the spline in the mesh otherwise
        void angleVectors(int numBranch, Eigen::MatrixXd &poly_points);

        // This function takes in the vectors and the width of a parametrization and ensures that the line segments
        // implied by the vectors do not overlap, as this would break the constant width assumption
        void intersectionVectors(int numBranch, Eigen::MatrixXd &poly_points);

        // TODO: implement this function
        // The purpose of this function is to take the points defining two branches and making sure that their lines
        // do not intersect
        bool intersectionBranches(Eigen::MatrixXd &poly_points, int first, int second);

        // TODO: Declare a function that takes in two parametrizations and returns their energy difference (do this first of ones below)
        // The purpose of this function is to take in two parametrizations and return their energy difference
        double energyDifference(MeshParametrization &first, MeshParametrization &second);

        // TODO: Declare a function that takes in two parametrizations and ensures they are in the linear elastic region
        // TODO: Consider building a function that takes in two parametrizations and returns a matrix of displacements,
        //  this can then be used to calculate the energy and check if we're in the linear elastic region
        // The purpose of this function is to take in two parametrizations and check the second is outside of the
        // linear elastic region, at which point our entire model will break down
        bool elasticRegion(MeshParametrization &first, MeshParametrization &second);

        // TODO: Declare a function that takes in a parametrization and creates/returns a mesh
        void generateMesh(MeshParametrization &parametrization, const std::string &mesh_name);
    };


    // TODO: Double check implementation of this
    // This function takes in a matrix of direction vectors for the parametrization and returns the normalized unit
    // vectors for that parametrization
    void MeshParametrization::normalizeVectors(Eigen::MatrixXd &vectors) {
        LF_ASSERT_MSG(vectors.rows() % 2 != 0 or vectors.rows() == 0 or vectors.cols() != 3,
                      "Shape of vectors does not allow for a correct parametrization in call to normalizeVectors");

        int num_branches = int(vectors.rows()) / 2;
        double eps = 1e-7;

        // Iterate through all branches
        for (int branch = 0; branch < num_branches; ++branch) {
            // Calculate norms
            auto norms = vectors.block<2, 3>(branch * 2, 0).colwise().norm();

            LF_ASSERT_MSG(norms.any() < eps, "Norm of vectors approaching zero in call to NormalizeVectors");

            // Make each set of vectors unit vectors
            vectors.block<2, 3>(branch * 2, 0) = vectors.block<2, 3>(branch * 2, 0).array().rowwise() / norms.array();
        }
    }


    // TODO: Double check that this implementation is correct
    // This method takes in _terminals, _widths, and _vectors, and outputs the polynomials points for each of them,
    // this assumes that the parametrization is already "correct" (unit vectors)
    // so only spits out polynomial points in correct order, each width corresponds to a set of 3 terminal points
    // Methodology:
    //  1. Normalize the vectors
    //  2. Calculate poly_points
    //  3. Check ordering of poly_points
    Eigen::MatrixXd MeshParametrization::polynomialPoints(const Eigen::VectorXd &widths, Eigen::MatrixXd &terminals, Eigen::MatrixXd &vectors) {

        // Normalize the vectors, this checks to make sure that vectors has an acceptable shape
        normalizeVectors(vectors);
        int num_branches = int(vectors.rows()) / 2;

        Eigen::MatrixXd poly_points(num_branches * 4, 3);
        poly_points.setZero();
        // TODO: Double check implementation of this
        // Initialize the polynomial points for each branch
        for (int branch = 0; branch < num_branches; branch++) {
            //Initialize the points on the "first side" of the branch
            poly_points.block<2, 3>(2 * branch, 0) = terminals.block<2, 3>(2 * branch, 0)
                    + widths(branch) * vectors.block<2, 3>(2 * branch, 0) / 2;

            //Initialize the points on the "second side" of the branch
            poly_points.block<2, 3>(2 * branch + 2, 0) = terminals.block<2, 3>(2 * branch, 0)
                                                     - widths(branch) * vectors.block<2, 3>(2 * branch, 0) / 2;
        }

        // Initialize useful variables for ordering the poly_points matrix
        double eps = 1e-5;
        Eigen::Matrix<double, 2, 2> direction_of_terminals;
        //Only require one column, if not parallel, we switch, if it is parallel, we're good
        Eigen::Matrix<double, 2, 1> dir_polynomial_points;

        // TODO: Double check implementation of this for loop, read through it again
        // TODO: Change the implementation to intersection in interval rather than parallel checking
        // This for loop goes through and makes sure that the polynomial points are ordered correctly for each branch,
        // avoiding overlapping polynomials
        for (int branch = 0; branch < num_branches; branch++) {
            // We only need to check twice with the assumption that the first terminal is correct
            // Obtain direction from between terminals 2-1, and 3-2
            direction_of_terminals.block<2, 1>(branch * 2, 0) = terminals.block<2, 1>(branch * 2, 1)
                                                                - terminals.block<2, 1>(branch * 2, 0);
            direction_of_terminals.block<2, 1>(branch * 2, 1) = terminals.block<2, 1>(branch * 2, 2)
                                                                - terminals.block<2, 1>(branch * 2, 1);

            // For both direction vectors, check whether the poly_points' vector lines up with terminals' vector
            for (int j = 0; j < 2; j++) {
                dir_polynomial_points = poly_points.block<2, 1>(branch * 4, j + 1)
                        - poly_points.block<2, 1>(branch * 4, j);

                double parallel_check = abs(dir_polynomial_points.dot(direction_of_terminals.block<2, 1>(branch * 2, j)))
                        / (dir_polynomial_points.norm() * direction_of_terminals.block<2, 1>(branch * 2, j).norm());

                //If parallel, we're good and can move on, else we need to switch the points and recheck
                if (abs(parallel_check - 1) > eps) {
                    poly_points.block<2, 1>(branch * 4, j + 1).swap(
                            poly_points.block<2, 1>(branch * 4 + 2, j + 1));

                    //Check if parallel again
                    dir_polynomial_points = poly_points.block<2, 1>(branch * 4, j + 1)
                                            - poly_points.block<2, 1>(branch * 4, j);

                    parallel_check = abs(dir_polynomial_points.dot(direction_of_terminals.block<2, 1>(branch * 2, j)))
                                     / (dir_polynomial_points.norm() * direction_of_terminals.block<2, 1>(branch * 2, j).norm());

                    LF_ASSERT_MSG(abs(parallel_check - 1) > eps, "Switched polynomial points and still not parallel, "
                                                                 "check implementation of polynomialPoints");
                }
                else {
                    continue;
                }
            }
        }
        return poly_points;
    }

    // TODO: Double check the implementation of this function
    void MeshParametrization::angleVectors(int numBranch, Eigen::MatrixXd &poly_points) {

        //Declare matrices I will use to check the angles between vectors
        Eigen::MatrixXd line_segments(2 * numBranch, 3);
        Eigen::MatrixXd terminal_vectors(2 * numBranch, 3);
        Eigen::MatrixXd dot_products(numBranch, 3);
        Eigen::MatrixXd norms(numBranch, 3);
        Eigen::MatrixXd angles(numBranch, 3);
        Eigen::MatrixXd cross_products(numBranch * 2, 2);

        // Because poly_points are well-defined on a per-polynomial basis, all the vectors either point inward or
        // outward, so we can check the angles using cross product and dot products
        // TODO: Implement the situation where there could be crossover, which we can check with cross products of of vectors
        //  between polynomial points, of which we only need to check one, which verifies all possibilities, nvm

        for (int branch = 0; branch < numBranch; ++branch) {
            // Line segments created by each terminal point, width, and vector, these are essentially vectors
            line_segments.block<2, 3>(2 * branch, 0) = poly_points.block<2, 3>(4 * branch + 2, 0) -
                    poly_points.block<2, 3>(4 * branch, 0);

            norms.block<1, 3>(branch, 0) = line_segments.block<2, 3>(2 * branch, 0).colwise().norm();

            // dot_products will contain the dot products between vecs 1 and 2, 1 and 3, and 2 and 3, in that order
            dot_products(branch, 0) = line_segments.block<2, 1>(2 * branch, 0).dot(
                    line_segments.block<2, 1>(2 * branch, 1));
            dot_products(branch, 1) = line_segments.block<2, 1>(2 * branch, 0).dot(
                    line_segments.block<2, 1>(2 * branch, 2));
            dot_products(branch, 2) = line_segments.block<2, 1>(2 * branch, 1).dot(
                    line_segments.block<2, 1>(2 * branch, 2));

            // Handle floating points errors with respect to the angle and calculate the angles
            // Angles will contain angles between vecs 1 and 2, 1 and 3, and 2 and 3, in that order
            angles(branch, 0) = std::acos(std::max(-1.0, std::min(1.0, dot_products(branch, 0) /
                    (norms(branch, 0) * norms(branch, 1)))));
            angles(branch, 0) = std::acos(std::max(-1.0, std::min(1.0, dot_products(branch, 1) /
                                                                       (norms(branch, 0) * norms(branch, 2)))));
            angles(branch, 0) = std::acos(std::max(-1.0, std::min(1.0, dot_products(branch, 2) /
                                                                       (norms(branch, 1) * norms(branch, 2)))));

            // Initialize cross products of each of 3 line_segment vectors
            cross_products(2 * branch, 0) = line_segments(2 * branch, 0) * line_segments(2 * branch + 1, 1)
                    - line_segments(2 * branch, 1) * line_segments(2 * branch + 1, 0);
            cross_products(2 * branch, 1) = line_segments(2 * branch, 0) * line_segments(2 * branch + 1, 2)
                                        - line_segments(2 * branch, 2) * line_segments(2 * branch + 1, 0);

            // Initialize vectors of each of 3 "terminal point" vectors
            // TODO: Consider adding vectors between terminal points to see how much curvature we gain, though it is
            //  likely already covered for with parallel check in polynomial_points and the above logic
            // TODO: Delete the following if unused
            terminal_vectors.block<2, 1>(2 * branch, 0) = poly_points.block<2, 1>(4 * branch, 1) -
                    poly_points.block<2, 1>(4 * branch, 0);
            terminal_vectors.block<2, 1>(2 * branch, 0) = poly_points.block<2, 1>(4 * branch, 2) -
                                                          poly_points.block<2, 1>(4 * branch, 1);
            terminal_vectors.block<2, 1>(2 * branch, 0) = poly_points.block<2, 1>(4 * branch, 0) -
                                                          poly_points.block<2, 1>(4 * branch, 2);


            // If the cross products change sign, then we have a cubic function, which a quadratic can't approximate
            if ((cross_products(branch, 0) < 0 and cross_products(branch, 1) > 0) or
            (cross_products(branch, 0) > 0 and cross_products(branch, 1) < 0)) {
                std::cout << "The cross products change sign with the following polynomial: \n" << poly_points << std::endl;
                LF_ASSERT_MSG(true, "Invalid polynomial angles");
            }
            // If the total angles sum up to more than pi we throw an exception as well
            else if (angles(branch, 0) + angles(branch, 1) >= M_PI) {
                std::cout << "The angles add up to more than 180 degrees" << std::endl;
                LF_ASSERT_MSG(true, "Invalid polynomial angles");
            }

        }
    }

    // TODO: Implement this function
    // TODO: Double check the implementation of this function
    // The purpose of this function is to make sure the vectors used to create the polynomial points do not overlap
    // , as this will create invalid curvature, and violates the constant width assumption
    void MeshParametrization::intersectionVectors(int numBranch, Eigen::MatrixXd &poly_points) {

        // Declare required matrices to check for overlap
        Eigen::MatrixXd line_segments(2 * numBranch, 3);
        Eigen::MatrixXd determinants(numBranch, 3);
        Eigen::MatrixXd parameters(numBranch * 3, 2);
        Eigen::Matrix<bool, Eigen::Dynamic, 3> columnCheck(numBranch, 3);

        // Iterate through branches
        for (int branch = 0; branch < numBranch; ++branch) {

            // Initialize the value of the vectors created by the polynomial points
            line_segments.block<2, 3>(2 * branch, 0) = poly_points.block<2, 3>(4 * branch + 2, 0) -
                                                       poly_points.block<2, 3>(4 * branch, 0);

            // Calculate the cross products of these vectors, if 0, they are parallel and therefore do not intersect
            // Cross products are between 1 and 2, 2 and 3, and 1 and 3, in that order
            determinants(numBranch, 0) = line_segments(2 * branch, 0) * line_segments(2 * branch + 1, 1) -
                                                    line_segments(2 * branch, 1) * line_segments(2 * branch + 1, 0);
            determinants(numBranch, 1) = line_segments(2 * branch, 1) * line_segments(2 * branch + 1, 2) -
                                                     line_segments(2 * branch, 2) * line_segments(2 * branch + 1, 1);
            determinants(numBranch, 2) = line_segments(2 * branch, 0) * line_segments(2 * branch + 1, 2) -
                                                     line_segments(2 * branch, 2) * line_segments(2 * branch + 1, 0);

            // TODO: delete these comments after this function is tested
//            double dx1 = p2.x - p1.x;
//            double dy1 = p2.y - p1.y;
//            double dx2 = p4.x - p3.x;
//            double dy2 = p4.y - p3.y;
//            double t = ((p3.x - p1.x) * dy2 - (p3.y - p1.y) * dx2) / det;
//            double u = -((p1.x - p3.x) * dy1 - (p1.y - p3.y) * dx1) / det;
            // TODO: Double check indexing of this through test cases on overlapping vectors (all possible combinations)
            // TODO: Some determinants can be zero, need to check this when dividing
            parameters(numBranch, 0) = ((poly_points(4 * branch, 1) - poly_points(4 * branch, 0)) * line_segments(2 * branch + 1, 1)
                                        - (poly_points(4 * branch + 1, 1) - poly_points(4 * branch + 1, 0)) * line_segments(2 * branch, 1)) /
                                        determinants(numBranch, 0);

            parameters(numBranch, 1) = -((poly_points(4 * branch, 0) - poly_points(4 * branch, 1)) * line_segments(2 * branch + 1, 0)
                                        - (poly_points(4 * branch + 1, 0) - poly_points(4 * branch + 1, 1)) * line_segments(2 * branch, 0)) /
                                       determinants(numBranch, 0);

            parameters(numBranch + 1, 0) = ((poly_points(4 * branch, 2) - poly_points(4 * branch, 1)) * line_segments(2 * branch + 1, 2)
                                            - (poly_points(4 * branch + 1, 2) - poly_points(4 * branch + 1, 1)) * line_segments(2 * branch, 2)) /
                                           determinants(numBranch, 1);

            parameters(numBranch + 1, 1) = -((poly_points(4 * branch, 1) - poly_points(4 * branch, 2)) * line_segments(2 * branch + 1, 1)
                                             - (poly_points(4 * branch + 1, 1) - poly_points(4 * branch + 1, 2)) * line_segments(2 * branch, 1)) /
                                           determinants(numBranch, 1);

            parameters(numBranch + 2, 0) = ((poly_points(4 * branch, 2) - poly_points(4 * branch, 0)) * line_segments(2 * branch + 1, 2)
                                            - (poly_points(4 * branch + 1, 2) - poly_points(4 * branch + 1, 0)) * line_segments(2 * branch, 2)) /
                                           determinants(numBranch, 2);

            parameters(numBranch + 2, 1) = -((poly_points(4 * branch, 0) - poly_points(4 * branch, 2)) * line_segments(2 * branch + 1, 0)
                                             - (poly_points(4 * branch + 1, 0) - poly_points(4 * branch + 1, 2)) * line_segments(2 * branch, 0)) /
                                           determinants(numBranch, 2);

            columnCheck.block<1, 3>(numBranch, 0) = (parameters.array() >= 0).colwise().all() &&
                    (parameters.array() <= 1).colwise().all();

            if (columnCheck.any()) {
                std::cout << "The branch with the following points contains intersecting vectors: " << poly_points << std::endl;
                LF_ASSERT_MSG(true, "A branch was found with overlapping vectors, curvature assumption invalid");
            }

        }
    }

    // TODO: double check this implementation
    // The purpose of this function is to take in two branches and check if any of the linear lines between their
    // polynomial points overlap, it returns true if there is no intersection
    bool MeshParametrization::intersectionBranches(Eigen::MatrixXd &poly_points, int first, int second) {

        Eigen::Matrix<double, 2, 2> points1;
        Eigen::Matrix<double, 2, 2> points2;
        Eigen::Matrix<double, 2, 2> diff;
        double cross, t, u;
        double eps = 1e-7;
        double bound = 1 - eps;

        // I have 8 total line segments, and I need to check if any of the first 4 intersect with any of the other 4
        for (int i = 0; i < 4; ++i) {

            points1 = poly_points.block<2, 2>(first * 4 + (i / 2) * 2, i % 2);

            for (int j = 0; j < 4; ++j) {

                points2 = poly_points.block<2, 2>(second * 4 + (j / 2) * 2, j % 2);
                diff.col(0) = points1.col(1) - points1.col(0);
                diff.col(1) = points2.col(1) - points2.col(0);
                cross = diff.determinant();
                t = ((points2(0, 0) - points1(0, 0)) * diff(1, 1) -
                        (points2(1, 0) - points1(1, 0)) * diff(0, 1)) / cross;
                u = ((points2(0, 0) - points1(0, 0)) * diff(1, 0) -
                     (points2(1, 0) - points1(1, 0)) * diff(0, 0)) / cross;

                if (t > eps and t < bound and u > eps and u < bound) {
                    return false;
                }
            }
        }
        return true;
    }

    // TODO: Implement the ability to handle sideways-U's
    // TODO: Consider forcing the _vectors to stay in a specific position depending on how NN moves them
    // TODO: Might wanna check that the _widths are the exact same, though the algorithm can force this
    // TODO: Might wanna check that the 3 terminal aren't extremely close together
    // TODO: Might wanna check that the terminal and _vectors don't add up in a way where poly_points overlap
    // The purpose of this function is to take in one parametrization, and make sure it is not overlapping on itself
    // For multi branch situations, it ensures the implied compatibility condition of the branch is held, I will
    // assume that there is only one center, otherwise the complexity skyrockets
    // Methodology:
    //  1. Determine the branch case
    //  2. Normalize the vectors
    //  3. Check compatibility conditions between terminals are held in multi-branch case
    //  4. Check that there is no overlap between branches
    // TODO: Consider creating an extra method that solely checks the one branch case, so you can split the implementation of this
    // TODO: Consider creating a method that takes two branches (or a branch + center) to check if they overlap in a bad way
    // TODO: Instead of throwing Lf_assert messages, consider just returning false and printing some info instead
    bool MeshParametrization::meshParamValidator(int num, Eigen::VectorXd &width, Eigen::MatrixXd &terminal, Eigen::MatrixXd &vector) {


        // TODO: In the 1 branch case, I just need to make sure that this does not fold over itself
        // TODO: In the 1 branch case, I also need to make sure there is not an insane curvature, which I can check through
        //  vector orientations, I believe I don't want the concavity to change before we reach the point
        // TODO: Consider making a mini-method that does this just for the 1-branch case, so that the multi-branch case
        //  can call it multiple times
        // The vectors form two line segments essentially, this case will look for the acute angle between them, as
        // anything more and there is too much curvature
        // It will also make sure the two line segments do not overlap, as this violates the width condition
        if (num == 1) {
            // Note: the calls to these functions will throw an exception if the parametrization is invalid and
            // will also print the reason behind the exception
            normalizeVectors(vector);
            Eigen::MatrixXd poly_points = polynomialPoints(width, terminal, vector);
            angleVectors(num, poly_points);
            intersectionVectors(num, poly_points);
        }

        // TODO In the 3+ branch case, I need to create an algorithm that makes sure that all branches meet up at the center,
        //  and that their widths/vectors allow a center
        // TODO: I also need to check all the individual branches as in num == 1 case
        // TODO: I also need to make sure that the branches don't overlap with each other, e.g. the centers point outwards
        //  and that branches don't touch their neighbouring branches (how to check the second part)
        else if (num >= 3) {

            // Declare parameters that will be useful for validating the parametrization
            double eps = 1e-7;
            normalizeVectors(vector);
            Eigen::MatrixXd poly_points = polynomialPoints(width, terminal, vector);

            // topoMapping is a mapping of (branch #, top/bot, own side) -> (branch connection #, top/bot, connec side)
            // , the purpose is to quickly find the points that match in a 3+ branch case
            // top == 0, bot == 1, side == 0 --> left in matrix, side == 1 --> right in matrix
            // TODO: Check if I actually need this topology variable, or if having next only suffices
            TupleMap topology;
            bool check = false;
            Eigen::VectorXd pairs(4);
            Tuple3i next;


            // Call angleVectors and intersection vectors to make sure there is not high curvature of the branches
            angleVectors(num, poly_points);
            intersectionVectors(num, poly_points);

            // Iterate through branches to find a point that is equal to our first point
            for (int attempt = 0; attempt < 2; attempt ++) {
                for (int branch = 1; branch < num; branch++) {
                    pairs(0) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch, 0)).norm();
                    pairs(1) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch + 2, 0)).norm();
                    pairs(2) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch, 2)).norm();
                    pairs(3) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch + 2, 2)).norm();
                    if (pairs.any() < eps) {
                        check = true;
                    }
                    // If the points map, we create the bidirectional mappings and choose the opposite side to fin the next point
                    if (pairs(0) < eps) {
                        topology[std::make_tuple(0, 0, attempt)] = std::make_tuple(branch, 0, 0);
                        topology[std::make_tuple(branch, 0, 0)] = std::make_tuple(0, 0, attempt);
                        next = std::make_tuple(branch, 1, 0);
                        break;
                    }
                    else if (pairs(1) < eps) {
                        topology[std::make_tuple(0, 0, attempt)] = std::make_tuple(branch, 1, 0);
                        topology[std::make_tuple(branch, 1, 0)] = std::make_tuple(0, 0, attempt);
                        next = std::make_tuple(branch, 0, 0);
                        break;
                    }
                    else if (pairs(2) < eps) {
                        topology[std::make_tuple(0, 0, attempt)] = std::make_tuple(branch, 0, 1);
                        topology[std::make_tuple(branch, 0, 1)] = std::make_tuple(0, 0, attempt);
                        next = std::make_tuple(branch, 1, 1);
                        break;
                    }
                    else if (pairs(3) < eps) {
                        topology[std::make_tuple(0, 0, attempt)] = std::make_tuple(branch, 1, 1);
                        topology[std::make_tuple(branch, 1, 1)] = std::make_tuple(0, 0, attempt);
                        next = std::make_tuple(branch, 0, 1);
                        break;
                    }
                }
                // If we haven't found an equal point, we keep searching
                if (!check) {
                    continue;
                }
                break;
            }
            LF_ASSERT_MSG(!check, "Have not found a point that matches another point in a 3+ branch param");

            // Now we have found points that match, we continue the loop for the remaining points
            for (int point = 1; point < num; ++point) {

                bool overall_check = false;
                check = false;
                int cur_branch = std::get<0>(next);
                int pos = std::get<1>(next) == 0 ? 0 : 2;
                int side = std::get<2>(next) == 0 ? 0 : 2;

                // We have next point to check on all branches, so we should loop through all branches except for our own
                for (int branch = 0; branch < num; ++branch) {

                    // Skip over branch if it's our current one
                    if (cur_branch == branch) {
                        continue;
                    }

                    // Now we can check all points on this branch to see which one is equal
                    pairs(0) = (poly_points.block<2, 1>(4 * cur_branch + 2 * pos, side * 2)
                                - poly_points.block<2, 1>(4 * branch, 0)).norm();
                    pairs(1) = (poly_points.block<2, 1>(4 * cur_branch + 2 * pos, side * 2)
                                - poly_points.block<2, 1>(4 * branch + 2, 0)).norm();
                    pairs(2) = (poly_points.block<2, 1>(4 * cur_branch + 2 * pos, side * 2)
                                - poly_points.block<2, 1>(4 * branch, 2)).norm();
                    pairs(3) = (poly_points.block<2, 1>(4 * cur_branch + 2 * pos, side * 2)
                                - poly_points.block<2, 1>(4 * branch + 2, 2)).norm();

                    if (pairs.any() < eps) {
                        check = true;
                    }
                    // If the points map to each other, we create the bidirectional mappings and choose the
                    // opposite side to find the next point
                    if (pairs(0) < eps) {
                        topology[next] = std::make_tuple(branch, 0, 0);
                        topology[std::make_tuple(branch, 0, 0)] = next;
                        next = std::make_tuple(branch, 1, 0);
                        break;
                    } else if (pairs(1) < eps) {
                        topology[next] = std::make_tuple(branch, 1, 0);
                        topology[std::make_tuple(branch, 1, 0)] = next;
                        next = std::make_tuple(branch, 0, 0);
                        break;
                    } else if (pairs(2) < eps) {
                        topology[next] = std::make_tuple(branch, 0, 1);
                        topology[std::make_tuple(branch, 0, 1)] = next;
                        next = std::make_tuple(branch, 1, 1);
                        break;
                    } else if (pairs(3) < eps) {
                        topology[next] = std::make_tuple(branch, 1, 1);
                        topology[std::make_tuple(branch, 1, 1)] = next;
                        next = std::make_tuple(branch, 0, 1);
                        break;
                    }
                    if (branch == 0 and check) {
                        overall_check = true;
                    }
                    if (!check) {
                        continue;
                    }
                    break;
                }
                // We have circled back completely if we find branch 0 again
                if (overall_check) {
                    break;
                }
                // If we reach here, we have not circled back and there is a problem in the connections
                // TODO: Perhaps print the polynomial points to be able to check the values
                std::cout << "The polynomial points for the failed 3+ branch param are: \n" << poly_points << std::endl;
                LF_ASSERT_MSG(true, "Branches do not form a complete circle shape, check parametrization");
                }

                // TODO: Now must check that no points point inwards and no close lines overlap using the topology, and
                //  this case is actually handled by just checking for intersecting lines
                // TODO: To check for intersection lines, decide whether to send to a mini method between two branches

            for (int i = 0; i < num; ++i) {
                for (int j = i; j < num; ++j) {
                    if(!intersectionBranches(poly_points, i, j)) {
                        std::cout << "Branches " << i << " and " << j << "of the following polynomial points overlap: \n"
                        << poly_points << std::endl;
                        LF_ASSERT_MSG(true, "The branches of a multi-branch parametrization overlap");
                    }
                }
            }
        }

        // A center can't exist with just 2 branches, and any other number outside of 1 or 3+ doesn't make sense
        else {
            return false;
        }
    }




    // TODO: implement this function
    // The purpose of this function is to take in a parametrization and create a mesh with the given mesh_name,
    // this will allow for the calculation of the energy difference between close by parametrizations
    void MeshParametrization::generateMesh(MeshParametrization &parametrization, const std::string &mesh_name) {

        // TODO: Consider putting Bezier curves through the points in order to better approximate a quadratic curve

        Eigen::MatrixXd poly_points = polynomialPoints(parametrization._widths, parametrization._terminals,
                                                       parametrization._vectors);
        // I'm choosing not to add a model name due to possible repeats, only initializing gmsh
        gmsh::initialize();
        // TODO: Decide on how to pick the size of the meshSize (once material and overall size is determined) on adding points
        // In this case, there are just 6 points to add, 2 linear lines, 2 splines, and a surface with all curves
        if (parametrization.numBranches == 1) {

            // Add the points defined by poly_points
            int p1 = gmsh::model::geo::addPoint(poly_points(0, 0), poly_points(1, 0), 0.0, 0);
            int p2 = gmsh::model::geo::addPoint(poly_points(0, 1), poly_points(1, 1), 0.0, 0);
            int p3 = gmsh::model::geo::addPoint(poly_points(0, 2), poly_points(1, 2), 0.0, 0);
            int p4 = gmsh::model::geo::addPoint(poly_points(2, 0), poly_points(3, 0), 0.0, 0);
            int p5 = gmsh::model::geo::addPoint(poly_points(2, 1), poly_points(3, 1), 0.0, 0);
            int p6 = gmsh::model::geo::addPoint(poly_points(2, 2), poly_points(3, 2), 0.0, 0);

            // Create our two splines and two lines, the points are ordered such that a curve loop is easy to define
            int l1 = gmsh::model::geo::addLine(p1, p4);
            int l2 = gmsh::model::geo::addLine(p6, p3);
            int s1 = gmsh::model::geo::addSpline({p3, p2, p1});
            int s2 = gmsh::model::geo::addSpline({p4, p5, p6});

            // Add a curve loop and create a surface
            int c1 = gmsh::model::geo::addCurveLoop({l1, s2, l2, s1});
            int surf1 = gmsh::model::geo::addPlaneSurface({c1});

            // TODO: Decide on adding physical groups once I decide how to incorporate BCs


        }

        else if (parametrization.numBranches >= 3) {

        }

        else {
            LF_ASSERT_MSG(true, "Parametrization was a number of branches other than 1 or 3+, invalid");
            return;
        }

        gmsh::model::geo::synchronize();

        // Set the order to 2 and generate the 2D mesh
        gmsh::option::setNumber("Mesh.ElementOrder", 2);
        gmsh::model::mesh::generate(2);
        // TODO: fix this path because due to Clion this file points to the cmake-build-debug folder
        gmsh::write("meshes/" + mesh_name);
    }



}

#endif //METALFOAMS_MESH_PARAMETRIZATION_H

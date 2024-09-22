//
// Created by Liam Curtis on 2024-09-09.
//

#include "../include/mesh_parametrization.h"

// TODO: fix comments to be more descriptive so any first-time reader can easily understand
// TODO: adjust any asserts where the statement is always true
// TODO: within meshParamValidator perhaps, in the multi-branch case, make sure we go from "in" to out when it comes to the terminal orderings

// TODO: Double check implementation of this
// TODO: See how to make sure this normalization stays correct after the NN continues to update these
// This function takes in a matrix of direction vectors for the parametrization and returns the normalized unit
// vectors for that parametrization, the expected matrix
void MeshParametrization::normalizeVectors (Eigen::MatrixXd &vectors) {
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

// TODO: test this function
// This method takes in _terminals, _widths, and _vectors, and outputs the polynomials points for each of them,
// this assumes that the parametrization is already "correct" (unit vectors)
// so only spits out polynomial points in correct order, each width corresponds to a set of 3 terminal points
// Methodology:
//  1. Normalize the vectors
//  2. Calculate poly_points
//  3. Check ordering of poly_points
// TODO: IDEA: when taking in a mesh, have some processing that determines min and max "length" + "width" of branches for future training
// TODO: This implementation will have to change if widths becomes a matrix
// TODO: Make sure that the ordering of multi-branch parts goes from "in" to "out" w.r.t the center
Eigen::MatrixXd MeshParametrization::polynomialPoints (MeshParametrizationData &param) {

    // Normalize the vectors, this checks to make sure that vectors has an acceptable shape
    normalizeVectors(param.vectors);

    Eigen::MatrixXd poly_points(param.numBranches * 4, 3);
    poly_points.setZero();
    // TODO: Double check implementation of this
    // Initialize the polynomial points for each branch
    for (int branch = 0; branch < param.numBranches; branch++) {
        //Initialize the points on the "first side" of the branch
        poly_points.block<2, 3>(2 * branch, 0) = param.terminals.block<2, 3>(2 * branch, 0)
                                                 + param.widths(branch) * param.vectors.block<2, 3>(2 * branch, 0) / 2;

        //Initialize the points on the "second side" of the branch
        poly_points.block<2, 3>(2 * branch + 2, 0) = param.terminals.block<2, 3>(2 * branch, 0)
                                                     - param.widths(branch) * param.vectors.block<2, 3>(2 * branch, 0) / 2;
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
    for (int branch = 0; branch < param.numBranches; branch++) {
        // We only need to check twice with the assumption that the first terminal is correct
        // Obtain direction from between terminals 2-1, and 3-2
        direction_of_terminals.block<2, 1>(branch * 2, 0) = param.terminals.block<2, 1>(branch * 2, 1)
                                                            - param.terminals.block<2, 1>(branch * 2, 0);
        direction_of_terminals.block<2, 1>(branch * 2, 1) = param.terminals.block<2, 1>(branch * 2, 2)
                                                            - param.terminals.block<2, 1>(branch * 2, 1);

        // For both direction vectors, check whether the poly_points' vector lines up with terminals' vector
        for (int j = 0; j < 2; j++) {
            dir_polynomial_points = poly_points.block<2, 1>(branch * 4, j + 1)
                                    - poly_points.block<2, 1>(branch * 4, j);

            double parallel_check = abs(dir_polynomial_points.dot(direction_of_terminals.block<2, 1>(branch * 2, j)))
                                    / (dir_polynomial_points.norm() *
                                       direction_of_terminals.block<2, 1>(branch * 2, j).norm());

            //If parallel, we're good and can move on, else we need to switch the points and recheck
            if (abs(parallel_check - 1) > eps) {
                poly_points.block<2, 1>(branch * 4, j + 1).swap(
                        poly_points.block<2, 1>(branch * 4 + 2, j + 1));

                //Check if parallel again
                dir_polynomial_points = poly_points.block<2, 1>(branch * 4, j + 1)
                                        - poly_points.block<2, 1>(branch * 4, j);

                parallel_check = abs(dir_polynomial_points.dot(direction_of_terminals.block<2, 1>(branch * 2, j)))
                                 / (dir_polynomial_points.norm() *
                                    direction_of_terminals.block<2, 1>(branch * 2, j).norm());

                LF_ASSERT_MSG(abs(parallel_check - 1) > eps, "Switched polynomial points and still not parallel, "
                                                             "check implementation of polynomialPoints");
            } else {
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

// This method takes in a parametrization and returns a matrix containing the points that overlap using our
// numbering convention
// TODO: Test this method
// TODO: Possibly delete this method, as it may not be needed at all given my map method in generateMesh
// TODO: Possibly change this to reflect the fact that widths could change
Eigen::MatrixXd MeshParametrization::connectionPoints(MeshParametrizationData &multiBranch) {
    // Declare useful help variables
    bool check = false;
    Eigen::VectorXd pairs(4);
    Tuple3i next;
    double eps = 1e-7;

    // TODO: I'm assuming here that we already have a correct parametrization, need to ensure
    //  this before calling generateMesh
    Eigen::MatrixXd poly_points = polynomialPoints(multiBranch);
    int num = multiBranch.numBranches;
    Eigen::MatrixXd connections(num, 2);

    // Iterate through branches to find a point that is equal to our points on the first branch
    for (int attempt = 0; attempt < 2; attempt ++) {
        for (int branch = 1; branch < num; branch++) {
            pairs(0) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch, 0)).norm();
            pairs(1) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch + 2, 0)).norm();
            pairs(2) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch, 2)).norm();
            pairs(3) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch + 2, 2)).norm();

            // We found a matching point
            if (pairs.any() < eps) {
                connections(0, 0) = attempt;
                check = true;
            }
                // If no points match, we continue looking for a matching point on the other side of the branch
            else {
                continue;
            }

            // If the points match, we add this using our numbering convention and initialize next
            if (pairs(0) < eps) {
                connections(0, 1) = 4 * branch;
                next = std::make_tuple(branch, 1, 0);
                break;
            }
            else if (pairs(1) < eps) {
                connections(0, 1) = 4 * branch + 2;
                next = std::make_tuple(branch, 0, 0);
                break;
            }
            else if (pairs(2) < eps) {
                connections(0, 1) = 4 * branch + 3;
                next = std::make_tuple(branch, 1, 1);
                break;
            }
            else if (pairs(3) < eps) {
                connections(0, 1) = 4 * branch + 5;
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

    // Now we have found points that match, we continue the loop for the remaining overlapping points
    for (int point = 1; point < num; ++point) {
        bool overall_check = false;
        check = false;
        int cur_branch = std::get<0>(next);
        int pos = std::get<1>(next) == 0 ? 0 : 1;
        int side = std::get<2>(next) == 0 ? 0 : 1;

        // We have next point to check on all branches, so we should loop through all branches except for our own
        for (int branch = 0; branch < num; ++branch) {
            // Skip over branch if it's our current one, as it can't self-overlap
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

            // We found a match, so we add this point to our connections matrix
            if (pairs.any() < eps) {
                connections(point, 0) = 4 * cur_branch + 3 * pos + 2 * side;
                check = true;
            }

            // If the points overlap, we add the point to the matrix and initialize next
            if (pairs(0) < eps) {
                connections(point, 1) = 4 * branch;
                next = std::make_tuple(branch, 1, 0);
                break;
            } else if (pairs(1) < eps) {
                connections(point, 1) = 4 * branch + 2;
                next = std::make_tuple(branch, 0, 0);
                break;
            } else if (pairs(2) < eps) {
                connections(point, 1) = 4 * branch + 3;
                next = std::make_tuple(branch, 1, 1);
                break;
            } else if (pairs(3) < eps) {
                connections(point, 1) = 4 * branch + 5;
                next = std::make_tuple(branch, 0, 1);
                break;
            }

            // If branch == 0 and check is true, then we have circled back completely, note both
            // conditions must be true for this to hold
            if (branch == 0 and check) {
                overall_check = true;
            }

            // If check is still false, we haven't yet seen a match, so the loop continues
            if (!check) {
                continue;
            }

            // If we reach this far, we have finished, so we exit the loop
            break;
        }

        // We have circled back completely if we find branch 0 again, so we may stop
        if (overall_check) {
            break;
        }

        // If we reach here, we have not circled back and there is a problem in the connections
        // TODO: Perhaps print the polynomial points to be able to check the values (or not)
        std::cout << "The polynomial points for the failed 3+ branch param are: \n" << poly_points << std::endl;
        LF_ASSERT_MSG(true, "Branches do not form a complete circle shape, check parametrization");
    }

    // Return the matrix of connection points
    return connections;
}

// TODO: Implement the ability to handle sideways-U's
// TODO: Consider forcing the _vectors to stay in a specific position depending on how NN moves them
// TODO: Might wanna check that the _widths are the exact same, though the algorithm can force this
// TODO: Might wanna check that the 3 terminal aren't extremely close together
// TODO: Might wanna check that the terminal and _vectors don't add up in a way where poly_points overlap
// TODO: Rethink this implementation knowing how we can now use splines for the sides of an edge, and the main thing to
//  check is if a "part" is in the linear elastic region or not
// TODO: make sure of simple things as well, such as non-negative widths
// TODO: Change this to reflect the fact that widths could be a matrix
bool MeshParametrization::meshParamValidator(MeshParametrizationData &param) {
    // TODO: In the 1 branch case, I just need to make sure that this does not fold over itself
    // TODO: In the 1 branch case, I also need to make sure there is not an insane curvature, which I can check through
    //  vector orientations, I believe I don't want the concavity to change before we reach the point
    // TODO: Consider making a mini-method that does this just for the 1-branch case, so that the multi-branch case
    //  can call it multiple times
    if (param.numBranches == 1) {
        // Note: the calls to these functions will throw an exception if the parametrization is invalid and
        // will also print the reason behind the exception
        normalizeVectors(param.vectors);
        Eigen::MatrixXd poly_points = polynomialPoints(param);
        angleVectors(param.numBranches, poly_points);
        intersectionVectors(param.numBranches, poly_points);
    }
        // TODO In the 3+ branch case, I need to create an algorithm that makes sure that all branches meet up at the center,
        //  and that their widths/vectors allow a center
        // TODO: I also need to check all the individual branches as in num == 1 case
        // TODO: I also need to make sure that the branches don't overlap with each other, e.g. the centers point outwards
        //  and that branches don't touch their neighbouring branches (how to check the second part)
    else if (param.numBranches >= 3) {
        // Declare parameters that will be useful for validating the parametrization
        double eps = 1e-7;
        normalizeVectors(param.vectors);
        Eigen::MatrixXd poly_points = polynomialPoints(param);

        // topoMapping is a mapping of (branch #, top/bot, own side) -> (branch connection #, top/bot, connec side)
        // , the purpose is to quickly find the points that match in a 3+ branch case
        // top == 0, bot == 1, side == 0 --> left in matrix, side == 1 --> right in matrix
        bool check = false;
        Eigen::VectorXd pairs(4);
        Tuple3i next;

        // Call angleVectors and intersection vectors to make sure there is not high curvature of the branches
        angleVectors(param.numBranches, poly_points);
        intersectionVectors(param.numBranches, poly_points);

        // Iterate through branches to find a point that is equal to our first point
        for (int attempt = 0; attempt < 2; attempt ++) {
            for (int branch = 1; branch < param.numBranches; branch++) {
                pairs(0) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch, 0)).norm();
                pairs(1) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch + 2, 0)).norm();
                pairs(2) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch, 2)).norm();
                pairs(3) = (poly_points.block<2, 1>(0, attempt * 2) - poly_points.block<2, 1>(4 * branch + 2, 2)).norm();
                if (pairs.any() < eps) {
                    check = true;
                }
                else {
                    continue;
                }
                // If the points map, we create the bidirectional mappings and choose the opposite side to fin the next point
                if (pairs(0) < eps) {
                    next = std::make_tuple(branch, 1, 0);
                    break;
                }
                else if (pairs(1) < eps) {
                    next = std::make_tuple(branch, 0, 0);
                    break;
                }
                else if (pairs(2) < eps) {
                    next = std::make_tuple(branch, 1, 1);
                    break;
                }
                else if (pairs(3) < eps) {
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
        for (int point = 1; point < param.numBranches; ++point) {
            bool overall_check = false;
            check = false;
            int cur_branch = std::get<0>(next);
            int pos = std::get<1>(next) == 0 ? 0 : 1;
            int side = std::get<2>(next) == 0 ? 0 : 1;

            // We have next point to check on all branches, so we should loop through all branches except for our own
            for (int branch = 0; branch < param.numBranches; ++branch) {
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
                    next = std::make_tuple(branch, 1, 0);
                    break;
                } else if (pairs(1) < eps) {
                    next = std::make_tuple(branch, 0, 0);
                    break;
                } else if (pairs(2) < eps) {
                    next = std::make_tuple(branch, 1, 1);
                    break;
                } else if (pairs(3) < eps) {
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

        for (int i = 0; i < param.numBranches; ++i) {
            for (int j = i; j < param.numBranches; ++j) {
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
    return true;
}


// TODO: Test this tolerance by making sure the strings actualy end up equal
std::string MeshParametrization::getPointKey(double x, double y, double z, double tolerance = 1e-6) {
    return std::to_string(std::round(x / tolerance)) + "_" +
           std::to_string(std::round(y / tolerance)) + "_" +
           std::to_string(std::round(z / tolerance));
}

// The purpose of this function is to take in a parametrization and create a mesh with the given mesh_name, this will
// allow for the necessary finite element calculations
// TODO: test this function
// TODO: very likely I will reset the gmsh model I am using so that this doesn't add points on the model we already opened
// TODO: Change this to reflect the fact that widths would be a matrix
// TODO: Decide on how to pick the size of the meshSize (once material and overall size is determined) on adding points
//  and adjust the mesh size in the following commands
void MeshParametrization::generateMesh(MeshParametrizationData &parametrization, const std::string &mesh_name) {
    Eigen::MatrixXd poly_points = polynomialPoints(parametrization);

    // TODO: Decide if I want to add a model name when initializing a mesh here to make sure there is no overlap
    // I'm choosing not to add a model name due to possible repeats, only initializing gmsh
    gmsh::initialize();

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

        // I add physical groups here on the lines, which is where I intend to incorporate displacement BCs
        gmsh::model::geo::addPhysicalGroup(1, {l1}, 1);
        gmsh::model::geo::addPhysicalGroup(1, {l2}, 2);
    }

    // In this case, the number of total points to add is highly variable
    else if (parametrization.numBranches >= 3) {

        // Initialize matrix that will hold the points that are connected, this will be the basis for centerLines
        Eigen::MatrixXd connectedPoints = connectionPoints(parametrization);

        // The purpose of centerLines is to hold the tags of lines that are in the center of this multi-branch
        std::vector<std::pair<int, std::pair<int, int>>> centerLines(parametrization.numBranches);
        int count = 0;

        // pointMap will hold the point -> pointTag mapping for this particular mesh
        std::map<std::string, int> pointMap;

        // Declare a functor that returns the tag of a point depending on whether it exists
        auto getOrCreatePoint = [&](double x, double y, double z, double meshSize) -> std::tuple<int, bool> {
            std::string key = getPointKey(x, y, z);
            auto it = pointMap.find(key);
            if (it != pointMap.end()) {
                return std::make_tuple(it->second, true);
            } else {
                int tag = gmsh::model::geo::addPoint(x, y, z, meshSize);
                pointMap[key] = tag;
                return std::make_tuple(tag, false);
            }
        };

        // Start a loop to add each branch to the mesh
        for (int branch = 0; branch < parametrization.numBranches; ++branch) {
            // This vector will hold the "names" of the points we add to the mesh
            std::vector<int> points;
            std::vector<bool> midPoints (6, false);

            for (int point = 0; point < 6; ++point) {
                int polyPoint = branch * 6 + point;

                // Declare indices we'll access in poly points to add the points
                int row = branch * 4 + (point / 3) * 2;
                int col = point % 3;

                // Either get the point tag or add a new point to our mesh and return the tag
                auto newPoint = getOrCreatePoint(poly_points(row, col), poly_points(row + 1, col), 0.0, 0);
                points.push_back(std::get<0>(newPoint));

                if (std::get<1>(newPoint)) {
                    midPoints[point] = true;
                }
            }

            // Create our two splines and two lines, using the points vector
            int l1 = gmsh::model::geo::addLine(points[0], points[3]);
            int l2 = gmsh::model::geo::addLine(points[5], points[2]);
            int s1 = gmsh::model::geo::addSpline({points[2], points[1], points[0]});
            int s2 = gmsh::model::geo::addSpline({points[3], points[4], points[5]});

            // Add a curve loop and create a surface
            int c1 = gmsh::model::geo::addCurveLoop({l1, s2, l2, s1});
            int surf1 = gmsh::model::geo::addPlaneSurface({c1});

            // If the points we added are duplicates, they are part of one of the center lines
            if (midPoints[3] or midPoints[0]) {
                centerLines[count] = std::pair<int, std::pair<int, int>>(l1, std::pair<int, int>(points[0], points[3]));
                count++;

                // Since we know this line is part of the center line, the opposite side must be a displacement BC line
                gmsh::model::geo::addPhysicalGroup(1, {l2}, branch + 1);
            }
            else if (midPoints[5] or midPoints[2]) {
                centerLines[count] = std::pair<int, std::pair<int, int>>(l2, std::pair<int, int>(points[5], points[2]));
                count++;

                // Since we know this line is part of the center line, the opposite side must be a displacement BC line
                gmsh::model::geo::addPhysicalGroup(1, {l1}, branch + 1);
            }
        }

        // We now need add the center part of the mesh using the initialized values in centerLines
        std::vector<int> lines(parametrization.numBranches);
        lines[0] = centerLines[0].first;

        int next = centerLines[0].second.second;
        int curLine = lines[0];

        // Iterate through the lines in centerLines
        for (int line = 1; line < parametrization.numBranches; ++line) {
            // See which line has the point we're interested in, and we add this to lines
            for (auto & centerLine : centerLines) {
                if (centerLine.first == curLine) {
                    continue;
                }
                else if (centerLine.second.first == next) {
                    lines[line] = centerLine.first;
                    curLine = centerLine.first;
                    next = centerLine.second.second;
                    break;
                }
                else if (centerLine.second.second == next) {
                    lines[line] = -centerLine.first;
                    curLine = centerLine.first;
                    next = centerLine.second.first;
                    break;
                }
            }
        }

        // Add the curve loop and the surface
        int c2 = gmsh::model::geo::addCurveLoop(lines);
        int surf2 = gmsh::model::geo::addPlaneSurface({c2});

        // TODO: Decide on adding the physical group for this center piece of the mesh
        // TODO: Additionally decide if I want to return the branch -> tag mapping, which could make things easier
    }
    else {
        LF_ASSERT_MSG(true, "Parametrization was a number of branches other than 1 or 3+, invalid");
        return;
    }

    gmsh::model::geo::synchronize();

    // Set the order to 2 and generate the 2D mesh
    gmsh::option::setNumber("Mesh.ElementOrder", 2);
    gmsh::model::mesh::generate(2);
    // TODO: Make sure that the below logic is correct wrt to putting the mesh in the right location and naming
    const std::filesystem::path here = __FILE__;
    auto working_dir = here.parent_path().parent_path();
    gmsh::write(working_dir / "meshes" / mesh_name);
    // gmsh::write("meshes/" + mesh_name);
}

// TODO: WORK IN PROGRESS
// TODO: test this function, and also test the functions in LineMapping that I'm using
// TODO: maybe add assert messages to ensure that the branches have the correct size
// This function takes the polynomial points for a branch, the displacement BCs for the branch, a point on one
// of the lines of the parametrization, and returns the corresponding displacement BC for that point
Eigen::Vector2d MeshParametrization::displacementBC (Eigen::MatrixXd branch, Eigen::MatrixXd displacement,
                                                     Eigen::Vector2d point) {
    LF_ASSERT_MSG(branch.rows() == 4, "poly points sent to displacementBC has incorrect number of rows, "
                                      "" << branch.rows());
    LF_ASSERT_MSG(branch.cols() == 3, "poly points sent to displacementBC has incorrect number of columns, "
                                      "" << branch.cols());

    // Note that the shape of branch is 4x3, so use this to generate the two possible lines the point might be on
    Eigen::Vector2d left_pointA = branch.block<2, 1>(0, 0);
    Eigen::Vector2d left_pointB = branch.block<2, 1>(2, 0);

    Eigen::Vector2d right_pointA = branch.block<2, 1>(0, 2);
    Eigen::Vector2d right_pointB = branch.block<2, 1>(2, 2);

    // Using this mapping solely to test which line the point is on
    LineMapping my_lines(left_pointA, left_pointB, right_pointA, right_pointB);
    int line = -1;

    // TODO: Depending on if the ordering of displacement works out, have an assert to see if displacement
    //  actually has the right shape for this
    // Find which line the point is on and adjust the other line to the "displaced" line, so that we have a mapping
    // from old line -> new displaced line
    if (my_lines.isPointOnFirstLine(point)) {
        my_lines.update(left_pointA, left_pointB, left_pointA + displacement.block<2, 1>(0, 0),
                left_pointB + displacement.block<2, 1>(2, 0));
    } else if (my_lines.isPointOnSecondLine(point)) {
        my_lines.update(right_pointA, right_pointB, right_pointA + displacement.block<2, 1>(0, 1),
                right_pointB + displacement.block<2, 1>(2, 1));
    } else {
        LF_ASSERT_MSG(false, "The given point is not on either line in displacementBC");
    }

    // We return the displacement vector for the newly mapped point
    return my_lines.mapPoint(point) - point;
}

// TODO: Test this function
// The purpose of this function is to mimic fixFlaggedSolutionComponents in LehrFEM++ but for the Linear Elasticity
// problem, which is slightly different due to the 2D nature of the problem. i.e. one dofNum leads to two indices (x and y)
void MeshParametrization::fixFlaggedSolutionComponentsLE (std::vector<std::pair<bool, Eigen::Vector2d>> &selectvals,
                                                          lf::assemble::COOMatrix<double> &A,
                                                          Eigen::Matrix<double, Eigen::Dynamic, 1> &phi) {
    // Note here N is technically "2N" since each node has two coordinates
    const lf::assemble::size_type N(A.cols());
    LF_ASSERT_MSG(A.rows() == N, "Matrix must be square!");
    LF_ASSERT_MSG(N == phi.size(),
                  "Mismatch N = " << N << " <-> b.size() = " << phi.size());

    Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_vec(N);

    // Iterate over half of the total N_dofs, as that is how large selectvals is
    for (int k = 0; k < N / 2; ++k) {
        const auto selVal = selectvals[k];

        // Add value to tmp_vec, knowing .second is a Eigen::Vector2d
        if (selVal.first) {
            tmp_vec[2 * k] = selVal.second[0];
            tmp_vec[2 * k + 1] = selVal.second[1];
        } else {
            tmp_vec[2 * k] = 0.0;
            tmp_vec[2 * k + 1] = 0.0;
        }
    }
    A.MatVecMult(-1.0, tmp_vec, phi);

    // Adjust the values in the rhs vector
    for (int k = 0; k < N / 2; ++k) {
        const auto selVal = selectvals[k];
        if (selVal.first) {
            phi[2 * k] = selVal.second[0];
            phi[2 * k + 1] = selVal.second[1];
        }
    }

    // Adjust the values in the stiffness matrix to 0 where selectvals is true
    // TODO: Verify that the following logic is correct
    A.setZero([&selectvals] (lf::assemble::gdof_idx_t i, lf::assemble::gdof_idx_t j) {
        return (selectvals[i / 2].first || selectvals[j / 2].first);
    });

    // TODO: Verify that the following logic is correct
    for (int k = 0; k < N / 2; ++k) {
        const auto selval = selectvals[k];
        if (selval.first) {
            A.AddToEntry(k, k, 1.0);
            A.AddToEntry(k + 1, k + 1, 1.0);
        }
    }
}

// TODO: WORK IN PROGRESS
// TODO: Implement this function
// TODO: Change this function to reflect the fact that we just want displacement vectors as the second thing
// TODO: Before generating the mesh, call meshParamValidator
// TODO: To reduce the overall workload, perhaps use the displacementEnergy function and call this from there
// This function takes in a parametrization and a displacement BC, and uses the corresponding Finite Element
// calculation to see if we are still in the elastic region
bool MeshParametrization::elasticRegion(MeshParametrizationData &first, Eigen::MatrixXd &displacement) {
    // This function should check if the second parametrization is within the linear elastic region of the first
    // You may want to use the displacementEnergy function and set a threshold for the maximum allowed displacement
    // or strain.

    // Placeholder implementation
    return true;
}

// TODO: WORK IN PROGRESS
// TODO: Decide on increasing the signature of the function in order to have extra parameters for accuracy of the calculation
// TODO: Also decide on if I should add a string so I know where to store the mesh
// TODO: Change the signature of this function to accept the necessary parameters such as E, nu, yield point
// TODO: Change this function to reflect the fact that we just want displacement vectors
// TODO: Before generating the mesh, call meshParamValidator
// TODO: Find a way to make sure the displacement ordering lines up well with the poly_points ordering
// TODO: Within this function, use elastic region in order to initialize the bool of the pair to return
// This function takes in a parametrization and a vector of displacements representing boundary conditions and returns
// the energy difference for the finite element calculation
std::pair<bool, double> MeshParametrization::displacementEnergy(MeshParametrizationData &param, Eigen::MatrixXd &displacement) {
    // Generate the mesh we want to perform a FEM calculation on
    // TODO: Decide on a better naming system since files will be created constantly
    generateMesh(param, "energy1");
    auto factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    const std::filesystem::path here = __FILE__;
    auto mesh_file = here.parent_path().parent_path();
    mesh_file += "meshes/energy1.msh";
    lf::io::GmshReader reader(std::move(factory), mesh_file);
    const std::shared_ptr<lf::mesh::Mesh> mesh_ptr = reader.mesh();

    // Initialize the finite element space we will use for our calculation
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);

    // Obtain the polynomial points for the parametrization to build the dirichlet conditions
    auto poly_points = polynomialPoints(param);

    const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs {dofh.NumDofs()};
    const lf::mesh::Mesh &mesh {*mesh_ptr};

    // Declare useful objects in order to initialize our matrices
    lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{mesh_ptr, 2, false};
    std::vector<std::pair<bool, Eigen::Vector2d>> displacementBCs;
    Eigen::Vector2d zeros;
    zeros.setZero();

    // This loop goes through all nodes to correctly initialize bd_flags and displacementBCs, which will be used
    // to mimic fixFlaggedSolutionComponents
    for (lf::assemble::gdof_idx_t dof_num = 0; dof_num < N_dofs; ++dof_num) {
        // TODO: Make sure this still works as node may not be a correct pointer
        const lf::mesh::Entity &node = dofh.Entity(dof_num);

        // If the node has a physical entity number, it is one of the BC nodes for displacement by current design
        // TODO: make sure of this
        if (!reader.PhysicalEntityNr(node).empty()) {
            // This node will have a displacement BC, so bd_flags is set to true
            bd_flags(node) = true;

            // We obtain the tag number, which helps to determine what this BC is and add it to the std::vector
            int tag = reader.PhysicalEntityNr(node)[0];

            Eigen::Vector2d BC = displacementBC(poly_points.block<4, 3>((tag - 1) * 4, 0),
                                                displacement.block((tag - 1) * 4, 0, 4, displacement.cols()),
                                                node.Geometry()->Global(node.Geometry()->RefEl().NodeCoords()));

            displacementBCs.emplace_back(true, BC);
        }
            // Otherwise, it is a node on some other part of the mesh, for which we're not interested
        else {
            // If false, there is a 0 traction BC, but it is set to 0 here in displacement terms for completeness
            displacementBCs.emplace_back(false, zeros);
        }
    }

    // TODO: Figure out what to do about these constants, should they be sent to this function or hard-programmed for aluminum?
    double E = 30000000.0;
    double v = 0.3;

    // Declare matrices and vector that will hold our solution, phi doesn't need to be changed for now
    lf::assemble::COOMatrix<double> A(N_dofs*2, N_dofs*2);
    Eigen::Matrix<double, Eigen::Dynamic, 1> phi(N_dofs*2);
    phi.setZero();

    // Build the stiffness matrix
    // TODO: double check whether this is a planeStrain calculation
    ParametricMatrixComputation::ParametricFEElementMatrix assemble{E, v, false};
    LinearElasticityAssembler::AssembleMatrixLocally(0, dofh, dofh, assemble, A);

    // Send to fixFlaggedSolutionsComponents to fit to the shape mentioned in NUMPDE 2.7.6.15
    fixFlaggedSolutionComponentsLE(displacementBCs, A, phi);

    // Solve the liner system and return the resulting displacement vector
    Eigen::SparseMatrix A_crs = A.makeSparse();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_crs);
    Eigen::VectorXd sol_vec = solver.solve(phi);

    // TODO: Change this return statement to fit the function's purpose using elasticRegion
    return {true, sol_vec.norm()};
}
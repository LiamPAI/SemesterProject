//
// Created by Liam Curtis on 2024-09-09.
//

#include "../include/mesh_parametrization.h"

#include <linear_matrix_computation.h>


// This function ensures the lengths of all vectors are sufficient so we don't run into floating-point errors, returns
// true if all vectors are of sufficient length and false otherwise
bool MeshParametrization::checkVectorLengths(const Eigen::MatrixXd &vectors){

    if (vectors.rows() % 2 != 0 or vectors.cols() != 3){
        return false;
    }
    for (int i = 0; i < vectors.rows() / 2; ++i) {
        auto norms = vectors.block<2, 3>(i * 2, 0).colwise().norm();

        if (!(norms.array() > 1e-7).all()) return false;

    }
    return true;
}

// This function takes in a matrix of direction vectors for the parametrization and updates the values so that the
// vectors are normalized
void MeshParametrization::normalizeVectors (Eigen::MatrixXd &vectors) {
    LF_ASSERT_MSG(vectors.rows() % 2 == 0 and vectors.rows() != 0 and vectors.cols() == 3,
                  "Shape of vectors does not allow for a correct parametrization in call to normalizeVectors");

    const int num_branches = int(vectors.rows()) / 2;

    // Iterate through all branches
    for (int branch = 0; branch < num_branches; ++branch) {
        double eps = 1e-7;
        // Calculate norms
        auto norms = vectors.block<2, 3>(branch * 2, 0).colwise().norm();

        LF_ASSERT_MSG((norms.array() >= eps).all(), "Norm of vectors approaching zero in call to NormalizeVectors");

        // Make each set of vectors unit vectors
        vectors.block<2, 3>(branch * 2, 0) = vectors.block<2, 3>(branch * 2, 0).array().rowwise() / norms.array();
    }
}

// This method takes in a parametrization and outputs the polynomials points for it, this assumes that the
// parametrization is already "correct"
// Methodology:
//  1. Normalize the vectors
//  2. Calculate poly_points
//  3. Check ordering of poly_points to ensure non-overlapping polynomials
Eigen::MatrixXd MeshParametrization::polynomialPoints (MeshParametrizationData &param) {

    // Normalize the vectors, this checks to make sure that vectors has an acceptable shape
    normalizeVectors(param.vectors);
    Eigen::MatrixXd poly_points(param.numBranches * 4, 3);
    poly_points.setZero();

    // Initialize the default polynomial points for each branch, note these are unordered
    for (int branch = 0; branch < param.numBranches; branch++) {

        // Initialize the polynomial points for each of the 3 terminals
        for (int i = 0; i < 3; ++i)
        {
            //Initialize the points on the "first side" of the branch
            poly_points.block<2, 1>(4 * branch, i) = param.terminals.block<2, 1>(2 * branch, i)
                                                     + param.widths(branch, i) * param.vectors.block<2, 1>(2 * branch, i) / 2;

            //Initialize the points on the "second side" of the branch
            poly_points.block<2, 1>(4 * branch + 2, i) = param.terminals.block<2, 1>(2 * branch, i)
                                                         - param.widths(branch, i) * param.vectors.block<2, 1>(2 * branch, i) / 2;
        }
    }

    // This for loop goes through and makes sure that the polynomial points are ordered correctly for each branch,
    // avoiding overlapping polynomials
    for (int branch = 0; branch < param.numBranches; branch++) {

        // For each part of the polynomial, we initialize straight lines from the points, and if they intersect,
        // we switch the points
        for (int i = 0; i < 2; ++i)
        {
            Eigen::Vector2d a = poly_points.block<2, 1>(4 * branch, i);
            Eigen::Vector2d b = poly_points.block<2, 1>(4 * branch, i + 1);
            Eigen::Vector2d c = poly_points.block<2, 1>(4 * branch + 2, i);
            Eigen::Vector2d d = poly_points.block<2, 1>(4 * branch + 2, i + 1);

            LineMapping poly_lines (a, b, c, d);

            // If they intersect, we swap the points at column i + 1, since for our basis we assume the first
            // terminal is correct
            if (poly_lines.linesIntersect())
            {
                poly_points.block<2, 1>(4 * branch, i + 1).swap(
                    poly_points.block<2, 1>(4 * branch + 2, i + 1));
            }
        }
    }
    return poly_points;
}

// This function calculates the angles between the different vectors using the given polynomial points, if any of
// these angles are above 45 degrees, it returns false, as we want reasonable curvature
bool MeshParametrization::angleVectors(MeshParametrizationData &param) {

    // Obtain the polynomial points for the parametrization, note that this normalizes the vectors already
    Eigen::MatrixXd poly_points = polynomialPoints(param);

    for (int branch = 0; branch < param.numBranches; ++branch) {

        // For each part of the polynomial, we initialize straight lines from the points, and we assert that
        // the angle is below 45 degrees
        for (int i = 0; i < 2; i++)
        {
            // Initialize the lines, which will both come using the same terminal
            Eigen::Vector2d a = poly_points.block<2, 1>(4 * branch, i);
            Eigen::Vector2d b = poly_points.block<2, 1>(4 * branch + 2, i);
            Eigen::Vector2d c = poly_points.block<2, 1>(4 * branch, i + 1);
            Eigen::Vector2d d = poly_points.block<2, 1>(4 * branch + 2, i + 1);

            LineMapping poly_lines (a, b, c, d);

            // If the angle is too large, we return false
            if (poly_lines.angleBetweenLines() > 45.001) {
                return false;
            }
        }
    }
    return true;
}

// The purpose of this function is to make sure the vectors used to create the polynomial points do not overlap, as
// this will create invalid curvature, it returns false if vectors overlap
bool MeshParametrization::intersectionVectors(MeshParametrizationData &param) {

    Eigen::MatrixXd poly_points = polynomialPoints(param);

    // Iterate through each branch, and check them individually
    for (int branch = 0; branch < param.numBranches; ++branch) {

        // Initialize the 6 points making up this branch so we can easily create lines with them
        Eigen::Vector2d a = poly_points.block<2, 1>(branch * 4, 0);
        Eigen::Vector2d b = poly_points.block<2, 1>(branch * 4, 1);
        Eigen::Vector2d c = poly_points.block<2, 1>(branch * 4, 2);
        Eigen::Vector2d d = poly_points.block<2, 1>(branch * 4 + 2, 0);
        Eigen::Vector2d e = poly_points.block<2, 1>(branch * 4 + 2, 1);
        Eigen::Vector2d f = poly_points.block<2, 1>(branch * 4 + 2, 2);

        // Initialize the 3 choose 2 number of lines possible, each representing the "vectors"
        LineMapping line1_2 (a, d, b, e);
        LineMapping line1_3 (a, d, c, f);
        LineMapping line2_3 (b, e, c, f);

        // If any of the lines intersect we return false
        if (line1_2.linesIntersect() or line1_3.linesIntersect() or line2_3.linesIntersect()) {
            return false;
        }
    }
    return true;
}

// The purpose of this function is to take in two branches and check if any of the linear lines between their
// polynomial points overlap, it returns true if there is no intersection, false otherwise
bool MeshParametrization::intersectionBranches(const Eigen::MatrixXd &poly_points_1, const Eigen::MatrixXd &poly_points_2) {

    // If the poly_points do not satisfy our shape requirements, we immediately return false
    if (poly_points_1.rows() != 4 or poly_points_1.cols() != 3 or poly_points_2.rows() != 4
        or poly_points_2.cols() != 3) {
        return false;
    }

    // I have 12 total line segments, and I need to check if any of the first 6 intersect with any of the other 6
    for (int i = 0; i < 6; ++i) {
        Eigen::Vector2d a;
        Eigen::Vector2d b;

        // In the first 4 iterations, we check the spline lines, and in the last 2 we check the straight lines
        if (i < 4) {
            a = poly_points_1.block<2, 1>((i / 2) * 2, i % 2);
            b = poly_points_1.block<2, 1>((i / 2) * 2, i % 2 + 1);
        }
        else {
            a = poly_points_1.block<2, 1>(0, (i - 4) * 2);
            b = poly_points_1.block<2, 1>(2, (i - 4) * 2);
        }

        // Iterate through the 4 spline portions of the second parametrization's points
        for (int j = 0; j < 4; ++j) {
            Eigen::Vector2d c = poly_points_2.block<2, 1>((j / 2) * 2, j % 2);
            Eigen::Vector2d d = poly_points_2.block<2, 1>((j / 2) * 2, j % 2 + 1);

            LineMapping lines (a, b, c, d);

            // Since we can use this method for multi-branches, we don't include the ends for intersection
            if (lines.linesIntersectWithoutEnds()) {
                return false;
            }
        }

        // Iterate through the 2 line portions of the second parametrization's points
        for (int j = 0; j < 2; ++j) {
            Eigen::Vector2d c = poly_points_2.block<2, 1>(0, j * 2);
            Eigen::Vector2d d = poly_points_2.block<2, 1>(2, j * 2);

            LineMapping lines(a, b, c, d);

            // Since we can use this method for multi-branches, we don't include the ends for intersection
            if (lines.linesIntersectWithoutEnds()) {
                return false;
            }
        }
    }
    return true;
}

// This function tests whether a branch self-intersects, this can happen with a misalignment of vectors that the
// function polynomialPoints doesn't cover for, returns false if the branch self-intersects
bool MeshParametrization::selfIntersection(const Eigen::MatrixXd &poly_points) {

    // If the poly_points do not satisfy our shape requirements, we immediately return false
    if (poly_points.rows() % 4 != 0 or poly_points.cols() != 3) {
        return false;
    }

    int num_branches = poly_points.rows() / 4;

    for (int branch = 0; branch < num_branches; ++branch) {

        Eigen::MatrixXd current_branch = poly_points.block<4, 3>(branch * 4, 0);

        // We have 3 line pairing types to check
        for (int i = 0; i < 3; ++i) {
            // Declare the 2 points we will be using on the first line
            Eigen::Vector2d a = current_branch.block<2, 1>((i / 2) * 2, i % 2);
            Eigen::Vector2d b = current_branch.block<2, 1>((i / 2) * 2, i % 2 + 1);

            // We iterate through the remaining lines and check if they intersect
            for (int j = i + 1; j < 4; ++j) {
                // Declare the 2 points we will be using on the second line
                Eigen::Vector2d c = current_branch.block<2, 1>((j / 2) * 2, j % 2);
                Eigen::Vector2d d = current_branch.block<2, 1>((j / 2) * 2, j % 2 + 1);

                LineMapping lines (a, b, c, d);

                if (lines.linesIntersectWithoutEnds()) {
                    return false;
                }
            }
        }
    }
    return true;
}

// TODO: Test this function
// This function takes in a matrix representing the points of a parametrization and returns the corresponding
// parametrization object, also ensuring it is "valid"
MeshParametrizationData MeshParametrization::pointToParametrization(const Eigen::MatrixXd &poly_points) {

    // Declare necessary objects for parametrization
    int num_branches = poly_points.rows() / 4;
    Eigen::MatrixXd widths(num_branches, 3);
    Eigen::MatrixXd terminals(num_branches * 2, 3);
    Eigen::MatrixXd vectors(num_branches * 2, 3);

    for (int i = 0; i < num_branches; ++i) {
        widths(i, 0) = (poly_points.block<2, 1>(4 * i, 0) -
            poly_points.block<2, 1>(4 * i + 2, 0)).norm();
        widths(i, 1) = (poly_points.block<2, 1>(4 * i, 1) -
        poly_points.block<2, 1>(4 * i + 2, 1)).norm();
        widths(i, 2) = (poly_points.block<2, 1>(4 * i, 2) -
            poly_points.block<2, 1>(4 * i + 2, 2)).norm();

        terminals.block<2, 1>(2 * i, 0) = (poly_points.block<2, 1>(4 * i, 0) +
            poly_points.block<2, 1>(4 * i + 2, 0)) / 2;
        terminals.block<2, 1>(2 * i, 1) = (poly_points.block<2, 1>(4 * i, 1) +
            poly_points.block<2, 1>(4 * i + 2, 1)) / 2;
        terminals.block<2, 1>(2 * i, 2) = (poly_points.block<2, 1>(4 * i, 2) +
            poly_points.block<2, 1>(4 * i + 2, 2)) / 2;

        vectors.block<2, 1>(2 * i, 0) = (poly_points.block<2, 1>(4 * i, 0) -
            poly_points.block<2, 1>(4 * i + 2, 0)).normalized();
        vectors.block<2, 1>(2 * i, 1) = (poly_points.block<2, 1>(4 * i, 1) -
        poly_points.block<2, 1>(4 * i + 2, 1)).normalized();
        vectors.block<2, 1>(2 * i, 2) = (poly_points.block<2, 1>(4 * i, 2) -
            poly_points.block<2, 1>(4 * i + 2, 2)).normalized();
    }

    MeshParametrizationData parametrization {num_branches, widths, terminals, vectors};
    LF_ASSERT_MSG(meshParamValidator(parametrization), "Invalid polynomial points sent to the function "
    "pointToParametrization, with values \n" << poly_points);
    return parametrization;
}

// This method takes in a parametrization and returns a matrix containing the points that overlap using our
// numbering convention, this assumes that the parametrization is "correct" already in that we go from "in" to out"
// for the multi-branch case, so the only valid overlapping points will be in the first column of poly_points
// If we don't find enough overlapping points for our parametrization, we return std::nullopt
// connections, which is the intended return matrix, has shape (numBranches * 2) by 2, since there are that many
// points that should match, the mapping is bidirectional, e.g. points 0 at index 0 will have value 3 in and index 3
// will have value 0
std::optional<Eigen::VectorXi> MeshParametrization::connectionPoints(MeshParametrizationData &multiBranch) {

    // If this is not a multi-branch parametrization, it is useless to call connectionPoints
    if (multiBranch.numBranches < 3){
        return std::nullopt;
    }

    // Find the matrix of polynomial points and initialize the matrix of connections
    Eigen::MatrixXd poly_points = polynomialPoints(multiBranch);
    Eigen::VectorXi connections = -1 * Eigen::VectorXi::Ones(multiBranch.numBranches * 2);
    int current = 0;

    // There will be a number of connections equal to the number of branches
    for (int connection = 0; connection < multiBranch.numBranches; connection++) {
        bool match = false;

        // Iterate through all branches to find a possible match
        for (int branch = 0; branch < multiBranch.numBranches; branch++) {

            // If we are looking at our current branch, we skip over it
            if (current / 2 == branch) {
                continue;
            }
            double check_1 = (poly_points.block<2, 1>(branch * 4, 0) -
                poly_points.block<2, 1>(current * 2, 0)).norm();
            double check_2 = (poly_points.block<2, 1>(branch * 4 + 2, 0) -
                            poly_points.block<2, 1>(current * 2, 0)).norm();

            // If one of these points match, "current" is switched to the next point we want to find a match for and
            // we update our connections matrix
            if (check_1 < 1e-7){
                match = true;
                if (connections[current] == -1 and connections[branch * 2] == -1) {
                    connections[current] = branch * 2;
                    connections[branch * 2] = current;
                }
                else {
                    match = false;
                    break;
                }
                current = branch * 2 + 1;
            }
            else if (check_2 < 1e-7) {
                match = true;
                if (connections[current] == -1 and connections[branch * 2 + 1] == -1) {
                    connections[current] = branch * 2 + 1;
                    connections[branch * 2 + 1] = current;
                }
                else {
                    match = false;
                    break;
                }
                current = branch * 2;
            }

            // Since we found a match, we can exit this inner loop
            if (match) {
                break;
            }
        }
        // If we haven't found a match, then the parametrization is incorrect, so we return "false"
        if (!match) {
            return std::nullopt;
        }
    }
    // Return the matrix of connection points
    return connections;
}

// This function uses the helper functions above to determine if a parametrization is valid, returns true if
// it is and false otherwise
// Criteria that are checked:
//  1. Splines do not overlap with themselves in a particular branch -> selfIntersection
//  2. The angles between vectors do not exceed 45 degrees, so our curvature is limited -> angleVectors
//  3. For each individual branch, the implied vectors do not overlap with each other -> intersectionVectors
//  4. In multi-branch parametrizations, the center points line up correctly with overlap -> connectionPoints
//  5. In multi-branch parametrizations, the terminals are ordered from the center outwards -> intersectionBranches
//  6. In multi-branch parametrizations, the separate branches do not overlap with each other -> intersectionBranches
bool MeshParametrization::meshParamValidator(MeshParametrizationData &param) {

    // Check that widths are all positive and vectors have non-zero lengths
    if (!(param.widths.array() > 1e-7).all()) return false;
    if (!checkVectorLengths(param.vectors)) return false;

    // For a single branch, we ensure the angles of vectors are not too steep, the vectors don't intersect, and that
    // the branch doesn't intersect with itself
    if (param.numBranches == 1) {
        Eigen::MatrixXd poly_points = polynomialPoints(param);

        // If any of the validating functions return false, then the parametrization is invalid and we return false
        if (!(angleVectors(param) and intersectionVectors(param) and selfIntersection(poly_points))) {
            return false;
        }
    }

    // In the multi-branch case, we ensure the above for all single branches, make sure they form a center, and that
    // the multiple branches don't intersect each other
    else if (param.numBranches >= 3) {
        // We first call connectionPoints to make sure that the points correctly overlap, otherwise return false
        if (connectionPoints(param) == std::nullopt) {
            return false;
        }

        Eigen::MatrixXd poly_points = polynomialPoints(param);

        // We now check each branch individually using the above helper methods to make sure they are all valid
        if (!(angleVectors(param) and intersectionVectors(param) and selfIntersection(poly_points))) {
            return false;
        }

        // We now make sure that none of the branches overlap with any of the others, not counting their center points
        // overlapping, if any intersect, we return false, this ensures that all branches point outward
        for (int i = 0; i < param.numBranches; ++i) {
            for (int j = i + 1; j < param.numBranches; ++j) {
                if(!intersectionBranches(poly_points.block<4, 3>(i * 4, 0),
                    poly_points.block<4, 3>(j * 4, 0))) {
                    return false;
                }
            }
        }
    }

    // A center can't exist with just 2 branches, and any other number outside 1 or 3+ doesn't make sense
    else {
        return false;
    }
    return true;
}

// TODO: Reorganize and comment this function with the new try-catch functionality
// The purpose of this function is to take in a parametrization and create a mesh with the given mesh_name, this will
// allow for the necessary finite element calculations, note that it is assumed the given parametrization is "correct"
void MeshParametrization::generateMesh(MeshParametrizationData &parametrization, const std::string &mesh_name,
    double mesh_size=0.01, int order=1) {

    Eigen::MatrixXd poly_points = polynomialPoints(parametrization);

    // I'm choosing not to add a model name due to possible repeats, only initializing gmsh
    gmsh::initialize();

    gmsh::option::setNumber("General.Verbosity", 2);

    // In this case, there are just 6 points to add, 2 linear lines, 2 splines, and a surface with all curves
    if (parametrization.numBranches == 1) {

        // Add the points defined by poly_points
        int p1 = gmsh::model::geo::addPoint(poly_points(0, 0), poly_points(1, 0), 0.0, mesh_size);
        int p2 = gmsh::model::geo::addPoint(poly_points(0, 1), poly_points(1, 1), 0.0, mesh_size);
        int p3 = gmsh::model::geo::addPoint(poly_points(0, 2), poly_points(1, 2), 0.0, mesh_size);
        int p4 = gmsh::model::geo::addPoint(poly_points(2, 0), poly_points(3, 0), 0.0, mesh_size);
        int p5 = gmsh::model::geo::addPoint(poly_points(2, 1), poly_points(3, 1), 0.0, mesh_size);
        int p6 = gmsh::model::geo::addPoint(poly_points(2, 2), poly_points(3, 2), 0.0, mesh_size);

        // Create our two splines and two lines, the points are ordered such that a curve loop is easy to define
        int l1 = gmsh::model::geo::addLine(p1, p4);
        int l2 = gmsh::model::geo::addLine(p6, p3);
        int s1 = gmsh::model::geo::addSpline({p3, p2, p1});
        int s2 = gmsh::model::geo::addSpline({p4, p5, p6});

        // Add a curve loop and create a surface
        int c1 = gmsh::model::geo::addCurveLoop({l1, s2, l2, s1});
        int pl1 = gmsh::model::geo::addPlaneSurface({c1});

        gmsh::model::geo::synchronize();

        // I add physical groups on the lines, which is where I intend to incorporate displacement BCs
        gmsh::model::addPhysicalGroup(1, {l1}, 1);
        gmsh::model::addPhysicalGroup(1, {l2}, 2);
        gmsh::model::addPhysicalGroup(2, {pl1}, 3);
    }

    // In this case, the number of total points to add is highly variable
    else if (parametrization.numBranches >= 3) {

        // Initialize matrix that will hold the points that are connected, this will be the basis for center_lines
        auto connected_points = connectionPoints(parametrization);

        Eigen::VectorXi connections = -1 * Eigen::VectorXi::Ones(parametrization.numBranches * 2);

        // The purpose of centerLines is to hold the tags of lines that are in the center of this multi-branch
        // It holds the tag of the line followed by the points used to define it, in the right order
        std::vector<std::pair<int, std::pair<int, int>>> center_lines(parametrization.numBranches);

        // Vectors to hold the would be tags of line and surfaces to later add physical grouos
        std::vector<int> line_tags(parametrization.numBranches);
        std::vector<int> surface_tags (parametrization.numBranches + 1);

        // Start a loop to add each branch to the mesh
        for (int branch = 0; branch < parametrization.numBranches; ++branch) {
            // This vector will hold the "names" of the points we add to the mesh
            std::vector<int> points;

            for (int point = 0; point < 6; ++point) {
                // Declare indices we'll access in poly points to add the points
                int row = branch * 4 + (point / 3) * 2;
                int col = point % 3;

                // Check if the point we are adding is one of the center points
                if (col == 0) {
                    int center_point = 2 * branch + point / 3;
                    // If this is true, then we've already "added" this point, so we add the tag of the point
                    // we've already added to the points vector
                    if (connected_points.value()[center_point] < center_point) {
                        points.push_back(connections[center_point]);
                    }
                    else {
                        int new_point = gmsh::model::geo::addPoint(poly_points(row, col),
                            poly_points(row + 1, col), 0.0, mesh_size);
                        points.push_back(new_point);
                        connections[center_point] = new_point;
                        connections[connected_points.value()[center_point]] = new_point;
                    }
                }
                // If we don't add a center point, we can immediately add it
                else {
                    int new_point = gmsh::model::geo::addPoint(poly_points(row, col),
                        poly_points(row + 1, col), 0.0, mesh_size);
                    points.push_back(new_point);
                }
            }
            // Create our two splines and two lines, using the points vector
            int l1 = gmsh::model::geo::addLine(points[0], points[3]);
            int l2 = gmsh::model::geo::addLine(points[5], points[2]);
            int s1 = gmsh::model::geo::addSpline({points[2], points[1], points[0]});
            int s2 = gmsh::model::geo::addSpline({points[3], points[4], points[5]});

            // Add a curve loop and create a surface
            int c1 = gmsh::model::geo::addCurveLoop({l1, s2, l2, s1});
            int pl1 = gmsh::model::geo::addPlaneSurface({c1});

            gmsh::model::geo::synchronize();
            // Add the line that will contain a displacement BC and the center line
            center_lines[branch] = std::pair(l1, std::pair(points[0], points[3]));
            line_tags[branch] = l2;
            surface_tags[branch] = pl1;
        }

        // We now need to add the center part of the mesh using the initialized values in center_lines, but we must
        // determine the correct orientation
        std::vector<int> lines(parametrization.numBranches);
        lines[0] = center_lines[0].first;

        int next_point = center_lines[0].second.second;
        int current_line = lines[0];

        // Iterate through the lines in centerLines
        for (int line = 1; line < parametrization.numBranches; ++line) {
            // See which line has the point we're interested in, and we add this to lines using the correct orientation
            for (auto &center_line : center_lines) {
                if (center_line.first == current_line) {
                    continue;
                }
                if (center_line.second.first == next_point) {
                    lines[line] = center_line.first;
                    current_line = center_line.first;
                    next_point = center_line.second.second;
                    break;
                }
                if (center_line.second.second == next_point) {
                    lines[line] = -center_line.first;
                    current_line = center_line.first;
                    next_point = center_line.second.first;
                    break;
                }
            }
        }
        // Add the curve loop and the surface
        int c2 = gmsh::model::geo::addCurveLoop(lines);
        int pl1 = gmsh::model::geo::addPlaneSurface({c2});

        gmsh::model::geo::synchronize();
        surface_tags[parametrization.numBranches] = pl1;

        // Add the physical groups using our line_tags and surface_tags vectors
        for (int branch = 0; branch < parametrization.numBranches; ++branch) {
            gmsh::model::addPhysicalGroup(1, {line_tags[branch]}, branch + 1);
        }
        for (int branch = 0; branch <= parametrization.numBranches; ++branch) {
            gmsh::model::addPhysicalGroup(2, {surface_tags[branch]}, parametrization.numBranches + branch + 1);
        }
    }

    else {
        LF_ASSERT_MSG(false, "Parametrization was a number of branches other than 1 or 3+, invalid");
        return;
    }

    // Create the paths for the .geo and .msh files
    const std::filesystem::path here = __FILE__;
    auto working_dir = here.parent_path().parent_path();
    auto geo_path = working_dir / "geometries" / (mesh_name + ".geo_unrolled");
    auto mesh_path = working_dir / "meshes" / (mesh_name + ".msh");

    // Set the order to 2 if desired, else default is order 1
    if (order == 2) gmsh::option::setNumber("Mesh.ElementOrder", 2);

    // Ensure the outputted mesh version is Version 2 ASCII
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::option::setNumber("Mesh.Binary", 0);

    gmsh::model::mesh::generate(2);

    // Write the .geo file, containing only geometric entities
    gmsh::write(geo_path.string());

    // Write the .msh file and finalize
    gmsh::write(mesh_path.string());
    gmsh::finalize();
}

// This function takes the polynomial points for a branch, the displacement BCs for the branch, a point on one
// of the lines of the parametrization, and returns the corresponding displacement BC for that point
Eigen::Vector2d MeshParametrization::displacementBC (const Eigen::MatrixXd &branch, const Eigen::MatrixXd &displacement,
                                                     const Eigen::Vector2d &point) {
    LF_ASSERT_MSG(branch.rows() == 4, "poly points sent to displacementBC has incorrect number of rows, "
                                      "" << branch.rows());
    LF_ASSERT_MSG(branch.cols() == 3, "poly points sent to displacementBC has incorrect number of columns, "
                                      "" << branch.cols());
    LF_ASSERT_MSG(displacement.rows() == 4, "displacement sent to displacementBC has incorrect number of rows, "
                                << displacement.rows());

    // Note that the shape of branch is 4x3, so use this to generate the two possible lines the point might be on
    Eigen::Vector2d left_pointA = branch.block<2, 1>(0, 0);
    Eigen::Vector2d left_pointB = branch.block<2, 1>(2, 0);

    Eigen::Vector2d right_pointA = branch.block<2, 1>(0, 2);
    Eigen::Vector2d right_pointB = branch.block<2, 1>(2, 2);

    // Using this mapping solely to test which line the point is on
    LineMapping my_lines(left_pointA, left_pointB, right_pointA, right_pointB);

    // Find which line the point is on and adjust the other line to the "displaced" line, so that we have a mapping
    // from old line -> new displaced line
    if (displacement.cols() == 1)
    {   // In this case, we have a multi-branch param, so our displacement occurs on the "right" of the branch
        if (my_lines.isPointOnSecondLine(point)) {
            my_lines.update(right_pointA, right_pointB, right_pointA + displacement.block<2, 1>(0, 0),
                    right_pointB + displacement.block<2, 1>(2, 0));
        }
        else {
            LF_ASSERT_MSG(false, "The given point is not on either line in displacementBC");
            return Eigen::Vector2d::Zero();
        }
    }
    else
    {   // In this case we have a single-branch param, so we need to see if the point is on the right or left
        if (my_lines.isPointOnFirstLine(point)) {
            my_lines.update(left_pointA, left_pointB, left_pointA + displacement.block<2, 1>(0, 0),
                left_pointB + displacement.block<2, 1>(2, 0));
        }
        else if (my_lines.isPointOnSecondLine(point)) {
            my_lines.update(right_pointA, right_pointB, right_pointA + displacement.block<2, 1>(0, 1),
                    right_pointB + displacement.block<2, 1>(2, 1));
        }
        else {
            LF_ASSERT_MSG(false, "The given point is not on either line in displacementBC");
            return Eigen::Vector2d::Zero();
        }
    }
    // We return the displacement vector for the newly mapped point
    return my_lines.mapPoint(point) - point;
}

// The purpose of this function is to mimic fixFlaggedSolutionComponents in LehrFEM++ but for the Linear Elasticity
// problem, which is slightly different due to the 2D nature of the problem. i.e. one dofNum
// leads to two indices (x and y)
void MeshParametrization::fixFlaggedSolutionComponentsLE (std::vector<std::pair<bool, Eigen::Vector2d>> &select_vals,
                                                          lf::assemble::COOMatrix<double> &A,
                                                          Eigen::Matrix<double, Eigen::Dynamic, 1> &phi) {
    // Note here N is technically "2N" since each node has two coordinates
    const lf::assemble::size_type N(A.cols());
    LF_ASSERT_MSG(A.rows() == N, "Matrix must be square!");
    LF_ASSERT_MSG(N == phi.size(), "Mismatch N = " << N << " <-> b.size() = " << phi.size());

    Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_vec(N);

    // Iterate over half of the total N_dofs, as that is how large selectvals is
    for (int k = 0; k < N / 2; ++k) {
        const auto &sel_val = select_vals[k];

        // Add value to tmp_vec, knowing .second is a Eigen::Vector2d
        if (sel_val.first) {
            tmp_vec[2 * k] = sel_val.second[0];
            tmp_vec[2 * k + 1] = sel_val.second[1];
        } else {
            tmp_vec[2 * k] = 0.0;
            tmp_vec[2 * k + 1] = 0.0;
        }
    }
    A.MatVecMult(-1.0, tmp_vec, phi);

    // Adjust the values in the rhs vector
    for (int k = 0; k < N / 2; ++k) {
        const auto &sel_val = select_vals[k];
        if (sel_val.first) {
            phi[2 * k] = sel_val.second[0];
            phi[2 * k + 1] = sel_val.second[1];
        }
    }

    // Adjust the values in the stiffness matrix to 0 where selectvals is true
    A.setZero([&select_vals] (lf::assemble::gdof_idx_t i, lf::assemble::gdof_idx_t j) {
        return (select_vals[i / 2].first || select_vals[j / 2].first);
    });

    // Adjust diagonal entries to 1 if selectvals is true
    for (int k = 0; k < N / 2; ++k) {
        const auto &sel_val = select_vals[k];
        if (sel_val.first) {
            A.AddToEntry(2 * k, 2 * k, 1.0);
            A.AddToEntry(2 * k + 1, 2 * k + 1, 1.0);
        }
    }
}

// This function takes in the stresses calculated at different quadrature points and checks if our
// material has exited the linear elastic region using the von mises stress
bool MeshParametrization::elasticRegion(const Eigen::MatrixXd &stresses, double yieldStrength) {

    // stresses and strains have the same number of columns, iterate through each column and check our criteria
    for (int i = 0; i < stresses.cols(); i++) {
        // Calculate the von Mises stress
        double sigma_xx = stresses(0, i);
        double sigma_yy = stresses(1, i);
        double sigma_xy = stresses(2, i);

        double von_mises = std::sqrt(sigma_xx*sigma_xx + sigma_yy*sigma_yy
                                    - sigma_xx*sigma_yy + 3*sigma_xy*sigma_xy);

        if (von_mises > yieldStrength) {
            return false;
        }
    }
    return true;
}

// TODO: Add timeout capabilities in case generateMesh goes wrong for some unforeseen reason, and recomment the
//  function once I know it works, mesh generation should never take more than 10s
// This function takes in a parametrization and a vector of displacements representing boundary conditions and returns
// the energy difference for the finite element calculation. Note that displacement has shape (4 * numBranches)
// by 1 if a multi branch, 2 if a single branch
std::pair<bool, double> MeshParametrization::displacementEnergy(MeshParametrizationData &param,
    Eigen::MatrixXd &displacement, const calculationParams &calc_params) {

    generateMesh(param, "displacementEnergy", calc_params.meshSize, calc_params.order);
    auto factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    const std::filesystem::path here = __FILE__;
    auto mesh_file = here.parent_path().parent_path();
    mesh_file += "/meshes/displacementEnergy.msh";
    lf::io::GmshReader reader(std::move(factory), mesh_file);
    const std::shared_ptr<lf::mesh::Mesh> mesh_ptr = reader.mesh();

    // Initialize the finite element space we will use for our calculation, depending on the order given
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space;
    if (calc_params.order == 1){
        fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
    }
    else {
        fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);
    }

    auto poly_points = polynomialPoints(param);
    const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs {dofh.NumDofs()};
    const lf::mesh::Mesh &mesh {*mesh_ptr};

    // Initialize useful objects to set up our boundary conditions
    lf::mesh::utils::CodimMeshDataSet bd_flags{mesh_ptr, 2, false};
    std::vector<std::pair<bool, Eigen::Vector2d>> displacementBCs(N_dofs);
    for (auto& bc : displacementBCs) {
        bc.first = false;
        bc.second = Eigen::Vector2d::Zero();
    }

    // This loop goes through all edges to identify which are on the boundaries we are interested in
    for (const lf::mesh::Entity *edge: mesh.Entities((1))) {

        // If the physical entity vector is non-empty, we
        if (!reader.PhysicalEntityNr(*edge).empty()) {
            int tag = reader.PhysicalEntityNr(*edge)[0];

            // Iterate through the nodes on this edge, and prescribe a displacement BC if we haven't done so already
            for (auto node : edge->SubEntities(1)) {
                if (!bd_flags(*node)) {
                    bd_flags(*node) = true;

                    Eigen::Vector2d BC;
                    if (poly_points.rows() == 4) {
                        BC = displacementBC(poly_points, displacement,
                            node->Geometry()->Global(node->Geometry()->RefEl().NodeCoords()));
                    }
                    else {
                        BC = displacementBC(poly_points.block<4, 3>((tag - 1) * 4, 0),
                            displacement.block((tag - 1) * 4, 0, 4, displacement.cols()),
                            node->Geometry()->Global(node->Geometry()->RefEl().NodeCoords()));
                    }

                    displacementBCs[mesh.Index(*node)] = {true, BC};
                }
            }
        }
    }

    // Declare matrices and vector that will hold our solution, phi doesn't need to be changed for now
    lf::assemble::COOMatrix<double> A(N_dofs*2, N_dofs*2);
    Eigen::Matrix<double, Eigen::Dynamic, 1> phi(N_dofs*2);
    phi.setZero();

    // Build the stiffness matrix
    // TODO: double check whether this is a planeStrain calculation (it is if this is set to true)
    if (calc_params.order == 1) {
        LinearMatrixComputation::LinearFEElementMatrix assemble{calc_params.youngsModulus,
            calc_params.poissonRatio, false};
        LinearElasticityAssembler::AssembleMatrixLocally(0, dofh, dofh, assemble, A);
    }
    else {
        ParametricMatrixComputation::ParametricFEElementMatrix assemble{calc_params.youngsModulus,
            calc_params.poissonRatio, false};
        LinearElasticityAssembler::AssembleMatrixLocally(0, dofh, dofh, assemble, A);
    }

    // Send to fixFlaggedSolutionsComponents to fit to the shape mentioned in NUMPDE 2.7.6.15
    fixFlaggedSolutionComponentsLE(displacementBCs, A, phi);

    // Solve the liner system and return the resulting displacement vector
    Eigen::SparseMatrix A_crs = A.makeSparse();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_crs);
    Eigen::VectorXd sol_vec = solver.solve(phi);

    bool linear_elastic;
    double energy = 0;

    // Calculate the stress and strains to check if we are still in the linear elastic region, and calculate the energy
    if (calc_params.order == 1){
        LinearMatrixComputation::LinearFEElementMatrix assemble{calc_params.youngsModulus,
            calc_params.poissonRatio, false};
        auto stresses_strains = LinearElasticityAssembler::stressStrainLoader(
            mesh_ptr, sol_vec, assemble, calc_params.order);
        energy = LinearElasticityAssembler::energyCalculator(mesh_ptr, sol_vec, assemble, calc_params.order);
        linear_elastic = elasticRegion(std::get<2>(stresses_strains), calc_params.yieldStrength);
    }
    else {
        ParametricMatrixComputation::ParametricFEElementMatrix assemble{calc_params.youngsModulus,
            calc_params.poissonRatio, false};
        auto stresses_strains = LinearElasticityAssembler::stressStrainLoader(
            mesh_ptr, sol_vec, assemble, calc_params.order);
        energy = LinearElasticityAssembler::energyCalculator(mesh_ptr, sol_vec, assemble, calc_params.order);
        linear_elastic = elasticRegion(std::get<2>(stresses_strains), calc_params.yieldStrength);
    }

    return {linear_elastic, energy};
}
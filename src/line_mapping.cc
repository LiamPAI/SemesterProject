//
// Created by Liam Curtis on 2024-09-09.
//

#include "../include/line_mapping.h"
#include <cmath>


// Constructor
LineMapping::LineMapping(Eigen::Vector2d a, Eigen::Vector2d b, Eigen::Vector2d c, Eigen::Vector2d d)
        : leftStart(std::move(a)), leftEnd(std::move(b)), rightStart(std::move(c)), rightEnd(std::move(d)) {
    computeTransformationMatrix();
}

// This function computes the necessary transformation matrix to map points from one line to another, useful for
// calculating displacement boundary conditions
void LineMapping::computeTransformationMatrix() {
    Eigen::Vector2d old_vec = leftEnd - leftStart;
    Eigen::Vector2d new_vec = rightEnd - rightStart;

    // Create a 2x2 matrix that maps the old vector to the new vector
    Eigen::Matrix2d A;
    A.col(0) = old_vec;
    A.col(1) = Eigen::Vector2d(-old_vec.y(), old_vec.x());

    Eigen::Matrix2d B;
    B.col(0) = new_vec;
    B.col(1) = Eigen::Vector2d(-new_vec.y(), new_vec.x());

    Eigen::Matrix2d linear_transform = B * A.inverse();

    // Create the full 3x3 transformation matrix
    transformationMatrix = Eigen::Matrix3d::Identity();
    transformationMatrix.block<2,2>(0,0) = linear_transform;
    transformationMatrix.block<2,1>(0,2) = rightStart - linear_transform * leftStart;
}

// Use the transformation matrix to map the point to the new line
Eigen::Vector2d LineMapping::mapPoint(const Eigen::Vector2d& point) const {
    Eigen::Vector3d homogeneous(point[0], point[1], 1);
    Eigen::Vector3d transformed = transformationMatrix * homogeneous;
    return transformed.head<2>();
}

// Update the lines we map between and recompute the transformation matrix
void LineMapping::update(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                         const Eigen::Vector2d& c, const Eigen::Vector2d& d) {
    leftStart = a;
    leftEnd = b;
    rightStart = c;
    rightEnd = d;
    computeTransformationMatrix();
}

// Calculates the distance of a point to a line, useful to check if the point in question as actually
// on the line we're interested in
double LineMapping::distanceToLine(const Eigen::Vector2d& point, const Eigen::Vector2d& lineStart, const Eigen::Vector2d& lineEnd) const {
    Eigen::Vector2d line = lineEnd - lineStart;
    Eigen::Vector2d point_vector = point - lineStart;
    double line_length_squared = line.squaredNorm();

    // Point-to-point distance if line has zero length
    if (line_length_squared == 0.0) {
        return point_vector.norm();
    }

    // Calculate the projection of pointVector onto the line
    double t = point_vector.dot(line) / line_length_squared;

    if (t < 0.0) {
        return (point - lineStart).norm();
    } else if (t > 1.0) {
        return (point - lineEnd).norm();
    }

    Eigen::Vector2d projection = lineStart + t * line;
    return (point - projection).norm();
}

// Test if the provided point is on the first line, or "left" line
bool LineMapping::isPointOnFirstLine(const Eigen::Vector2d& point, double tolerance) const {
    return distanceToLine(point, leftStart, leftEnd) <= tolerance;
}

// Test if the provided point is on the second line, or "right" line
bool LineMapping::isPointOnSecondLine(const Eigen::Vector2d& point, double tolerance) const {
    return distanceToLine(point, rightStart, rightEnd) <= tolerance;
}

// This function will return a bool, true if the lines intersect along the interval for which they are defined,
// false otherwise
bool LineMapping::linesIntersect() const
{
    // Define the vectors for each line
    Eigen::Vector2d v1 = leftEnd - leftStart;
    Eigen::Vector2d v2 = rightEnd - rightStart;

    // Calculate the determinant to see if they are parallel, in which case we check if they overlap or not
    double det = v1.x() * v2.y() - v1.y() * v2.x();
    if (std::abs(det) < 1e-9) {
        Eigen::Vector2d diff = rightStart - leftStart;
        double cross = diff.x() * v1.y() - diff.y() * v1.x();

        // If the lines are co-linear, we check if they overlap or not
        if (std::abs(cross) < 1e-9) {
            double t0 = diff.dot(v1) / v1.dot(v1);
            double t1 = t0 + v2.dot(v1) / v1.dot(v1);

            if (t0 > t1) std::swap(t0, t1);

            // Check if overlap exists and is not just at endpoints
            return t0 <= 1 && t1 >= 0;
        }
        return false;
    }

    // Calculate parametrization parameters, which vary on the interval [0, 1]
    Eigen::Vector2d diff = rightStart - leftStart;
    double t = (diff.x() * v2.y() - diff.y() * v2.x()) / det;
    double s = (diff.x() * v1.y() - diff.y() * v1.x()) / det;

    // If the parameters are on the interval [0, 1], then the lines intersect
    return (t >= 0 and t <= 1 and s >= 0 and s <= 1);
}

// This function is identical to linesIntersect() but does not count the end points of the line as a
// point of intersection, note this function also counts overlap as an intersection
bool LineMapping::linesIntersectWithoutEnds() const
{
    // Define the vectors for each line
    Eigen::Vector2d v1 = leftEnd - leftStart;
    Eigen::Vector2d v2 = rightEnd - rightStart;

    // Calculate the determinant to see if they are parallel, in which case we check if they overlap or not
    double det = v1.x() * v2.y() - v1.y() * v2.x();
    if (std::abs(det) < 1e-9) {
        Eigen::Vector2d diff = rightStart - leftStart;
        double cross = diff.x() * v1.y() - diff.y() * v1.x();

        // If the lines are co-linear, we check if they overlap or not
        if (std::abs(cross) < 1e-9) {
            double t0 = diff.dot(v1) / v1.dot(v1);
            double t1 = t0 + v2.dot(v1) / v1.dot(v1);

            if (t0 > t1) std::swap(t0, t1);

            // Check if overlap exists and is not just at endpoints
            return t0 < 1 - 1e-8 && t1 > 1e-8 && !(std::abs(t0) < 1e-8 || std::abs(t1 - 1) < 1e-8);
        }
        return false;
    }

    // Calculate parametrization parameters, which vary on the interval [0, 1]
    Eigen::Vector2d diff = rightStart - leftStart;
    double t = (diff.x() * v2.y() - diff.y() * v2.x()) / det;
    double s = (diff.x() * v1.y() - diff.y() * v1.x()) / det;

    // If the parameters are on the interval [0, 1], then the lines intersect
    return (t > 1e-8 and t < 1 - 1e-8 and s > 1e-8 and s < 1 - 1e-8);
}

// This function returns the acute angle between two lines in units of degrees, using the properties of dot product
double LineMapping::angleBetweenLines() const
{
    // Calculate normalized direction vectors, so it is easier to obtain the angle
    Eigen::Vector2d dir1 = (leftEnd - leftStart).normalized();
    Eigen::Vector2d dir2 = (rightEnd - rightStart).normalized();

    // Calculate the dot product of the two vectors
    double dot_product = dir1.dot(dir2);

    // Ensure it is on the interval [-1, 1] to get a proper angle;
    dot_product = std::max(-1.0, std::min(1.0, dot_product));

    // Calculate the angle between the two vectors, in degrees, and return it
    double angle = std::acos(std::abs(dot_product)) * 180.0 / M_PI;

    return angle;
}


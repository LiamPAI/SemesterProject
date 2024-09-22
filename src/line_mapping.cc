//
// Created by Liam Curtis on 2024-09-09.
//

#include "../include/line_mapping.h"
#include <cmath>
#include <utility>

// TODO: Comment all this code with viable descriptions

// Constructor
LineMapping::LineMapping(Eigen::Vector2d a, Eigen::Vector2d b, Eigen::Vector2d c, Eigen::Vector2d d)
        : leftStart(std::move(a)), leftEnd(std::move(b)), rightStart(std::move(c)), rightEnd(std::move(d)) {
    computeTransformationMatrix();
}

// This function computes the necessary transformation matrix to map points from one line to another
void LineMapping::computeTransformationMatrix() {
    // Translate the old line to the origin
    Eigen::Vector2d translationToOrigin = -leftStart;

    // Calculate the angle of rotation
    Eigen::Vector2d oldDir = (leftEnd - leftStart).normalized();
    Eigen::Vector2d newDir = (rightEnd - rightStart).normalized();
    double angle = std::atan2(newDir.y(), newDir.x()) - std::atan2(oldDir.y(), oldDir.x());

    Eigen::Matrix2d rotation;
    rotation << std::cos(angle), -std::sin(angle),
            std::sin(angle),  std::cos(angle);

    // Translate to the new start point
    Eigen::Vector2d translationToNewStart = rightStart;

    // Combine both transformations
    Eigen::Matrix3d T1 = Eigen::Matrix3d::Identity();
    T1.block<2,1>(0,2) = translationToOrigin;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R.block<2,2>(0,0) = rotation;

    Eigen::Matrix3d T2 = Eigen::Matrix3d::Identity();
    T2.block<2,1>(0,2) = translationToNewStart;

    transformationMatrix = T2 * R * T1;
}

// Use the transformation matrix to map the point
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

// Calculates the distance of a point to a line
double LineMapping::distanceToLine(const Eigen::Vector2d& point, const Eigen::Vector2d& lineStart, const Eigen::Vector2d& lineEnd) const {
    Eigen::Vector2d line = lineEnd - lineStart;
    Eigen::Vector2d pointVector = point - lineStart;
    double lineLengthSquared = line.squaredNorm();

    // Point-to-point distance if line has zero length
    if (lineLengthSquared == 0.0) {
        return pointVector.norm();
    }

    // Calculate the projection of pointVector onto the line
    double t = pointVector.dot(line) / lineLengthSquared;

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
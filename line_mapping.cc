//
// Created by Liam Curtis on 2024-09-09.
//

#include "line_mapping.h"
#include <cmath>

// TODO: Comment all this code with viable descriptions

void LineMapping::computeTransformationMatrix() {
    // Step 1: Translate old line to origin
    Eigen::Vector2d translationToOrigin = -oldStart;

    // Step 2: Compute rotation
    Eigen::Vector2d oldDir = (oldEnd - oldStart).normalized();
    Eigen::Vector2d newDir = (newEnd - newStart).normalized();
    double angle = std::atan2(newDir.y(), newDir.x()) - std::atan2(oldDir.y(), oldDir.x());

    Eigen::Matrix2d rotation;
    rotation << std::cos(angle), -std::sin(angle),
            std::sin(angle),  std::cos(angle);

    // Step 3: Translate to new start point
    Eigen::Vector2d translationToNewStart = newStart;

    // Combine all transformations
    Eigen::Matrix3d T1 = Eigen::Matrix3d::Identity();
    T1.block<2,1>(0,2) = translationToOrigin;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R.block<2,2>(0,0) = rotation;

    Eigen::Matrix3d T2 = Eigen::Matrix3d::Identity();
    T2.block<2,1>(0,2) = translationToNewStart;

    transformationMatrix = T2 * R * T1;
}

double LineMapping::distanceToLine(const Eigen::Vector2d& point, const Eigen::Vector2d& lineStart, const Eigen::Vector2d& lineEnd) const {
    Eigen::Vector2d line = lineEnd - lineStart;
    Eigen::Vector2d pointVector = point - lineStart;
    double lineLengthSquared = line.squaredNorm();

    if (lineLengthSquared == 0.0) {
        return pointVector.norm();  // Point-to-point distance if line has zero length
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

LineMapping::LineMapping(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                         const Eigen::Vector2d& c, const Eigen::Vector2d& d)
        : oldStart(a), oldEnd(b), newStart(c), newEnd(d) {
    computeTransformationMatrix();
}

Eigen::Vector2d LineMapping::mapPoint(const Eigen::Vector2d& point) const {
    Eigen::Vector3d homogeneous(point[0], point[1], 1);
    Eigen::Vector3d transformed = transformationMatrix * homogeneous;
    return transformed.head<2>();
}

void LineMapping::update(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                         const Eigen::Vector2d& c, const Eigen::Vector2d& d) {
    oldStart = a;
    oldEnd = b;
    newStart = c;
    newEnd = d;
    computeTransformationMatrix();
}

bool LineMapping::isPointOnOldLine(const Eigen::Vector2d& point, double tolerance) const {
    return distanceToLine(point, oldStart, oldEnd) <= tolerance;
}

bool LineMapping::isPointOnNewLine(const Eigen::Vector2d& point, double tolerance) const {
    return distanceToLine(point, newStart, newEnd) <= tolerance;
}
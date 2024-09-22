//
// Created by Liam Curtis on 2024-09-09.
//

#ifndef METALFOAMS_LINE_MAPPING_H
#define METALFOAMS_LINE_MAPPING_H

#include <Eigen/Core>
#include <Eigen/Dense>

// TODO: Comment all this code with viable descriptions
// TODO: Perhaps add to this code so that it can easily give displacement BCs for splines as well in gmsh
// The purpose of this class is to map points from one line to another line, which serves to provide the displacement
// boundary conditions on a mesh parametrization

class LineMapping {
private:
    Eigen::Vector2d leftStart, leftEnd, rightStart, rightEnd;
    Eigen::Matrix3d transformationMatrix;

    void computeTransformationMatrix();
    double distanceToLine(const Eigen::Vector2d& point, const Eigen::Vector2d& lineStart, const Eigen::Vector2d& lineEnd) const;

public:
    LineMapping(Eigen::Vector2d a, Eigen::Vector2d b, Eigen::Vector2d c, Eigen::Vector2d d);

    Eigen::Vector2d mapPoint(const Eigen::Vector2d& point) const;
    void update(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                const Eigen::Vector2d& c, const Eigen::Vector2d& d);
    bool isPointOnFirstLine(const Eigen::Vector2d& point, double tolerance = 1e-6) const;
    bool isPointOnSecondLine(const Eigen::Vector2d& point, double tolerance = 1e-6) const;
};

#endif //METALFOAMS_LINE_MAPPING_H

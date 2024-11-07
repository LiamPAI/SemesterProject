//
// Created by Liam Curtis on 2024-09-09.
//

#ifndef METALFOAMS_LINE_MAPPING_H
#define METALFOAMS_LINE_MAPPING_H

#include <Eigen/Core>
#include <Eigen/Dense>

// The purpose of this class is to map points from one line to another line, which serves to provide the displacement
// boundary conditions on a mesh parametrization

/**
 * @brief Class for mapping points between two line segments
 * @details Provides functionality to map points from one line segment to another,
 *          find if lines intersect, and analyze geometric relationships between
 *          the segments
 */
class LineMapping {
private:
    Eigen::Vector2d leftStart;  ///< Start point of first line segment
    Eigen::Vector2d leftEnd;    ///< End point of first line segment
    Eigen::Vector2d rightStart; ///< Start point of second line segment
    Eigen::Vector2d rightEnd;   ///< End point of second line segment
    Eigen::Matrix3d transformationMatrix;   ///< Matrix for point transformation

    /**
     * @brief Computes the transformation matrix for point mapping
     * @details Updates the internal transformation matrix based on current line positions
     */
    void computeTransformationMatrix();

    /**
     * @brief Calculates the shortest distance from a point to a line segment
     * @param point Point to check
     * @param lineStart Start point of line segment
     * @param lineEnd End point of line segment
     * @return Shortest distance from point to line segment
     */
    double distanceToLine(const Eigen::Vector2d& point, const Eigen::Vector2d& lineStart, const Eigen::Vector2d& lineEnd) const;

public:
    /**
     * @brief Constructs a line mapping between two line segments
     * @param a Start point of first line
     * @param b End point of first line
     * @param c Start point of second line
     * @param d End point of second line
     */
    LineMapping(Eigen::Vector2d a, Eigen::Vector2d b, Eigen::Vector2d c, Eigen::Vector2d d);

    /**
     * @brief Maps a point from first line segment to second line segment
     * @param point Point to map (should be on first line)
     * @return Mapped point on second line
     */
    Eigen::Vector2d mapPoint(const Eigen::Vector2d& point) const;

    /**
     * @brief Updates the positions of both line segments
     * @param a New start point of first line
     * @param b New end point of first line
     * @param c New start point of second line
     * @param d New end point of second line
     */
    void update(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                const Eigen::Vector2d& c, const Eigen::Vector2d& d);

    /**
     * @brief Checks if a point lies on the first line segment
     * @param point Point to check
     * @param tolerance Distance tolerance for point-line comparison
     * @return true if point lies on first line segment within tolerance
     */
    bool isPointOnFirstLine(const Eigen::Vector2d& point, double tolerance = 1e-6) const;

    /**
     * @brief Checks if a point lies on the second line segment
     * @param point Point to check
     * @param tolerance Distance tolerance for point-line comparison
     * @return true if point lies on second line segment within tolerance
     */
    bool isPointOnSecondLine(const Eigen::Vector2d& point, double tolerance = 1e-6) const;

    /**
      * @brief Checks if the two line segments intersect
      * @return true if line segments intersect (including endpoints)
      */
    bool linesIntersect() const;

    /**
     * @brief Checks if the two line segments intersect excluding endpoints
     * @return true if line segments intersect at non-endpoint locations
     */
    bool linesIntersectWithoutEnds() const;

   /**
    * @brief Calculates the angle between the two line segments
    * @return Angle between lines in degrees
    */
    double angleBetweenLines() const;
};

#endif //METALFOAMS_LINE_MAPPING_H

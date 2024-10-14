//
// Created by Liam Curtis on 2024-08-30.
//

#ifndef METALFOAMS_GRAPH_MESH_H
#define METALFOAMS_GRAPH_MESH_H

#include <lf/uscalfe/uscalfe.h>
#include <map>
#include <mesh_parametrization.h>
#include <data_operations.h>
#include <set>

// The purpose of the following data structure is to store the planned mesh, which is a 2D graph-like mesh, where
// both the edges and the nodes are 2D. The naming system required for this to work is that all "nodes" and "edges"
// are labeled, such that the surrounding curves (or boundary) are labeled "NB0, NB1, .." and "EB0, EB1,...",
// respectively. Additionally, the physical 2D surface as well must be labeled, using the same number as the
// corresponding boundary name. The below functions will also provide some of the training parameters for the NN,
// as well as the compatibility conditions when running the "FEM" calculation on the NN

// This struct is used to give an ordering on the compatibility conditions when optimizing the trained NN to output
// the result from an FEM calculation
struct CompatibilityCondition {
    std::pair<int, int> indices; // Indices of the two parts which rely on each other
    std::pair<int, int> firstLocation; // Location of the points in the first index (num of branch, side)
    std::pair<int, int> secondLocation; // Location of the points in the second index (num of branch, side)

    CompatibilityCondition(std::pair<int, int> inds, std::pair<int, int> firstLoc, std::pair<int, int> secondLoc)
        : indices(std::move(inds)),
        firstLocation(std::move(firstLoc)),
        secondLocation(std::move(secondLoc)) {}
};

// The two structs below are used when generating the compatibility conditions to ensure each condition is unique
struct IntPairHash {
    std::size_t operator()(const std::pair<int, int> &p) const {
        int a = std::min(p.first, p.second);
        int b = std::max(p.first, p.second);
        return ((a + b) * (a + b + 1) / 2) + b;
    }
};

struct IntPairEqual {
    bool operator()(const std::pair<int, int> &left, const std::pair<int, int> &right) const {
        return (left.first == right.first and left.second == right.second) or
        (left.first == right.second and left.second == right.first);
    }
};

// TODO: When actually training the NN, decide if these 4 pairs will end up being enough
// The following struct is used in the method getNNTrainingParams to obtain some of the physical ranges
// we should be training the NN on
struct NNTrainingParams {
    std::pair<double, double> minMaxLength;
    std::pair<double, double> minMaxWidth;
    std::pair<double, double> minMaxWidthDiff;
    std::pair<double, double> minMaxAngleDiff;

    NNTrainingParams(std::pair<double, double> length, std::pair<double, double> width,
        std::pair<double, double> widthDiff, std::pair<double, double> angleDiff)
        : minMaxLength(std::move(length)),
        minMaxWidth(std::move(width)),
        minMaxWidthDiff(std::move(widthDiff)),
        minMaxAngleDiff(std::move(angleDiff)) {}
};

struct Node2D {
    int id;
    std::vector<int> surfaceTags;
    std::vector<int> boundaryTags;
    std::set<int> connectedEdges;
};

struct Edge2D {
    int id;
    std::vector<int> surfaceTags;
    std::vector<int> boundaryTags;
    std::vector<int> connectedNodes;
};

struct MeshPart {
    enum Type {NODE, EDGE} type;
    int id;  // Node ID or Edge ID
    int subId;  // Used for edge parts to distinguish between multiple parts of the same edge
    std::vector<int> surfaceTags;
    std::vector<int> boundaryTags;
    std::vector<std::pair<int, std::pair<double, double>>> connectedEdges;  // Edge ID, (start, end) portions
    std::vector<int> connectedEdgeTags;  // Only used for NODE type, stores tags of connected edges
    std::vector<std::tuple<int, int, std::pair<double, double>>> curveTags; // Curve tag, orientation (1 or -1), and portion
};

struct PartGraph {
    std::vector<MeshPart> parts;
    std::vector<std::vector<size_t>> adjacencyList;
};

class GraphMesh {
private:
    std::map<int, Node2D> nodes;
    std::map<int, Edge2D> edges;

    bool isNode(const std::string& name);
    bool isNodeBoundary(const std::string& name);
    bool isEdge(const std::string& name);
    bool isEdgeBoundary(const std::string& name);
    int extractId(const std::string& name);

public:
    GraphMesh();
    ~GraphMesh();

    void loadMeshFromFile(const std::string& filename);
    void closeMesh();
    void buildGraphFromMesh();

    // getter methods for easy retrieval if needed
    std::set<int> getConnectedEdges(int nodeId);
    std::vector<int> getNodesOfEdge(int edgeId);
    std::set<int> getConnectedNodes(int nodeId);
    std::vector<int> getNodeSurfaces(int nodeId);
    std::vector<int> getNodeBoundary(int nodeId);
    std::vector<int> getEdgeSurfaces(int edgeId);
    std::vector<int> getEdgeBoundary(int edgeId);

    // The functions below help with splitting up the mesh and obtaining the required parametrizations
    PartGraph splitMesh(double targetPartSize, double nodeEdgePortion);
    Eigen::MatrixXd getPartGeometry(const MeshPart &part);

    std::vector<Eigen::MatrixXd> getGeometryPolynomialPoints(const PartGraph &part_graph);
    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXi>> getMeshPolynomialPoints(
        const PartGraph &part_graph, const std::string &mesh_file_name, int order);
    std::vector<MeshParametrizationData> getGeometryParametrizations(const PartGraph &part_graph);
    std::vector<MeshParametrizationData> getMeshParametrizations(const PartGraph &part_graph,
        const std::string &mesh_file_name, int order);
    std::vector<Eigen::MatrixXd> centerPointParametrizations(const std::vector<Eigen::MatrixXd> &point_vector);
    std::vector<MeshParametrizationData> centerMeshParametrizations(
        const std::vector<MeshParametrizationData> &param_vector);
    NNTrainingParams getNNTrainingParams(const PartGraph &part_graph);
    std::vector<CompatibilityCondition> getCompatibilityConditions(const PartGraph &part_graph);


    double getEdgeLength(int edgeId);
    int findSharedLine(const std::vector<int>& nodeBoundaryTags, const std::vector<int>& edgeBoundaryTags);
    std::vector<int> findSplineCurves(const std::vector<int>& edgeBoundaryTags);
    int determineSplineOrientation(int splineTag, int sharedLineTag);

    // Print methods for easy testing
    void printMeshGeometry();
    void buildSplitAndPrintMesh(const std::string& filename, double targetPartSize, double nodeEdgePortion);
    void printGraphState();
    void printPartGraphState(const PartGraph& partGraph);
    void printCompatibilityConditions(const std::vector<CompatibilityCondition> &conditions);
    void printTrainingParams (const NNTrainingParams &params);
    void printMeshData(const std::vector<Eigen::MatrixXd> &geom_poly_points,
        const std::vector<Eigen::MatrixXd> &mesh_poly_points,
        const std::vector<Eigen::MatrixXi> &mesh_node_indices);
    void compareMeshParametrizationData(const std::vector<MeshParametrizationData> &data1,
                                        const std::vector<MeshParametrizationData> &data2);
    void printMatrixComparison(const Eigen::MatrixXd &m1, const Eigen::MatrixXd &m2);

};

#endif //METALFOAMS_GRAPH_MESH_H

//
// Created by Liam Curtis on 2024-08-30.
//

#ifndef METALFOAMS_GRAPH_MESH_H
#define METALFOAMS_GRAPH_MESH_H

#include <lf/uscalfe/uscalfe.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/io/io.h>

// The purpose of the following data structure is to store the planned mesh, which is a 2D graph-like mesh, where
// both the edges and the nodes are 2D
// TODO: Describe the naming system required for this structure to work

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
    enum Type { NODE, EDGE } type;
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
    std::vector<std::vector<size_t>> adjacencyList;  //
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
    void buildGraphFromMesh();

    // getter methods for easy retrieval if needed
    std::set<int> getConnectedEdges(int nodeId);
    std::vector<int> getNodesOfEdge(int edgeId);
    std::set<int> getConnectedNodes(int nodeId);
    std::vector<int> getNodeSurfaces(int nodeId);
    std::vector<int> getNodeBoundary(int nodeId);
    std::vector<int> getEdgeSurfaces(int edgeId);
    std::vector<int> getEdgeBoundary(int edgeId);

    // Print methods for easy testing
    void printMeshGeometry();
    void buildSplitAndPrintMesh(const std::string& filename, double targetPartSize, double nodeEdgePortion);
    void printGraphState();
    void printPartGraphState(const PartGraph& partGraph);

    // The functions below help with splitting up the mesh and obtaining the required parametrizations
    PartGraph splitMesh(double targetPartSize, double nodeEdgePortion);
    std::vector<std::vector<double>> getPartGeometry(const MeshPart& part);
    double getEdgeLength(int edgeId);
    int findSharedLine(const std::vector<int>& nodeBoundaryTags, const std::vector<int>& edgeBoundaryTags);
    std::vector<int> findSplineCurves(const std::vector<int>& edgeBoundaryTags);
    int determineSplineOrientation(int splineTag, int sharedLineTag);

};

#endif //METALFOAMS_GRAPH_MESH_H

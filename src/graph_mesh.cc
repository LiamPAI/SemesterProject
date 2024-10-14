//
// Created by Liam Curtis on 2024-08-30.
//

#include "../include/graph_mesh.h"

#include <gmsh.h>
#include <iostream>
#include <filesystem>
#include <unordered_set>

// The following are helpers function that aid in the building of the "graph" for the mesh the NN is attempting to
// find the minimization energy for. The following spits the mesh into 2D edges and nodes through a labelling system
// All "nodes" are labeled as N1, N2, ...
// The node boundary (the curves surrounding the node) for each node is NB1, NB2, ... note that for N1 the node boundary must be NB1
// The same system is used for edges, i.e. E1, E2, ... and EB1, EB2, ...

// Some rules so that these functions work correctly:
//      - The file must be a .geo file, this allows for the methods to query points along each curve
//      - The naming convention for nodes and edges must be followed (e.g. the surface must be named either N# or E#,
//      and the boundary would be NB# and EB#, respectively)
//      - The curves making up a node must be lines, the curves making up the ends of an edge must be lines, and the
//      curves making up the middle portion of an edge must be a spline

// TODO: Delete all print statements once done testing
// TODO: make node portion length optional when building a graph

// Constructor and Destructor
GraphMesh::GraphMesh() {
    gmsh::initialize();
}

GraphMesh::~GraphMesh() {
    if (gmsh::isInitialized()) {
        gmsh::finalize();
    }
}

// Checking methods depending on N, NB, E, or EB labels
bool GraphMesh::isNode(const std::string& name) {
    return name[0] == 'N' && name[1] != 'B';
}

bool GraphMesh::isNodeBoundary(const std::string& name) {
    return name.substr(0, 2) == "NB";
}

bool GraphMesh::isEdge(const std::string& name) {
    return name[0] == 'E' && name[1] != 'B';
}

bool GraphMesh::isEdgeBoundary(const std::string& name) {
    return name.substr(0, 2) == "EB";
}

// Extracts the Id (e.g. for N4 the id is 4) from the physical name of the entity
int GraphMesh::extractId(const std::string& name) {
    return std::stoi(name.substr(isNodeBoundary(name) || isEdgeBoundary(name) ? 2 : 1));
}

// Opens the file with the mesh we are interested in, note that this file will have a *.msh extension
// TODO: Remove print statements once I know this works properly
void GraphMesh::loadMeshFromFile(const std::string& filename) {
    try {
        const std::filesystem::path here = __FILE__;
        auto working_dir = here.parent_path().parent_path();
        gmsh::open(working_dir / "meshes" / filename);
        //gmsh::model::setCurrent("testNE2");  // Use the name without .msh extension

        // Get all surfaces in the model
        std::vector<std::pair<int, int>> surfaces;
        gmsh::model::getEntities(surfaces, 2);  // 2 is for surfaces

        std::cout << "Number of surfaces found: " << surfaces.size() << std::endl;

        for (const auto& surface : surfaces) {
            int surfaceTag = surface.second;
            std::vector<std::pair<int, int>> bounds;
            gmsh::model::getBoundary({{2, surfaceTag}}, bounds, false, false, false);

            std::cout << "Surface " << surfaceTag << " has " << bounds.size() << " boundary entities:" << std::endl;
            for (const auto& bound : bounds) {
                std::cout << "  Dimension: " << bound.first << ", Tag: " << bound.second << std::endl;
            }
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Error loading mesh file: " << e.what() << std::endl;
        throw;
    }
}

void GraphMesh::closeMesh() {
    try {
        gmsh::clear();
        gmsh::finalize();
    }
    catch (const std::runtime_error &e) {
        std::cerr << "Gmsh error: " << e.what() << std::endl;
    }
}

// This builds the graph we will use when setting up mini-meshes for the NN to train on, sets up
// TODO: remove print statement once I know this is working properly
// TODO: Determine if the naming of nodes and edges is index dependent (e.g. if I'm starting from N0 vs N7)
void GraphMesh::buildGraphFromMesh() {
    // Physical groups will be e.g. N2, EB2, note there should be (N + E) * 2 total groups, as each group will have a
    // label for the 2D surface and the boundary surrounding it
    std::vector<std::pair<int, int>> groups;
    gmsh::model::getPhysicalGroups(groups);

    std::cout << "Found " << groups.size() << " physical groups.\n";

    // Iterate through each group using its dimension and corresponding tag
    for (const auto& [dim, tag] : groups) {

        // Extract the name and id from this group
        std::string name;
        gmsh::model::getPhysicalName(dim, tag, name);
        int id = extractId(name);

        // Obtain entities for this group for a node this will be the physical surface, and for a boundary, this will include
        // the multiple curves that make up the boundary
        std::vector<int> entities;
        gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entities);

        std::cout << "Processing group: " << name << " (dim: " << dim << ", tag: " << tag << ", id: " << id << ")\n";
        std::cout << "  Entities: ";
        for (int entity : entities) std::cout << entity << " ";
        std::cout << "\n";

        // Check which type of physical group this is and add the corresponding info to the correct struct
        if (isNode(name)) {
            std::cout << "  Identified as Node\n";
            nodes[id].id = id;
            nodes[id].surfaceTags = entities;
        } else if (isNodeBoundary(name)) {
            std::cout << "  Identified as Node Boundary\n";
            nodes[id].boundaryTags = entities;
        } else if (isEdge(name)) {
            std::cout << "  Identified as Edge\n";
            edges[id].id = id;
            edges[id].surfaceTags = entities;
        } else if (isEdgeBoundary(name)) {
            std::cout << "  Identified as Edge Boundary\n";
            edges[id].boundaryTags = entities;
        } else {
            std::cout << "  Warning: Unrecognized group name format\n";
        }

    }

    // Iterate through edges to establish connections between nodes and edges
    std::cout << "Finding connections...\n";
    for (auto& [edgeId, edge] : edges) {
        std::cout << "Processing Edge " << edgeId << ":\n";
        std::set<int> connectedNodes;

        // Iterate through the boundaries of the edge to see if any are associated to a node (at least 1 should be)
        for (int boundaryTag : edge.boundaryTags) {
            std::cout << "  Checking boundary tag: " << boundaryTag << "\n";
            // Obtain the tags of curves on this particular boundary to see if they are part of other entities
            std::vector<int> curveGroups;
            gmsh::model::getPhysicalGroupsForEntity(1, boundaryTag, curveGroups);
            std::cout << "    Associated node groups: ";
            for (int curveGroup : curveGroups) std::cout << curveGroup << " ";
            std::cout << "\n";
            // Iterate through the curves associated to this boundaryTag
            for (int curveGroup : curveGroups) {
                std::string entityName;
                // Obtain the name of the group with this tag
                gmsh::model::getPhysicalName(1, curveGroup, entityName);
//                std::cout << "    Connected to Node with Name: " << nodeName << "\n";
                // If the name indcates a node through our naming system, we update connectedNodes and connectedEdges
                if (isNodeBoundary(entityName)) {
                    int nodeId = extractId(entityName);
                    std::cout << "    Connected to Node " << nodeId << "\n";
                    connectedNodes.insert(nodeId);
                    nodes[nodeId].connectedEdges.insert(edgeId);
                }
            }
        }

        // Add our connectedNodes to our edge, this should always be of size either 1 or 2
        edge.connectedNodes.assign(connectedNodes.begin(), connectedNodes.end());
        std::cout << "  Edge " << edgeId << " is connected to nodes: ";
        for (int nodeId : edge.connectedNodes) std::cout << nodeId << " ";
        std::cout << "\n";
    }
}

// The purpose of this method is to use a graphMesh and split the mesh into mini portions made up of node parts and edge parts
// Node parts consist of the "node" of the overall mesh and portions of all the edges connected to it
// Edge parts consist of portions of the edge, parametrized by a percentage of the overall branch
// Using these individual parts, we can then obtain the geometries of each part to set up parametrizations
// TODO: Remove print statements once I know this is working properly
PartGraph GraphMesh::splitMesh(double targetPartSize, double nodeEdgePortion = 0.1) {
    std::cout << "Starting splitMesh function with targetPartSize: " << targetPartSize << " and nodeEdgePortion: " << nodeEdgePortion << std::endl;

    // partGraph will contain all the meshParts and their respective connectivity
    // partIndices is a mapping (Edge v Node type, ID, subID) -> index in the partGraph for easy access
    // throughout this function
    PartGraph partGraph;
    std::map<std::tuple<MeshPart::Type, int, int>, size_t> partIndices;

    // TODO: Maybe make sure targetPartSize is somehow less than the overall edge length

    // Iterate through all nodes to create their respective parts
    std::cout << "Processing nodes..." << std::endl;
    for (const auto& [nodeId, node] : nodes) {
        // Initialize what we can of our meshPart, connectedEdges, connectedEdgeTags, and curveTags remain
        std::cout << "  Processing node " << nodeId << std::endl;
        MeshPart part;
        part.type = MeshPart::NODE;
        part.id = nodeId;
        part.subId = 0; // subId is only relevant for edge parts
        part.surfaceTags = node.surfaceTags;
        part.boundaryTags = node.boundaryTags;

        // Store the tags of connected edges to update the connectedEdgeTags since this is a node part,
        // also store the part or portion of each edge belonging to this node part
        std::vector<int> connectedEdgeTags;
        for (int edgeId : node.connectedEdges) {
            std::cout << "    Processing connected edge " << edgeId << std::endl;

            // We determine what portion of the connected edge we want depending on if it is the first node,
            // making sure, for later connection reasons, that we go from the node outwards
            const auto& edge = edges[edgeId];
            bool isFirstNode = (edge.connectedNodes[0] == nodeId);
            double startPortion = isFirstNode ? 0.0 : 1.0;
            double endPortion = isFirstNode ? nodeEdgePortion : (1.0 - nodeEdgePortion);
            part.connectedEdges.emplace_back(edgeId,std::make_pair(startPortion, endPortion));
            connectedEdgeTags.push_back(edgeId);

            // Find the spline curves for this connected edge
            std::vector<int> splineTags = findSplineCurves(edge.boundaryTags);
            std::cout << "    Found " << splineTags.size() << " spline tags for edge " << edgeId << std::endl;
            LF_ASSERT_MSG(splineTags.size() == 2, "The spline tags returned for edge with id " << edgeId <<
                " returned " << splineTags.size() << " splines, when it should return 2");

            // For the splines we want to interpolate from, we add the tag, their orientation, and portion to curveTags
            if (splineTags.size() == 2) {
                int sharedLineTag = findSharedLine(node.boundaryTags, edge.boundaryTags);
                std::cout << "    Shared line tag: " << sharedLineTag << std::endl;

                // Once we have determined the shared line, we can obtain the orientation of both splines on this edge
                if (sharedLineTag != -1) {
                    for (int splineTag : splineTags) {
                        int orientation = determineSplineOrientation(splineTag, sharedLineTag);
                        if (isFirstNode) {
                            std::cout << "    Spline tag: " << splineTag << ", Orientation: " << orientation << " , startPortion: " << startPortion << ", endPortion: " << endPortion << std::endl;
                            part.curveTags.emplace_back(splineTag, orientation, std::make_pair(startPortion, endPortion));
                        }
                        else {
                            std::cout << "    Spline tag: " << splineTag << ", Orientation (switched): " << -orientation << " , startPortion: " << startPortion << ", endPortion: " << endPortion << std::endl;
                            part.curveTags.emplace_back(splineTag, -orientation, std::make_pair(startPortion, endPortion));
                        }
                    }
                }
            }
        }
        part.connectedEdgeTags = connectedEdgeTags;

        // Update the partIndices mapping to easily update the adjacency list later
        partIndices[std::make_tuple(MeshPart::NODE, nodeId, 0)] = partGraph.parts.size();
        partGraph.parts.push_back(std::move(part));
        std::cout << "  Finished processing node " << nodeId << std::endl;
    }

    // Iterate through all edges, where the plan is to divide up each portion that is not part of a node
    std::cout << "Processing edges..." << std::endl;
    for (const auto& [edgeId, edge] : edges) {
        std::cout << "  Processing edge " << edgeId << std::endl;
        double edgeLength = getEdgeLength(edgeId);
        std::cout << "  Edge length: " << edgeLength << std::endl;

        // Adjust the number of divisions based on whether the edge is connected to one or two nodes
        double effectiveLength = edge.connectedNodes.size() == 1 ? edgeLength * (1 - nodeEdgePortion) : edgeLength * (1 - 2 * nodeEdgePortion);
        int edgeDivisions = std::max(1, static_cast<int>(std::round(effectiveLength / targetPartSize)));
        std::cout << "  Effective length: " << effectiveLength << ", Edge divisions: " << edgeDivisions << std::endl;

        // Depending on whether this edge is connected to 1 or 2 nodes we initialize the portion we're splitting
        double startPortion = nodeEdgePortion;
        double endPortion = edge.connectedNodes.size() == 1 ? 1.0 : 1.0 - nodeEdgePortion;
        double step = (endPortion - startPortion) / edgeDivisions;
        std::cout << "  Start portion: " << startPortion << ", End portion: " << endPortion << ", Step: " << step << std::endl;

        // Obtain splines for which we will interpolate from later
        std::vector<int> splineTags = findSplineCurves(edge.boundaryTags);
        std::cout << "  Found " << splineTags.size() << " spline tags for edge " << edgeId << std::endl;

        std::vector<int> splineOrientations(2);
        // We check if it's empty, though it shouldn't be assuming correct naming
        if (!edge.connectedNodes.empty()) {

            // We find the shared line tag in order to determine spline orientations
            int sharedLineTag = findSharedLine(nodes[edge.connectedNodes[0]].boundaryTags, edge.boundaryTags);
            std::cout << "  Shared line tag: " << sharedLineTag << std::endl;

            // sharedLineTag is equal to -1 if a shared line wasn't found in the first place, shouldn't happen
            if (sharedLineTag != -1) {
                splineOrientations[0] = determineSplineOrientation(splineTags[0], sharedLineTag);
                splineOrientations[1] = determineSplineOrientation(splineTags[1], sharedLineTag);
                std::cout << "  Spline orientations: " << splineOrientations[0] << ", " << splineOrientations[1] << std::endl;
            }
        }
        else {
            LF_ASSERT_MSG(false, "Edge does not have a connected node in splitMesh");
        }

        // We now add these edge parts along with their corresponding portions
        for (int i = 0; i < edgeDivisions; ++i) {
            std::cout << "    Creating edge part " << i << " for edge " << edgeId << std::endl;
            MeshPart part;
            part.type = MeshPart::EDGE;
            part.id = edgeId;
            part.subId = i;
            part.surfaceTags = edge.surfaceTags;
            part.boundaryTags = edge.boundaryTags;

            // Add the portion for this edge, that we calculated above
            double partStart = startPortion + i * step;
            double partEnd = partStart + step;
            part.connectedEdges.emplace_back(edgeId, std::make_pair(partStart, partEnd));
            std::cout << "    Part range: " << partStart << " to " << partEnd << std::endl;

            // Add the orientations we just found above for this edge
            part.curveTags.emplace_back(splineTags[0], splineOrientations[0], std::make_pair(partStart, partEnd));
            part.curveTags.emplace_back(splineTags[1], splineOrientations[1], std::make_pair(partStart, partEnd));

            // Add this part to our mapping for easy access later regarding the adjacency list
            partIndices[std::make_tuple(MeshPart::EDGE, edgeId, i)] = partGraph.parts.size();
            partGraph.parts.push_back(std::move(part));
        }
    }

    // Build adjacency list to correct size now that all parts have been initialized
    std::cout << "Building adjacency list..." << std::endl;
    partGraph.adjacencyList.resize(partGraph.parts.size());

    // Now that all edge parts are defined, we see which edge parts are connected to which node
    std::cout << "Connecting nodes to edge parts..." << std::endl;
    for (const auto& [nodeId, node] : nodes) {
        std::cout << "  Processing node " << nodeId << std::endl;

        // Obtain the index for this node part in adjacencyList
        size_t nodePartIndex = partIndices[std::make_tuple(MeshPart::NODE, nodeId, 0)];

        // Iterate through this node's connectedEdges
        for (int edgeId : node.connectedEdges) {
            std::cout << "    Processing connected edge " << edgeId << std::endl;
            const auto& edge = edges[edgeId];

            // Determine which side of the edge to connect to this node
            bool isFirstNode = (edge.connectedNodes[0] == nodeId);
            int sideIndex = isFirstNode ? 0 : 1;

            // Find the first and last edge parts on this side, there should always be at least 2
            size_t firstEdgePartIndex = partIndices[std::make_tuple(MeshPart::EDGE, edgeId, 0)];
            size_t lastEdgePartIndex = partIndices[std::make_tuple(MeshPart::EDGE, edgeId,
                                                                   std::count_if(partGraph.parts.begin(), partGraph.parts.end(),
                                                                                 [edgeId](const MeshPart& part) { return part.type == MeshPart::EDGE && part.id == edgeId; }) - 1)];

            // Connect the node to the appropriate edge parts, and the edge part to the appropriate node
            if (sideIndex == 0) {
                std::cout << "    Connecting node " << nodeId << " to first edge part of edge " << edgeId << std::endl;
                partGraph.adjacencyList[nodePartIndex].push_back(firstEdgePartIndex);
                partGraph.adjacencyList[firstEdgePartIndex].push_back(nodePartIndex);
            } else {
                std::cout << "    Connecting node " << nodeId << " to last edge part of edge " << edgeId << std::endl;
                partGraph.adjacencyList[nodePartIndex].push_back(lastEdgePartIndex);
                partGraph.adjacencyList[lastEdgePartIndex].push_back(nodePartIndex);
            }
        }
    }

    // Iterate through all Edges to connect adjacent edgeParts
    std::cout << "Connecting adjacent edge parts..." << std::endl;
    for (const auto& pair : edges) {
        int edgeId = pair.first;
        Edge2D edge = pair.second;

        std::cout << "  Processing edge " << edgeId << std::endl;
        // This returns the number of parts making up the sub-divisions of the edge
        int edgeDivisions = std::count_if(partGraph.parts.begin(), partGraph.parts.end(),
                                          [edgeId](const MeshPart& part) { return part.type == MeshPart::EDGE && part.id == edgeId; });
        std::cout << "  Edge divisions: " << edgeDivisions << std::endl;

        // Loop through almost all edge Divisions
        for (int i = 0; i < edgeDivisions - 1; ++i) {

            // Correctly update the adjacency list of these divisions
            size_t currentPartIndex = partIndices[std::make_tuple(MeshPart::EDGE, edgeId, i)];
            size_t nextPartIndex = partIndices[std::make_tuple(MeshPart::EDGE, edgeId, i + 1)];
            std::cout << "    Connecting edge part " << i << " to edge part " << (i+1) << std::endl;
            partGraph.adjacencyList[currentPartIndex].push_back(nextPartIndex);
            partGraph.adjacencyList[nextPartIndex].push_back(currentPartIndex);
        }
    }

    std::cout << "splitMesh function completed. Total parts created: " << partGraph.parts.size() << std::endl;
    return partGraph;
}

// This function helps to find the shared line between a node and an edge
int GraphMesh::findSharedLine(const std::vector<int>& nodeBoundaryTags, const std::vector<int>& edgeBoundaryTags) {
    for (int nodeTag : nodeBoundaryTags) {
        // Check if this node tag is shared with the edge boundary tags
        if (std::find(edgeBoundaryTags.begin(), edgeBoundaryTags.end(), nodeTag) != edgeBoundaryTags.end()) {
            // Verify that this is indeed a line curve
            return nodeTag;
        }
    }
    return -1; // No shared line found, should never be reached ideally, unless a naming error has occurred with the original mesh
}

// This function helps to find the two spline curves of an edge, to do so, I assume that since the ends of an edge are
// lines, the 2 middle curves are made up of splines, or "Nurb" within gmsh
std::vector<int> GraphMesh::findSplineCurves(const std::vector<int>& edgeBoundaryTags) {

    std::vector<int> splineTags;
    std::string curveType;

    // Print contents of the two vectors for later testing
    std::cout << "edgeBoundaryTags contents:" << std::endl;
    for (const auto& pair : edgeBoundaryTags) {
        std::cout << pair << std::endl;
    }

    // iterate through edgeBoundary tags to find which ones are the splines
    for (int edgeTag : edgeBoundaryTags) {
        gmsh::model::getType(1, edgeTag, curveType);
        if (curveType == "Nurb") {
            splineTags.emplace_back(edgeTag);
        }
    }

    return splineTags;
}

// This function helps to determine the orientation of a spline relative to a shared line between a node and an edge
// TODO: double-check this with slightly different geometries and splines in different orientations
int GraphMesh::determineSplineOrientation(int splineTag, int sharedLineTag) {
    std::vector<std::pair<int, int>> splinePoints, linePoints;
    gmsh::model::getBoundary({{1, splineTag}}, splinePoints, false);
    gmsh::model::getBoundary({{1, sharedLineTag}}, linePoints, false);

    std::cout << "The splineTag is " << splineTag << " and the line tag is " << sharedLineTag << std::endl;

    std::cout << "splinePoints contents:" << std::endl;
    for (const auto& pair : splinePoints) {
        std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
    }
    std::cout << "linePoints contents:" << std::endl;
    for (const auto& pair : linePoints) {
        std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
    }

    // Check if the start point of the spline is on the shared line, then it is oriented in the way we want
    bool splineStartOnLine = (splinePoints[0].second == linePoints[0].second) ||
            (splinePoints[0].second == linePoints[1].second);
    return splineStartOnLine ? 1 : -1;
}

// This function takes in a graph of parts and returns the corresponding vector of polynomial points in the same order
// as defined in the parts vector of PartGraph, note that since this comes from the part_graph, and call
// getPartGeometry, that these are the exact points of the geometry file, which may not be nodes of the actual mesh
std::vector<Eigen::MatrixXd> GraphMesh::getGeometryPolynomialPoints(const PartGraph &part_graph) {
    std::vector<Eigen::MatrixXd> points_vector;
    for (const auto& part : part_graph.parts) {
        auto geometry = getPartGeometry(part);
        points_vector.push_back(geometry);
    }
    return points_vector;
}

// This function takes in a part_graph and the file name to the corresponding mesh that generated our partGraph,
// and returns the actual polynomial points of the mesh, which will correspond to the closest nodes on the mesh,
// and we can therefore easily compare displacement vectors, it also returns the indices of the nodes we used
std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXi>> GraphMesh::getMeshPolynomialPoints(
    const PartGraph &part_graph, const std::string &mesh_file_name, int order) {

    // We next read the mesh file name into LehrFEM to easily retrieve the nodes and their coordinates
    auto factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    std::filesystem::path project_root = std::filesystem::current_path().parent_path();
    std::filesystem::path mesh_dir = project_root / "meshes";
    std::filesystem::path full_path = mesh_dir / (mesh_file_name + ".msh");

    lf::io::GmshReader reader(std::move(factory), full_path);
    const std::shared_ptr<lf::mesh::Mesh> mesh_ptr = reader.mesh();
    const lf::mesh::Mesh &mesh {*mesh_ptr};

    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space;
    if (order == 1) {
        fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
    }
    else if (order == 2) {
        fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);
    }
    else {
        LF_ASSERT_MSG(false, "Incorrect order sent to getMeshPolynomialPoints: " << order);
    }
    const lf::assemble::DofHandler &dofh {fe_space->LocGlobMap()};

    std::vector<Eigen::MatrixXd> geom_poly_points = getGeometryPolynomialPoints(part_graph);
    std::vector<Eigen::MatrixXd> mesh_poly_points;
    std::vector<Eigen::MatrixXi> mesh_node_indices;

    // Iterate through all point matrices
    for (const auto &points : geom_poly_points) {
        int num_points = points.rows() / 2 * 3;
        Eigen::MatrixXd mesh_points(points.rows(), points.cols());
        Eigen::MatrixXi mesh_indices(points.rows() / 2, points.cols());

        // Iterate through all points of this particular matrix
        for (int i = 0; i < num_points; ++i) {
            double closest_distance = std::numeric_limits<double>::max();
            Eigen::Vector2d closest_point = Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
            auto current_point = points.block<2, 1>(i / 3 * 2, i % 3);
            int current_index = -1;

            // Iterate through all nodes of the mesh to find the closest one
            for (const lf::mesh::Entity *node : mesh.Entities(2)) {
                auto possible_point = node->Geometry()->Global(node->RefEl().NodeCoords());
                double new_distance = (possible_point - current_point).norm();
                if (new_distance < closest_distance) {
                    closest_distance = new_distance;
                    closest_point = possible_point;
                    current_index = dofh.GlobalDofIndices(*node)[0];
                }
            }
            mesh_points.block<2, 1>(i / 3 * 2, i % 3) = closest_point;
            mesh_indices(i / 3, i % 3) = current_index;
        }
        mesh_poly_points.push_back(mesh_points);
        mesh_node_indices.push_back(mesh_indices);
    }
    return {mesh_poly_points, mesh_node_indices};
}

// This function takes in a graph of parts and returns the corresponding vector of parametrizations in the same order
// as defined in the parts vector of PartGraph, note these are the true parametrizations, as they were obtained
// from the .geo file
std::vector<MeshParametrizationData> GraphMesh::getGeometryParametrizations(const PartGraph &part_graph) {
    std::vector<MeshParametrizationData> params_vector;

    for (const auto& part : part_graph.parts) {
        auto geometry = getPartGeometry(part);
        auto parametrization = MeshParametrization::pointToParametrization(geometry);
        params_vector.push_back(parametrization);
    }
    return params_vector;
}

// This function takes in a graph of parts, a file name, and an order and outputs the corresponding vector of
// parametrizations on the actual mesh, where the points used to determine these parametrizations are the actual
// nodes on the mesh
std::vector<MeshParametrizationData> GraphMesh::getMeshParametrizations(const PartGraph &part_graph,
    const std::string &mesh_file_name, int order) {

    std::vector<MeshParametrizationData> params_vector;
    auto [mesh_point_vector, mesh_node_vector] = getMeshPolynomialPoints(part_graph, mesh_file_name, order);

    for (const auto& points : mesh_point_vector) {
        auto parametrization = MeshParametrization::pointToParametrization(points);
        params_vector.push_back(parametrization);
    }
    return params_vector;
}

// This function takes in a vector of parametrizations represented by their points, and returns the same vector of parametrizations,
// but with each parametrization centered around the origin so that they are normalized for the NN
std::vector<Eigen::MatrixXd> GraphMesh::centerPointParametrizations(const std::vector<Eigen::MatrixXd> &point_vector) {
    std::vector<Eigen::MatrixXd> centered_point_vector;

    for (const auto &points : point_vector) {
        Eigen::Vector2d center = DataOperations::findCenter(MeshParametrization::pointToParametrization(points));
        Eigen::MatrixXd centered_points (points);
        for (int i = 0; i < points.rows() / 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                centered_points.block<2, 1>(2 * i, j) =
                    centered_points.block<2, 1>(2 * i, j) - center;
            }
        }
        centered_point_vector.push_back(centered_points);
    }
    return centered_point_vector;
}

// This function takes in a vector of MeshParametrizationData and returns the same vector but with each parametrization
// centered around the origin, so that they are "normalized" for the NN
std::vector<MeshParametrizationData> GraphMesh::centerMeshParametrizations(
    const std::vector<MeshParametrizationData> &param_vector) {
    std::vector<MeshParametrizationData> centered_param_vector;
    for (const auto &param : param_vector) {
        Eigen::Vector2d center = DataOperations::findCenter(param);
        MeshParametrizationData centered_param (param);
        for (int i = 0; i < param.numBranches; ++i) {
            for (int j = 0; j < 3; ++j) {
                centered_param.terminals.block<2, 1>(2 * i, j) =
                    centered_param.terminals.block<2, 1>(2 * i, j) - center;
            }
        }
        centered_param_vector.push_back(centered_param);
    }
    return centered_param_vector;
}

// This function takes in a graph of our parts and returns training parameters for the neural network, such as the
// range to train for widths, lengths, angles, and so on
NNTrainingParams GraphMesh::getNNTrainingParams(const PartGraph &part_graph) {
    auto params_vector = getGeometryParametrizations(part_graph);
    double min_length = std::numeric_limits<double>::max();
    double max_length = 0;
    double min_width = std::numeric_limits<double>::max();
    double max_width = 0;
    double min_width_diff = std::numeric_limits<double>::max();
    double max_width_diff = 0;
    double min_angle_diff = std::numeric_limits<double>::max();
    double max_angle_diff = 0;

    for (auto param : params_vector) {
        double approx_length;
        Eigen::Vector2d zeros = Eigen::Vector2d::Zero();
        if (param.numBranches == 1) {
            approx_length = (param.terminals.block<2 ,1>(0, 1) -
                    param.terminals.block<2 ,1>(0, 0)).norm() +
                    (param.terminals.block<2 ,1>(0, 2) -
                   param.terminals.block<2 ,1>(0, 1)).norm();

            if (approx_length > max_length) max_length = approx_length;
            if (approx_length < min_length) min_length = approx_length;

            min_width_diff = std::min(std::min(std::abs(param.widths(0, 0) / param.widths(0, 1) - 1),
                std::abs(param.widths(0, 1) / param.widths(0, 2) - 1)),
                std::min(std::abs(param.widths(0, 0) / param.widths(0, 2) - 1), min_width_diff));
            max_width_diff = std::max(std::max(std::abs(param.widths(0, 0) / param.widths(0, 1) - 1),
                std::abs(param.widths(0, 1) / param.widths(0, 2) - 1)),
                std::max(std::abs(param.widths(0, 0) / param.widths(0, 2) - 1), max_width_diff));

            LineMapping vectors0_and_1(zeros, param.vectors.col(0), zeros, param.vectors.col(1));
            LineMapping vectors1_and_2(zeros, param.vectors.col(1), zeros, param.vectors.col(2));
            LineMapping vectors0_and_2(zeros, param.vectors.col(0), zeros, param.vectors.col(2));

            min_angle_diff = std::min(std::min(vectors0_and_1.angleBetweenLines(), vectors1_and_2.angleBetweenLines()),
                std::min(vectors0_and_2.angleBetweenLines(), min_angle_diff));
            max_angle_diff = std::max(std::max(vectors0_and_1.angleBetweenLines(), vectors1_and_2.angleBetweenLines()),
                std::max(vectors0_and_2.angleBetweenLines(), max_angle_diff));
        }
        else {
            for (int i = 0; i < param.numBranches; ++i) {
                approx_length = (param.terminals.block<2 ,1>(2 * i, 1) -
                    param.terminals.block<2 ,1>(2 * i, 0)).norm() +
                    (param.terminals.block<2 ,1>(2 * i, 2) -
                    param.terminals.block<2 ,1>(2 * i, 1)).norm();

                if (approx_length > max_length) max_length = approx_length;
                if (approx_length < min_length) min_length = approx_length;

                min_width_diff = std::min(std::min(std::abs(param.widths(i, 0) / param.widths(i, 1) - 1),
                std::abs(param.widths(i, 1) / param.widths(i, 2) - 1)),
                std::min(std::abs(param.widths(i, 0) / param.widths(i, 2) - 1), min_width_diff));
                max_width_diff = std::max(std::max(std::abs(param.widths(i, 0) / param.widths(i, 1) - 1),
                    std::abs(param.widths(i, 1) / param.widths(i, 2) - 1)),
                    std::max(std::abs(param.widths(i, 0) / param.widths(i, 2) - 1), max_width_diff));

                LineMapping vectors0_and_1(zeros, param.vectors.block<2, 1>(2 * i, 0),
                    zeros, param.vectors.block<2, 1>(2 * i, 1));
                LineMapping vectors1_and_2(zeros, param.vectors.block<2, 1>(2 * i, 1),
                    zeros, param.vectors.block<2, 1>(2 * i, 2));
                LineMapping vectors0_and_2(zeros, param.vectors.block<2, 1>(2 * i, 0),
                    zeros, param.vectors.block<2, 1>(2 * i, 2));

                min_angle_diff = std::min(std::min(vectors0_and_1.angleBetweenLines(), vectors1_and_2.angleBetweenLines()),
                    std::min(vectors0_and_2.angleBetweenLines(), min_angle_diff));
                max_angle_diff = std::max(std::max(vectors0_and_1.angleBetweenLines(), vectors1_and_2.angleBetweenLines()),
                    std::max(vectors0_and_2.angleBetweenLines(), max_angle_diff));
            }
        }

        double min_width_param = param.widths.minCoeff();
        double max_width_param = param.widths.maxCoeff();

        if (min_width_param < min_width) min_width = min_width_param;
        if (max_width_param > max_width) max_width = max_width_param;
    }
    return {{min_length, max_length}, {min_width, max_width},
        {min_width_diff, max_width_diff}, {min_angle_diff, max_angle_diff}};
}

// This function takes in a PartGraph and returns a vector of all the compatibility conditions in the PartGraph
std::vector<CompatibilityCondition> GraphMesh::getCompatibilityConditions(const PartGraph &part_graph) {
    std::vector<CompatibilityCondition> compatibility_conditions;
    std::unordered_set<std::pair<int, int>, IntPairHash, IntPairEqual> processed_conditions;

    // Lambda expression to check if we've already added this condition or not previously
    auto process_condition = [&processed_conditions](int a, int b) {
        std::pair<int, int> current_condition = {std::min(a, b), std::max(a, b)};
        if (processed_conditions.find(current_condition) == processed_conditions.end()) {
            processed_conditions.insert(current_condition);
            return true;
        }
        return false;
    };

    for (int i = 0; i < part_graph.parts.size(); i++) {
        MeshPart part = part_graph.parts[i];

        if (part.type == MeshPart::NODE) {
            // Need to iterate through all connected parts for this node
            for (int j = 0; j < part_graph.adjacencyList[i].size(); j++) {

                // If we have not processed this condition yet, we do it now
                if (process_condition(i, part_graph.adjacencyList[i][j])) {

                    if (part_graph.parts[part_graph.adjacencyList[i][j]].type == MeshPart::NODE) {

                        // Find the position of our node i within the adjacent node's adjacency list
                        auto it =
                            std::find(part_graph.adjacencyList[part_graph.adjacencyList[i][j]].begin(),
                            part_graph.adjacencyList[part_graph.adjacencyList[i][j]].end(), i);

                        LF_ASSERT_MSG(it != part_graph.adjacencyList[part_graph.adjacencyList[i][j]].end(),
                            "Two nodes are connected in the part graph, but their adjacency lists are incorrect");
                        int index = std::distance(part_graph.adjacencyList[part_graph.adjacencyList[i][j]].begin(), it);

                        std::pair<int, int> indices {i, part_graph.adjacencyList[i][j]};
                        std::pair<int, int> first {j, 2};
                        std::pair<int, int> second {index, 2};
                        compatibility_conditions.emplace_back(indices, first, second);
                    }
                    else {
                        // Since this adjacent part is an edge, we must check if the node is first or second in the
                        // corresponding graph to see which side the equivalent points are on
                        const auto& edge = edges[part_graph.parts[part_graph.adjacencyList[i][j]].id];
                        bool is_first_node = (edge.connectedNodes[0] == part_graph.parts[i].id);

                        // Initialize the pairs that will make up this compatibility condition
                        std::pair<int, int> indices {i, part_graph.adjacencyList[i][j]};
                        std::pair<int, int> first {j, 2};
                        std::pair<int, int> second {0, is_first_node ? 0 : 2};
                        compatibility_conditions.emplace_back(indices, first, second);
                    }
                }
            }
        }

        else {
            // Need to iterate through all connected parts for this node
            for (int j = 0; j < part_graph.adjacencyList[i].size(); j++) {
                // If we have not processed this condition yet, we do it now
                if (process_condition(i, part_graph.adjacencyList[i][j])) {

                    if (part_graph.parts[part_graph.adjacencyList[i][j]].type == MeshPart::NODE) {
                        auto it =
                            std::find(part_graph.adjacencyList[part_graph.adjacencyList[i][j]].begin(),
                            part_graph.adjacencyList[part_graph.adjacencyList[i][j]].end(), i);
                        LF_ASSERT_MSG(it != part_graph.adjacencyList[part_graph.adjacencyList[i][j]].end(),
                        "An edge is connected to a node in the part graph, but their adjacency lists are incorrect");

                        int index = std::distance(part_graph.adjacencyList[part_graph.adjacencyList[i][j]].begin(), it);
                        const auto& edge = edges[part_graph.parts[i].id];
                        bool is_first_node = (edge.connectedNodes[0] == part_graph.parts[i].id);

                        std::pair<int, int> indices {i, part_graph.adjacencyList[i][j]};
                        std::pair<int, int> first {0, is_first_node ? 0 : 2};
                        std::pair<int, int> second {index, 2};
                        compatibility_conditions.emplace_back(indices, first, second);
                    }
                    else {
                        bool is_first_part = part_graph.parts[i].subId <
                            part_graph.parts[part_graph.adjacencyList[i][j]].subId;

                        std::pair<int, int> indices {i, part_graph.adjacencyList[i][j]};
                        std::pair<int, int> first {0, is_first_part ? 2 : 0};
                        std::pair<int, int> second {0, is_first_part ? 0 : 2};
                        compatibility_conditions.emplace_back(indices, first, second);
                    }
                }
            }
        }
    }
    return compatibility_conditions;
}

// TODO: Test this guy one more time with a very complicated geometry mesh where there are multiple edges
//  connected to 2 nodes
// This function takes in a meshPart and returns a matrix containing the polynomial points of the corresponding
// parametrization
Eigen::MatrixXd GraphMesh::getPartGeometry(const MeshPart& part) {
    std::vector<std::vector<double>> geometry;

    std::cout << "Debug: Starting getPartGeometry for part type: " << (part.type == MeshPart::NODE ? "NODE" : "EDGE") << std::endl;

    // iterate through the curveTags of this part in order to acquire points along the curve
    for (const auto& [curveTag, orientation, portions] : part.curveTags) {
        std::cout << "Debug: Processing curveTag: " << curveTag << " with orientation: " << orientation
        << " and portions: [" << portions.first << ", " << portions.second << "]" << std::endl;

        std::vector<double> tMin, tMax;
        gmsh::model::getParametrizationBounds(1, curveTag, tMin, tMax);
        std::cout << "Debug: Parametrization bounds: tMin = " << tMin[0] << ", tMax = " << tMax[0] << std::endl;

        double startPortion = portions.first;
        double endPortion = portions.second;

        // Adjust tMin and tMax based on the portion and orientation
        double tStart, tEnd;
        if (orientation == 1) {
            tStart = tMin[0] + startPortion * (tMax[0] - tMin[0]);
            tEnd = tMin[0] + endPortion * (tMax[0] - tMin[0]);
        } else {
            tStart = tMax[0] - startPortion * (tMax[0] - tMin[0]);
            tEnd =  tMax[0] - endPortion * (tMax[0] - tMin[0]);
        }
        std::cout << "Debug: Adjusted t values: tStart = " << tStart << ", tEnd = " << tEnd << std::endl;

        // Get start point
        std::vector<double> startPoint;
        gmsh::model::getValue(1, curveTag, {tStart}, startPoint);
        geometry.push_back(startPoint);
        std::cout << "Debug: Start point: (" << startPoint[0] << ", " << startPoint[1] << ", " << startPoint[2] << ")" << std::endl;

        // Get middle point
        std::vector<double> midPoint;
        gmsh::model::getValue(1, curveTag, {(tStart + tEnd) / 2}, midPoint);
        geometry.push_back(midPoint);
        std::cout << "Debug: Middle point: (" << midPoint[0] << ", " << midPoint[1] << ", " << midPoint[2] << ")" << std::endl;

        // Get end point
        std::vector<double> endPoint;
        gmsh::model::getValue(1, curveTag, {tEnd}, endPoint);
        geometry.push_back(endPoint);
        std::cout << "Debug: End point: (" << endPoint[0] << ", " << endPoint[1] << ", " << endPoint[2] << ")" << std::endl;
    }

    Eigen::MatrixXd poly_points(geometry.size() / 3 * 2, 3);

    for (int i = 0; i < geometry.size() / 3; ++i) {
        poly_points(2 * i, 0) = geometry[3 * i][0];
        poly_points(2 * i + 1, 0) = geometry[3 * i][1];
        poly_points(2 * i, 1) = geometry[3 * i + 1][0];
        poly_points(2 * i + 1, 1) = geometry[3 * i + 1][1];
        poly_points(2 * i, 2) = geometry[3 * i + 2][0];
        poly_points(2 * i + 1, 2) = geometry[3 * i + 2][1];
    }
    std::cout << "Debug: Finished getPartGeometry. Returned " << geometry.size() << " points." << std::endl;

    return poly_points;
}

double GraphMesh::getEdgeLength(int edgeId) {

    // Find the spline curves of this edge
    const auto& edge = edges.at(edgeId);
    std::vector<int> splineTags = findSplineCurves(edge.boundaryTags);

    double totalLength = 0.0;
    // Iterate through both splines to find the length of each one
    for (int splineTag : splineTags) {
        std::vector<double> tMin, tMax;
        gmsh::model::getParametrizationBounds(1, splineTag, tMin, tMax);

        std::cout << "For splineTag " << splineTag << " the min params are " << tMin[0] << std::endl;
        std::cout << "For splineTag " << splineTag << " the max params are " << tMax[0] << std::endl;

        const int numSamples = 10; // Adjust this for better approximation
        double splineLength = 0.0;
        std::vector<double> prevPoint;

        // We loop through the spline and add up lengths given by the points
        for (int i = 0; i <= numSamples; ++i) {
            double t = tMin[0] + (tMax[0] - tMin[0]) * i / numSamples;
            std::vector<double> point;
            gmsh::model::getValue(1, splineTag, {t}, point);


            std::cout << "Value of t is " << t << std::endl;
            std::cout << "Value of point is " << point[0] << ", " << point[1] << ", " << point[2] << std::endl;

            if (i > 0) {
                double segmentLength = std::sqrt(
                        std::pow(point[0] - prevPoint[0], 2) +
                        std::pow(point[1] - prevPoint[1], 2) +
                        std::pow(point[2] - prevPoint[2], 2)
                );
                splineLength += segmentLength;
                std::cout << "Segment length added on this iteration is " << segmentLength << " with total length being " << splineLength << std::endl;
            }

            prevPoint = point;
        }

        totalLength += splineLength;
    }

    // We want the average of the two splines approximately
    return totalLength / splineTags.size();
}


// Various get methods that are self-explanatory
std::set<int> GraphMesh::getConnectedEdges(int nodeId) {
    return nodes[nodeId].connectedEdges;
}

std::vector<int> GraphMesh::getNodesOfEdge(int edgeId) {
    return edges[edgeId].connectedNodes;
}

// Get neighbouring nodes given a node id
std::set<int> GraphMesh::getConnectedNodes(int nodeId) {
    std::set<int> connectedNodes;
    for (int edgeId : nodes[nodeId].connectedEdges) {
        const auto& edgeNodes = edges[edgeId].connectedNodes;
        connectedNodes.insert(edgeNodes.begin(), edgeNodes.end());
    }
    connectedNodes.erase(nodeId);
    return connectedNodes;
}

std::vector<int> GraphMesh::getNodeSurfaces(int nodeId) {
    return nodes[nodeId].surfaceTags;
}

std::vector<int> GraphMesh::getNodeBoundary(int nodeId) {
    return nodes[nodeId].boundaryTags;
}

std::vector<int> GraphMesh::getEdgeSurfaces(int edgeId) {
    return edges[edgeId].surfaceTags;
}

std::vector<int> GraphMesh::getEdgeBoundary(int edgeId) {
    return edges[edgeId].boundaryTags;
}

// Prints info about the mesh geometry for further testing
void GraphMesh::printMeshGeometry() {
    std::cout << std::setfill('=') << std::setw(50) << "\n";
    std::cout << "Mesh Geometry:\n";
    std::cout << std::setfill('=') << std::setw(50) << "\n";

    std::cout << "Nodes:\n";
    std::cout << std::setfill('-') << std::setw(30) << "\n";
    for (const auto& [nodeId, node] : nodes) {
        std::cout << "Node " << nodeId << ":\n";
        std::cout << "  Surfaces: ";
        for (int tag : node.surfaceTags) std::cout << tag << " ";
        std::cout << "\n  Boundary: ";
        for (int tag : node.boundaryTags) std::cout << tag << " ";
        std::cout << "\n  Connected Edges: ";
        for (int edge : node.connectedEdges) std::cout << edge << " ";
        std::cout << "\n";
        std::cout << std::setfill('-') << std::setw(30) << "\n";
    }

    std::cout << "Edges:\n";
    std::cout << std::setfill('-') << std::setw(30) << "\n";
    for (const auto& [edgeId, edge] : edges) {
        std::cout << "Edge " << edgeId << ":\n";
        std::cout << "  Surfaces: ";
        for (int tag : edge.surfaceTags) std::cout << tag << " ";
        std::cout << "\n  Boundary: ";
        for (int tag : edge.boundaryTags) std::cout << tag << " ";
        std::cout << "\n  Connected Nodes: ";
        for (int node : edge.connectedNodes) std::cout << node << " ";
        std::cout << "\n";
        std::cout << std::setfill('-') << std::setw(30) << "\n";
    }
}

void GraphMesh::buildSplitAndPrintMesh(const std::string& filename, double targetPartSize, double nodeEdgePortion) {
    loadMeshFromFile(filename);

    buildGraphFromMesh();

    std::cout << "Initial Graph State:" << std::endl;
    printGraphState();

    PartGraph partGraph = splitMesh(targetPartSize, nodeEdgePortion);

    std::cout << "\nSplit Mesh State:" << std::endl;
    printPartGraphState(partGraph);

    std::cout << "\nPart Geometries:" << std::endl;
    int count = 0;
    for (const auto& part : partGraph.parts) {
        std::cout << "Part Type: " << (part.type == MeshPart::NODE ? "Node" : "Edge")
                  << ", ID: " << part.id << ", SubID: " << part.subId << ", Part Index: " << count << std::endl;

        auto geometry = getPartGeometry(part);
        std::cout << "Value of poly_points is\n" << geometry << std::endl;
        std::cout << std::endl;
        count++;
    }

    auto training_params = getNNTrainingParams(partGraph);
    printTrainingParams(training_params);

    auto conditions = getCompatibilityConditions(partGraph);
    printCompatibilityConditions(conditions);

    auto geom_points = getGeometryPolynomialPoints(partGraph);

    auto [mesh_points, mesh_indices] = getMeshPolynomialPoints(partGraph, "testNE1", 1);
    printMeshData(geom_points, mesh_points, mesh_indices);

    auto mesh_parametrizations = getMeshParametrizations(partGraph, "testNE1", 1);
    auto geom_parametrizations = getGeometryParametrizations(partGraph);

    compareMeshParametrizationData(geom_parametrizations, mesh_parametrizations);

    auto centered_mesh_parametrizations = centerMeshParametrizations(mesh_parametrizations);
    compareMeshParametrizationData(mesh_parametrizations, centered_mesh_parametrizations);

    auto centered_mesh_points = centerPointParametrizations(mesh_points);
    printMeshData(mesh_points, centered_mesh_points, mesh_indices);
}

void GraphMesh::printGraphState() {
    std::cout << "Nodes:" << std::endl;
    for (const auto& [nodeId, node] : nodes) {
        std::cout << "  Node " << nodeId << ":" << std::endl;
        std::cout << "    Surface Tags: ";
        for (int tag : node.surfaceTags) std::cout << tag << " ";
        std::cout << "\n    Boundary Tags: ";
        for (int tag : node.boundaryTags) std::cout << tag << " ";
        std::cout << "\n    Connected Edges: ";
        for (int edge : node.connectedEdges) std::cout << edge << " ";
        std::cout << std::endl;
    }

    std::cout << "\nEdges:" << std::endl;
    for (const auto& [edgeId, edge] : edges) {
        std::cout << "  Edge " << edgeId << ":" << std::endl;
        std::cout << "    Surface Tags: ";
        for (int tag : edge.surfaceTags) std::cout << tag << " ";
        std::cout << "\n    Boundary Tags: ";
        for (int tag : edge.boundaryTags) std::cout << tag << " ";
        std::cout << "\n    Connected Nodes: ";
        for (int node : edge.connectedNodes) std::cout << node << " ";
        std::cout << std::endl;
    }
}

void GraphMesh::printPartGraphState(const PartGraph& partGraph) {
    for (size_t i = 0; i < partGraph.parts.size(); ++i) {
        const auto& part = partGraph.parts[i];
        std::cout << "Part " << i << " - Type: " << (part.type == MeshPart::NODE ? "Node" : "Edge")
                  << ", ID: " << part.id << ", SubID: " << part.subId << std::endl;
        std::cout << "  Surface Tags: ";
        for (int tag : part.surfaceTags) std::cout << tag << " ";
        std::cout << "\n  Boundary Tags: ";
        for (int tag : part.boundaryTags) std::cout << tag << " ";
        std::cout << "\n  Connected Edges: ";
        for (const auto& [edgeId, portion] : part.connectedEdges)
            std::cout << "(" << edgeId << ": " << portion.first << "-" << portion.second << ") ";
        std::cout << "\n  Curve Tags: ";
        for (const auto& [tag, orientation, portions] : part.curveTags)
            std::cout << "(" << tag << ": " << orientation << ", (" << portions.first << ", " << portions.second << ")) ";
        std::cout << "\n  Adjacent Parts: ";
        for (int adj : partGraph.adjacencyList[i]) std::cout << adj << " ";
        std::cout << std::endl;
    }
}

void GraphMesh::printCompatibilityConditions(const std::vector<CompatibilityCondition> &conditions) {
    std::cout << "Compatibility Conditions:\n";
    for (size_t i = 0; i < conditions.size(); ++i) {
        const auto& condition = conditions[i];
        std::cout << "Condition " << i + 1 << ":\n";
        std::cout << "  Indices:         [" << condition.indices.first << ", " << condition.indices.second << "]\n";
        std::cout << "  First Location:  [Branch: " << condition.firstLocation.first
                  << ", Side: " << condition.firstLocation.second << "]\n";
        std::cout << "  Second Location: [Branch: " << condition.secondLocation.first
                  << ", Side: " << condition.secondLocation.second << "]\n";
        std::cout << "\n";
    }
}

void GraphMesh::printTrainingParams (const NNTrainingParams &params) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "NNTrainingParams:\n";
    std::cout << "  Length range:     [" << params.minMaxLength.first << ", " << params.minMaxLength.second << "]\n";
    std::cout << "  Width range:      [" << params.minMaxWidth.first << ", " << params.minMaxWidth.second << "]\n";
    std::cout << "  Width diff range: [" << params.minMaxWidthDiff.first << ", " << params.minMaxWidthDiff.second << "]\n";
    std::cout << "  Angle diff range: [" << params.minMaxAngleDiff.first << ", " << params.minMaxAngleDiff.second << "]\n";
}

void GraphMesh::printMeshData(const std::vector<Eigen::MatrixXd>& geom_poly_points,
        const std::vector<Eigen::MatrixXd>& mesh_poly_points,
        const std::vector<Eigen::MatrixXi>& mesh_node_indices) {

    for (size_t i = 0; i < geom_poly_points.size(); ++i) {
        std::cout << "Matrix " << i + 1 << ":" << std::endl;
        const auto& geom_points = geom_poly_points[i];
        const auto& mesh_points = mesh_poly_points[i];
        const auto& indices = mesh_node_indices[i];

        std::cout << std::setw(15) << "Geom X" << std::setw(15) << "Geom Y"
                  << std::setw(15) << "Mesh X" << std::setw(15) << "Mesh Y"
                  << std::setw(15) << "Index" << std::setw(15) << "Distance" << std::endl;

        for (int row = 0; row < indices.rows(); ++row) {
            for (int col = 0; col < indices.cols(); ++col) {
                double geom_x = geom_points(2*row, col);
                double geom_y = geom_points(2*row+1, col);
                double mesh_x = mesh_points(2*row, col);
                double mesh_y = mesh_points(2*row+1, col);

                double distance = std::sqrt(std::pow(geom_x - mesh_x, 2) + std::pow(geom_y - mesh_y, 2));

                std::cout << std::setw(15) << std::fixed << std::setprecision(6) << geom_x
                          << std::setw(15) << std::fixed << std::setprecision(6) << geom_y
                          << std::setw(15) << std::fixed << std::setprecision(6) << mesh_x
                          << std::setw(15) << std::fixed << std::setprecision(6) << mesh_y
                          << std::setw(15) << indices(row, col)
                          << std::setw(15) << std::fixed << std::setprecision(6) << distance << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

void GraphMesh::compareMeshParametrizationData(const std::vector<MeshParametrizationData>& data1,
                                    const std::vector<MeshParametrizationData>& data2) {
    for (size_t i = 0; i < data1.size(); ++i) {
        const auto& d1 = data1[i];
        const auto& d2 = data2[i];

        std::cout << "Comparison of MeshParametrizationData " << i + 1 << ":\n";

        // Compare numBranches
        std::cout << std::setw(20) << "numBranches:" << std::setw(10) << d1.numBranches
                  << std::setw(10) << d2.numBranches << "\n\n";

        // Compare widths
        std::cout << "Widths:\n";
        printMatrixComparison(d1.widths, d2.widths);
        // Compare terminals
        std::cout << "Terminals:\n";
        printMatrixComparison(d1.terminals, d2.terminals);
        // Compare vectors
        std::cout << "Vectors:\n";
        printMatrixComparison(d1.vectors, d2.vectors);
        std::cout << "\n";
    }
}

void GraphMesh::printMatrixComparison(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2) {
    int width = 15;
    int precision = 6;
    // Print column headers
    std::cout << std::string(m1.cols() * width, ' ') << " | " << std::string(m2.cols() * width, ' ') << "\n";
    for (int i = 0; i < m1.rows(); ++i) {
        // Print row of first matrix
        for (int j = 0; j < m1.cols(); ++j) {
            std::cout << std::setw(width) << std::fixed << std::setprecision(precision) << m1(i, j);
        }
        std::cout << " | ";
        // Print row of second matrix
        for (int j = 0; j < m2.cols(); ++j) {
            std::cout << std::setw(width) << std::fixed << std::setprecision(precision) << m2(i, j);
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
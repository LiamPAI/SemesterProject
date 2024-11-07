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
/**
 * @brief Container for mesh part compatibility conditions
 * @details Stores information about how different parts of a split mesh relate to each other by providing indices
 * for which displacement vectors, points, or terminals must be equal
 */
struct CompatibilityCondition {
    std::pair<int, int> indices; ///< Indices of connected parts (first, second)
    std::pair<int, int> firstLocation; ///< Location in first part (branch number, side)
    std::pair<int, int> secondLocation; ///< Location in second part (branch number, side)

    /**
     * @brief Constructs compatibility condition between mesh parts
     * @param inds Part indices pair
     * @param firstLoc Location in first part
     * @param secondLoc Location in second part
     */
    CompatibilityCondition(std::pair<int, int> inds, std::pair<int, int> firstLoc, std::pair<int, int> secondLoc)
        : indices(std::move(inds)),
        firstLocation(std::move(firstLoc)),
        secondLocation(std::move(secondLoc)) {}
};

// TODO: Make sure that this defined in the pybinds, as I will be returning it
// This struct is used to provide the displacement boundary conditions of the mesh, designed to be used with the NN
// to run the overall FEM calculation
/**
 * @brief Container for fixed displacement boundary conditions
 * @details Stores displacement boundary conditions for neural network FEM calculations by providing indices for
 * which to extract the exact displacement BC
 */
struct FixedDisplacementCondition {
    std::pair<int, int> indices;    ///< Part index and side index
    Eigen::Vector4d displacements;  ///< Prescribed displacements

    /**
     * @brief Constructs displacement boundary condition
     * @param inds Part and side indices
     * @param disps Displacement vector
     */
    FixedDisplacementCondition(std::pair<int, int> inds, Eigen::Vector4d disps)
        : indices(std::move(inds)),
        displacements(std::move(disps)) {}
};

// The two structs below are used when generating the compatibility conditions to ensure each condition is unique
/**
 * @brief Hash function for integer pairs
 * @details Provides unique hashing for unordered pairs of integers to store compatibility conditions
 */
struct IntPairHash {
    std::size_t operator()(const std::pair<int, int> &p) const {
        int a = std::min(p.first, p.second);
        int b = std::max(p.first, p.second);
        return ((a + b) * (a + b + 1) / 2) + b;
    }
};

/**
 * @brief Equality comparison for integer pairs
 * @details Treats pairs as unordered (a,b) equals (b,a), useful when storing the compatibility conditions uniquely
 */
struct IntPairEqual {
    bool operator()(const std::pair<int, int> &left, const std::pair<int, int> &right) const {
        return (left.first == right.first and left.second == right.second) or
        (left.first == right.second and left.second == right.first);
    }
};

// TODO: When actually training the NN, decide if these 4 pairs will end up being enough
// The following struct is used in the method getNNTrainingParams to obtain some of the physical ranges
// we should be training the NN on
/**
 * @brief Parameters for neural network training
 * @details Contains physical parameter ranges for training neural networks
 *          on mesh geometries
 */
struct NNTrainingParams {
    std::pair<double, double> minMaxLength;    ///< Min/max length range
    std::pair<double, double> minMaxWidth;      ///< Min/max width range
    std::pair<double, double> minMaxWidthDiff;  ///< Min/max width difference range
    std::pair<double, double> minMaxAngleDiff;  ///< Min/max angle difference range

    /**
     * @brief Constructs training parameters with ranges
     * @param length Length range
     * @param width Width range
     * @param widthDiff Width difference range
     * @param angleDiff Angle difference range
     */
    NNTrainingParams(std::pair<double, double> length, std::pair<double, double> width,
        std::pair<double, double> widthDiff, std::pair<double, double> angleDiff)
        : minMaxLength(std::move(length)),
        minMaxWidth(std::move(width)),
        minMaxWidthDiff(std::move(widthDiff)),
        minMaxAngleDiff(std::move(angleDiff)) {}
};

/**
 * @brief 2D node in mesh graph
 * @details Represents a vertex in the mesh with tags of its surface, tags of the boundary, and
 * connectivity information relative to edges
 */
struct Node2D {
    int id;                         ///< Node identifier
    std::vector<int> surfaceTags;   ///< Associated surface tags
    std::vector<int> boundaryTags;  ///< Associated boundary tags
    std::set<int> connectedEdges;   ///< Connected edge identifiers
};

/**
 * @brief 2D edge in mesh graph
 * @details Represents an edge in the mesh with tags of its surface, tags of the boundary, and
 * connectivity information relative to nodes
 */
struct Edge2D {
    int id;                         ///< Edge identifier
    std::vector<int> surfaceTags;   ///< Associated surface tags
    std::vector<int> boundaryTags;  ///< Associated boundary tags
    std::vector<int> connectedNodes;///< Connected node identifiers
};

// TODO: Ensure that this is all we need when defining the boundary line of the graph
/**
 * @brief Boundary line information
 * @details Represents a boundary segment including its physical tag and curve tag for identification
 */
struct BoundaryLine {
    int id;             ///< Boundary identifier
    int physicalTag;    ///< Physical group tag
    int curveTag;       ///< Curve identifier
    std::pair<Eigen::Vector2d, Eigen::Vector2d> points;///< Endpoint coordinates
};

/**
 * @brief Part of a subdivided mesh, either node or edge
 * @details Represents either a node or edge segment in the subdivided mesh and all of its associated information
 * to extract the geometry and connectivity relative to other parts
 */
struct MeshPart {
    enum Type {NODE, EDGE} type;    ///< Part type (node or edge)
    int id;  ///< Primary identifier (Node or Edge ID)
    int subId;  ///< Subdivision identifier (used for multiple edge parts on the same edge)
    std::vector<int> surfaceTags; ///< Associated surface tags
    std::vector<int> boundaryTags;  ///< Associated boundary tags
    std::vector<std::pair<int, std::pair<double, double>>> connectedEdges;  ///< Edge ID, (start, end) portions
    std::vector<int> connectedEdgeTags;  ///< Edge tags (for NODE type only)
    std::vector<std::tuple<int, int, std::pair<double, double>>> curveTags; ///< Curve tag, orientation (1 or -1), and portion
};

/**
 * @brief Graph representation of subdivided mesh
 * @details Contains mesh parts and their connectivity information
 */
struct PartGraph {
    std::vector<MeshPart> parts; ///< Mesh parts
    std::vector<std::vector<size_t>> adjacencyList; ///< Connectivity graph
};

/**
 * @brief Class for managing 2D graph-like meshes
 * @details Handles mesh loading, processing, and analysis for FEM calculations
 *          and neural network training. Supports mesh subdivision and
 *          parametrization generation.
 */
class GraphMesh {
private:
    std::map<int, Node2D> nodes;        ///< Map of node IDs to nodes
    std::map<int, Edge2D> edges;        ///< Map of edge IDs to edges
    std::map<int, BoundaryLine> boundaries; ///< Map of boundary IDs to boundary lines

    /**
     * @brief Checks if name represents a boundary condition
     * @details A physical group being labeled "BC#" would identify a boundary condition
     * @param name String to check
     * @return true if name is a BC identifier
     */
    bool isBC(const std::string &name);

    /**
     * @brief Checks if name represents a node
     * @details A physical group being labeled "N#" would identify a node
     * @param name String to check
     * @return true if name is a node identifier
     */
    bool isNode(const std::string &name);

    /**
     * @brief Checks if name represents a node boundary
     * @details A physical group being labeled "NB#" would identify a node boundary
     * @param name String to check
     * @return true if name is a node boundary identifier
     */
    bool isNodeBoundary(const std::string &name);

    /**
     * @brief Checks if name represents an edge
     * @details A physical group being labeled "E#" would identify an edge
     * @param name String to check
     * @return true if name is an edge identifier
     */
    bool isEdge(const std::string &name);

    /**
     * @brief Checks if name represents an edge boundary
     * @details A physical group being labeled "EB#" would identify an edge boundary
     * @param name String to check
     * @return true if name is an edge boundary identifier
     */
    bool isEdgeBoundary(const std::string &name);

    /**
     * @brief Extracts ID number from physical group name
     * @param name String containing ID
     * @return Extracted ID number
     */
    int extractId(const std::string &name);

    /**
     * @brief Extracts boundary condition ID from name
     * @param name String containing BC ID
     * @return Extracted BC ID number
     */
    int extractBCId(const std::string &name);

public:
    /** @brief Default constructor */
    GraphMesh();

    /** @brief Destructor */
    ~GraphMesh();

    /**
     * @brief Loads mesh from file
     * @param filename Path to mesh file (must include the .geo extension)
     */
    void loadMeshFromFile(const std::string& filename);

    /**
     * @brief Finalizes mesh construction
     */
    void closeMesh();

    /**
     * @brief Constructs graph representation from mesh
     * @details In doing so, it uses the opened mesh from @link loadMeshFromFile @endlink to initialize the fields
     * of the @link GraphMesh @endlink class
     */
    void buildGraphFromMesh();

    // getter methods for easy retrieval if needed
    /**
     * @brief Gets edges connected to a node
     * @details Uses @link GraphMesh @endlink fields to obtain the connected edges
     * @param nodeId Node identifier
     * @return Set of connected edge IDs
     */
    std::set<int> getConnectedEdges(int nodeId);

    /**
     * @brief Gets nodes connected to an edge
     * @details Uses @link GraphMesh @endlink fields to obtain the connected nodes
     * @param edgeId Edge identifier
     * @return Vector of connected node IDs
     */
    std::vector<int> getNodesOfEdge(int edgeId);

    /**
     * @brief Gets nodes connected to a node
     * @details Uses @link GraphMesh @endlink fields to obtain the connected nodes
     * @param nodeId Node identifier
     * @return Set of connected node IDs
     */
    std::set<int> getConnectedNodes(int nodeId);

    /**
     * @brief Gets surface tags for a node
     * @details Uses @link GraphMesh @endlink fields to obtain the node surface tags
     * @param nodeId Node identifier
     * @return Vector of surface tags
     */
    std::vector<int> getNodeSurfaces(int nodeId);

    /**
     * @brief Gets boundary tags for a node
     * @details Uses @link GraphMesh @endlink fields to obtain the node boundary tags
     * @param nodeId Node identifier
     * @return Vector of boundary tags
     */
    std::vector<int> getNodeBoundary(int nodeId);

    /**
     * @brief Gets surface tags for an edge
     * @details Uses @link GraphMesh @endlink fields to obtain the edge surface tags
     * @param edgeId Edge identifier
     * @return Vector of surface tags
     */
    std::vector<int> getEdgeSurfaces(int edgeId);

    /**
     * @brief Gets boundary tags for an edge
     * @details Uses @link GraphMesh @endlink fields to obtain the edge boundary tags
     * @param edgeId Edge identifier
     * @return Vector of boundary tags
     */
    std::vector<int> getEdgeBoundary(int edgeId);


    // The functions below help with splitting up the mesh and obtaining the required parametrizations
    /**
     * @brief Splits mesh into parts
     * @details The algorithm begins from the nodes and initializes those parts first according to the provided
     * nodeEdgePortion, which must be under 0.5. Afterwards, the remaining edge parts are initialized according to the
     * provided targetPartSize
     * @param targetPartSize Target size for parts (in units of length)
     * @param nodeEdgePortion Portion (percentage) of edges connected to this node to include
     * @return Graph of mesh parts
     */
    PartGraph splitMesh(double targetPartSize, double nodeEdgePortion);

    /**
     * @brief Gets geometry for a mesh part
     * @details Uses the stored curve tags along with the respective portions to query the curve and obtain the points
     * @param part Mesh part to analyze
     * @return Matrix of geometric points
     */
    Eigen::MatrixXd getPartGeometry(const MeshPart &part);

    // The following functions all help to obtain the necessary data structures to train the NN and run the "FEM"
    // calculation for linear elasticity
    /**
     * @brief Gets polynomial points for the geometry
     * @details These geometry points are on the actual splines, but may not be points on the actual mesh
     * @param part_graph Graph of mesh parts
     * @return Vector of point matrices for each part, in the same order as in the PartGraph
     */
    std::vector<Eigen::MatrixXd> getGeometryPolynomialPoints(const PartGraph &part_graph);

    /**
     * @brief Gets polynomial points for the mesh
     * @details The returned points are on the actual mesh, and not solely on the geometry
     * @param part_graph Graph of mesh parts
     * @param mesh_file_name Mesh file path
     * @param order Polynomial order
     * @return Pair of point matrices and node indices for future retrieval
     */
    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXi>> getMeshPolynomialPoints(
        const PartGraph &part_graph, const std::string &mesh_file_name, int order);

    /**
    * @brief Gets parametrizations for the geometry
    * @details The returned vector of @link MeshParametrizationData @endlink is semantically equivalent to what is
    * returned from @link getGeometryPolynomialPoints @endlink, but in a different type form
    * @param part_graph Graph of mesh parts
    * @return Vector of parametrization data
    */
    std::vector<MeshParametrizationData> getGeometryParametrizations(const PartGraph &part_graph);

    /**
      * @brief Gets parametrizations for mesh
      * @details The returned vector of @link MeshParametrizationData @endlink is semantically equivalent to what is
      * returned from @link getMeshPolynomialPoints @endlink, but in a different type form
      * @param part_graph Graph of mesh parts
      * @param mesh_file_name Mesh file path
      * @param order Polynomial order
      * @return Vector of parametrization data
      */
    std::vector<MeshParametrizationData> getMeshParametrizations(const PartGraph &part_graph,
        const std::string &mesh_file_name, int order);

    /**
     * @brief Gets initial displacement vectors
     * @details Currently, the initial displacement vectors are all initialized to 0, as a first guess for
     * optimize_displacement_vectors
     * @param part_graph Graph of mesh parts
     * @return Vector of displacement matrices
     */
    std::vector<Eigen::MatrixXd> getInitialDisplacements(const PartGraph &part_graph);

    /**
     * @brief Centers parametrizations containing the polynomial points
     * @details The returned vector is useful when optimizing the displacement vectors on a trained neural network. By
     * centering these parametrizations at the origin, we ensure translation-invariance of the neural network.
     * @param point_vector Vector of point matrices
     * @return Vector of centered point matrices
     */
    static std::vector<Eigen::MatrixXd> centerPointParametrizations(const std::vector<Eigen::MatrixXd> &point_vector);

    /**
     * @brief Centers parametrizations of type @link MeshParametrizationData @endlink
     * @details The returned vector is useful when optimizing the displacement vectors on a trained neural network. By
     * centering these parametrizations at the origin, we ensure translation-invariance of the neural network.
     * @param param_vector Vector of parametrization data
     * @return Vector of centered parametrization data
     */
    static std::vector<MeshParametrizationData> centerMeshParametrizations(
        const std::vector<MeshParametrizationData> &param_vector);

    /**
     * @brief Gets training parameters for neural network
     * @details The included training parameters are detailed in @link NNTrainingParams @endlink
     * @param part_graph Graph of mesh parts
     * @return Training parameter ranges
     */
    NNTrainingParams getNNTrainingParams(const PartGraph &part_graph);

    /**
     * @brief Gets compatibility conditions between parts
     * @details Adjacent graph parts will need to have identical displacement vectors on the line that they share,
     * this function returns a vector of data structures containing indices such that these conditions can be checked
     * @param part_graph Graph of mesh parts
     * @return Vector of compatibility conditions
     */
    std::vector<CompatibilityCondition> getCompatibilityConditions(const PartGraph &part_graph);

    /**
     * @brief Gets fixed displacement conditions
     * @details Use the additional "free" edges that do not have compatbility conditions, displacement boundary
     * conditions can be fixed with this function
     * @param part_graph Graph of mesh parts
     * @param displacements Vector of numeric displacement specifications, the integer is the id of the physical group
     * and the vector contains the actual displacement boundary conditions
     * @return Vector of displacement conditions of type @link FixedDisplacementCondition @endlink
     */
    std::vector<FixedDisplacementCondition> getFixedDisplacementConditions(
        const PartGraph &part_graph, const std::vector<std::pair<int, Eigen::Vector4d>> &displacements);

    /**
     * @brief Performs FEM calculation on a larger mesh
     * @param mesh_file_name Mesh file path
     * @param displacements Vector of displacement specifications
     * @param calc_params Calculation parameters
     * @return Pair of displacement vector solution and total energy energy
     */
    std::pair<Eigen::VectorXd, double> meshFEMCalculation(const std::string &mesh_file_name,
        const std::vector<std::pair<int, Eigen::Vector4d>> &displacements, const calculationParams &calc_params);

    /**
     * @brief Computes displacement vectors for each mesh parametrizations
     * @details Uses the result from @link meshFEMCalculation @endlink to return all the displacement vectors for
     * each parametrization sent to the neural network, in the same order as in the PartGraph
     * @param part_graph Graph of mesh parts
     * @param mesh_file_name Mesh file path
     * @param fixed_displacements Fixed displacement conditions
     * @param calc_params Calculation parameters
     * @return Pair of vectors, the first represents the displacement vectors, the second the equivalent parametrization
     */
    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> meshDisplacementVectors(
        const PartGraph &part_graph, const std::string &mesh_file_name,
        const std::vector<std::pair<int, Eigen::Vector4d>> &fixed_displacements, const calculationParams &calc_params);

    /**
     * @brief Computes energy change for each mesh parametrization individually according to displacement vectors
     * @details Uses the result from @link meshDisplacementVectors @endlink to compute the energy change associated
     * with the displacement vectors for each parametrization. Note that due to the non-interactions between adjacent
     * parts, the overall energy will be higher
     * @param part_graph Graph of mesh parts
     * @param mesh_file_name Mesh file path
     * @param fixed_displacements Fixed displacement conditions
     * @param calc_params Calculation parameters
     * @return Tuple of displacement vectors, parametrization points, and energy changes
     */
    std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>, std::vector<double>> meshEnergy(
        const PartGraph &part_graph, const std::string &mesh_file_name,
        const std::vector<std::pair<int, Eigen::Vector4d>> &fixed_displacements, const calculationParams &calc_params);

    // TODO: Perhaps also implement a method that will take two of these, and analyze them/print out their contents,
    //  e.g. with energy norm, displacement norm, and so on (perhaps find other ways as well to analyze them)
    /**
    * @brief Gets length of an edge
    * @details Repeatedly queries the spline curves of this edge to obtain an approximate length
    * @param edgeId Edge identifier
    * @return Edge length
    */
    double getEdgeLength(int edgeId);

    /**
     * @brief Finds shared boundary line between an edge an a node
     * @details As specified in the geometry of an overall mesh, a curve of type Line will separate an edge from
     * its node part, and this functions finds the tag of that line to know the direction of orientation of node -> edge
     * @param nodeBoundaryTags Node boundary tags
     * @param edgeBoundaryTags Edge boundary tags
     * @return Shared line tag
     */
    int findSharedLine(const std::vector<int>& nodeBoundaryTags, const std::vector<int>& edgeBoundaryTags);

     /**
      * @brief Finds spline curves in boundary tags of an edge
      * @details Knowing that the only allowed types for the spline are of type "Nurb", the curves are returned
      * accordingly
      * @param edgeBoundaryTags Edge boundary tags
      * @return Vector of spline curve tags
      */
    std::vector<int> findSplineCurves(const std::vector<int>& edgeBoundaryTags);

     /**
      * @brief Determines spline orientation relative to node
      * @details Splines are parametrized in Gmsh on the interval [0, 1], this function helps to determine on which
      * node the 0 begins, so we can query the correct portion of a spline when extrapolating a part's geometry
      * @param splineTag Spline tag
      * @param sharedLineTag Shared line tag
      * @return Orientation indicator (1 or -1)
      */
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
    void printMeshParamsAndDisplacements(const std::vector<MeshParametrizationData> &mesh_params,
                                        const std::vector<Eigen::MatrixXd> &displacements);
    void printMeshParamsAndDisplacementsAndEnergies(const std::vector<MeshParametrizationData> &mesh_params,
                                  const std::vector<Eigen::MatrixXd> &displacements,
                                  const std::vector<double> &energies);
    void printMeshParamsAndDisplacementsAndTrueEnergies(const std::vector<MeshParametrizationData> &mesh_params,
                               const std::vector<Eigen::MatrixXd> &displacements,
                               const std::vector<double> &NN_energies, const std::vector<double> &true_energies);
    void printMeshPointsAndDisplacements(const std::vector<ParametrizationPoints> &mesh_points,
                                const std::vector<Eigen::MatrixXd> &displacements,
                                const std::vector<double> &energies);
    void printMeshMatricesAndDisplacements(const std::vector<Eigen::MatrixXd> &mesh_points,
                                const std::vector<Eigen::MatrixXd> &displacements,
                                const std::vector<double> &NN_energies, const std::vector<double> &true_energies);
    void printFixedDisplacementConditions(const std::vector<FixedDisplacementCondition>& conditions);
    void printMeshParamsAndEnergies(const std::tuple<std::vector<Eigen::MatrixXd>,
                                                 std::vector<Eigen::MatrixXd>,
                                                 std::vector<double>>& data);
};

#endif //METALFOAMS_GRAPH_MESH_H

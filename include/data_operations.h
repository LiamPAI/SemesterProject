//
// Created by Liam Curtis on 2024-09-12.
//

#ifndef METALFOAMS_DATA_OPERATIONS_H
#define METALFOAMS_DATA_OPERATIONS_H

#include "mesh_parametrization.h"
#include <utility>
#include <vector>
#include <string>
#include <variant>
#include <random>
#include <unordered_set>
#include <fstream>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// The purpose of this file is to introduce the data structures and methods required to generate, save, and manipulate
// the parametrization datasets for the NNs to be trained on, the goal is for the datasets to be relatively
// customizable and diverse so I can observe what aspects the NN handles well or not

// TODO: Consider adding a function that actually "flips" the parametrizations, which I believe will be different
//  from a rotation, but check first how the NN performs w.r.t rotation invariance
// TODO: find out if this FEM problem is scale-invariant, if so find out how to scale the parametrization
//  entries accordingly
// TODO: IDEA: If it leaves the linear elastic region, make the energy difference high such that the NN wants to
//  avoid it? do this depending on the results from testing/training the neural network



// Overload the () operator so that we have a hash function for std::vector<int>,
// useful when creating the point data set so we don't repeat flipped vector entries in the parametrization dataset
/**
* @brief Hash functor for vectors of integers
* @details Useful for dealing with tags in the @link ParametrizationEntry @endlink and @link PointEntry @endlink
*/
struct VectorHash {
    /**
    * @brief Compute hash value for a vector of integers
    * @param v Vector to be hashed
    * @return size_t Hash value of the vector
    */
    size_t operator()(const std::vector<int>& v) const {
        std::hash<int> hasher;
        size_t seed = 0;
        for (int i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2); // "Fowler-Noll-Vo" hash function
        }
        return seed;
    }
};

/// @ brief Total number of @link TagIndex @endlink entries
constexpr size_t NUM_TAGS = 4; // Amount of tags

// Meaning behind each of the tags, they indicate the index we use within each entry for a datset
/**
* @brief Indices for different tag types
*/
enum TagIndex {
    BASE = 0,               ///< Base Parametrization
    BASE_PERTURBATION = 1,  ///< Perturbation from base parametrization
    DISPLACEMENT = 2,       ///< Displacement BC for perturbation
    ROTATION = 3,           ///< Rotation for each displacement BC
};

// Another struct that contains the exact same information as MeshParametrization butter, but only the 6 points of
// each branch, to see how well the NN performs on both
/**
* @brief Container for parametrization of type points
* @details The points matrix follows a strict local ordering, specifically for parametrizations with 3+ branches
*
* Matrix Structure:
* For a parametrization with n branches, the points matrix has dimensions (4n * 3):
* - 4n rows: the top 2 rows all represent one side of the parametrization, the bottom rows represent the points
* on the other side
* - 3 columns: For the 3 parts of the parametrization that we are taking points from, where the middle column
* represents the "middle" points of the parametrization
*
* Local Ordering Convention:
* For multi-branch parametrizations, column 0 contains the points related to the "center" shape, such as a triangle
* a 3 branch multi parametrization
*   - Top 4 rows represent the 1st branch, the second 4 rows represent the 2nd branch, and so on
*/
struct ParametrizationPoints {
    int numBranches;        ///< Number of branches in the parametrization(1, 3+)
    Eigen::MatrixXd points; ///< Matrix containing the locally-ordered point data

    /**
    * @brief Constructs parametrization points for a given number of branches
    * @param num Number of branches
    * @param p Matrix of points for this parametrization (must have 4 * num rows)
    * @throws LFException if matrix dimensions don't match the branch count
    */
    ParametrizationPoints(int num, Eigen::MatrixXd p) : numBranches(num), points(std::move(p)) {
        LF_ASSERT_MSG(numBranches == points.rows() / 4,
                      "Number of branches is not equal to the correct number of rows when initializing "
                      "ParametrizationPoints, num is " << numBranches <<
                      " and points has " << points.rows() << " rows");
    }
};

// Instead of the parametrization, the actual polynomial points represented by the parametrization are used here, this
// is for the purpose of testing both methods to see which works better when training the NN
/**
* @brief Container for a complete data entry containing @link ParametrizationPoints @endlink
* @details  Stores a data entry of type @link ParametrizationPoints @endlink along with the displacement vectors,
* energy difference, and the vector of tags
*
* Local Ordering Convention:
* - Displacements: (4 x 2) for single-branch parametrizations, and (4n x 1) for multi-branch parametrizations
*
* -Single-branch parametrizations:
*   Given the 4 x 3 structure of the equivalent points matrix the left column of displacements will correspond to the
*   left points in the points matrix, and vice versa for the right side
* -Multi-branch parametrizations:
*   Given that the left-side points of the points matrix correspond to the "center" points of the parametrization, the
*   displacements matrix corresponds to the right side of the points matrix, with each point being matched up directly,
*   as they contain the same amount of rows
*/
struct PointEntry {
    int numBranches;                ///< Number of branches in the parametrization
    ParametrizationPoints points;   ///< Points containing the geometry of the parametrization
    Eigen::MatrixXd displacements;  ///< Matrix of displacement vectors as boundary conditions
    double energyDifference;        ///< Energy difference of the FEM problem
    std::vector<int> tags;          ///< Vector of tags to identify the data entry

    /**
    * @brief Constructs a complete parametrization entry
    * @param num Number of branches
    * @param p Parametrization containing points
    * @param disp Displacement vectors as boundary conditions
    * @param eD Energy difference
    * @param tags Vector of classification tags
    */
    PointEntry(int num, ParametrizationPoints p, Eigen::MatrixXd disp, double eD, std::vector<int> tags)
        : numBranches(num), points(std::move(p)), displacements(std::move(disp)), energyDifference(eD),
        tags(std::move(tags)) {}
};

// This data entry simply uses the mesh parametrizations as defined in mesh_parametrization.h
/**
* @brief Container for a complete data entry containing @link MeshParametrization @endlink
* @details Stores a data entry of type @link MeshParametrization @endlink along with the displacement vectors,
* energy difference, and the vector of tags
*
* Local Ordering Convention:
* - Displacements: (4 x 2) for single-branch parametrizations, and (4n x 1) for multi-branch parametrizations
*       Given the semantic equivalence between @link MeshParametrizationData @endlink and @link ParametrizationPoints
*       @endlink, the displacements matrix must be ordered the exact same as described for @link PointEntry @endlink
*/
struct ParametrizationEntry {
    int numBranches;                   ///< Number of branches in the parametrization
    MeshParametrizationData param;     ///< Parametrization containing the geometry of the branch
    Eigen::MatrixXd displacements;     ///< Matrix of displacement vectors as boundary conditions
    double energyDifference;           ///< Energy difference of the FEM problem
    std::vector<int> tags;             ///< Vector of tags to identify the data entry

    /**
    * @brief Constructs a complete mesh parametrization data entry
    * @param num Number of branches
    * @param p Mesh parametrization
    * @param disp Displacement vectors as boundary conditions
    * @param eD Energy difference
    */
    ParametrizationEntry(int num, MeshParametrizationData p, Eigen::MatrixXd disp, double eD)
        : numBranches(num), param(std::move(p)), displacements(std::move(disp)), energyDifference(eD),
        tags(NUM_TAGS, -1) {}
};

// The datasets are simply vectors of the original entries and their types
/**
* @brief Type alias for a collection of mesh parametrization entries
* @see ParametrizationEntry For individual entry structure
*/
using ParametrizationDataSet = std::vector<ParametrizationEntry>;

/**
* @brief Type alias for a collection of point entries
* @see PointEntry For individual entry structure
*/
using PointDataSet = std::vector<PointEntry>;

// enum for storing the dataset in a binary file and identifying which is which
/**
* @brief Enumeration for dataset type identification in binary storage
*/
enum class DataSetType {
    Parametrization, ///< Dataset contains entries of type @link ParametrizationEntry @endlink
    Point           ///< Dataset contains entries of type @link PointEntry @endlink
};

// A dataset can contain either type of entry
/**
* @brief Type alias for a variant that can hold either type of dataset
* @see ParametrizationDataSet For mesh-based datasets
* @see PointDataSet For point-based datasets
*/
using DataSet = std::variant<ParametrizationDataSet, PointDataSet>;

// A struct that will contain all the necessary parameters to construct a complete dataset
/**
* @brief Parameters for generating datasets containing type @link ParametrizationDataSet @endlink
* @details Contains all configuration parameters needed for generating parametrization datasets,
*          including size, number of branches, geometric, physical, and perturbation parameters
*/
struct GenerationParams {
    int datasetSize;                    ///< Size of dataset to generate
    int numBranches;                    ///< Number of branches (must be 1 or >= 3)
    std::pair<double, double> lengthInterval; ///< Min/max length interval [min, max]
    std::pair<double, double> widthInterval; ///< Min/max width interval [min, max]
    std::pair<double, int> flipParams; ///< Flip probability and linear scale factor
    std::pair<double, double> dataRotationParams; ///< Rotation granularity and random proportion
    double modulusOfElasticity;         ///< Young's modulus (E)
    double poissonRatio;                ///< Poisson's ratio (ν)
    double yieldStrength;               ///< Material yield strength
    int numPerturbations;               ///< Number of perturbations to generate
    double perturbProbability;          ///< Probability of applying perturbation [0,1]
    std::pair<double, double> widthPerturb; ///< (% current branch width, % branch length)
    std::pair<double, double> vectorPerturb; ///< (max angle change, reference branch length)
    std::pair<double, double> terminalPerturb; ///< (% branch length, % width)
    int numDisplacements; ///< Number of displacement BCs to generate
    std::pair<double, double> percentYieldStrength; ///< Min/max % of yield strength
    double displaceProbability; ///< Probability of applying displacement [0,1]
    double meshSize; ///< Size parameter for mesh generation
    int order; ///< Order of finite elements calculation (1 or 2)
    unsigned seed; ///< Random number generator seed
    std::mt19937 rng; ///< Random number generator

    /**
    * @brief Constructs generation parameters with validation
    * @param size Dataset size (>= 1)
    * @param numB Number of branches (1 or >= 3)
    * @param lenInterval Length interval [min, max]
    * @param widInterval Width interval [min, max]
    * @param flipP Flip parameters (probability, amount)
    * @param dataRotP Rotation parameters (granularity, proportion)
    * @param modulus Young's modulus
    * @param poisson Poisson's ratio
    * @param yieldStren Yield strength
    * @param numPer Number of perturbations
    * @param perProb Perturbation probability
    * @param widPer Width perturbation parameters
    * @param vecPer Vector perturbation parameters
    * @param termPer Terminal perturbation parameters
    * @param numDisp Number of displacements
    * @param perStrength Percent yield strength range
    * @param dispProb Displacement probability
    * @param meshS Mesh size
    * @param o Element order
    * @param s Random seed
    */
    GenerationParams(int size, int numB, std::pair<double, double> lenInterval, std::pair<double, double> widInterval,
                     std::pair<double, int> flipP, std::pair<double, double> dataRotP, double modulus, double poisson,
                     double yieldStren, int numPer, double perProb, std::pair<double, double> widPer,
                     std::pair<double, double> vecPer, std::pair<double, double> termPer,
                     int numDisp, std::pair<double, double> perStrength, double dispProb, double meshS, int o,
                     unsigned s = std::chrono::system_clock::now().time_since_epoch().count())
            : datasetSize(size),
            numBranches(numB),
            lengthInterval(std::move(lenInterval)),
            widthInterval(std::move(widInterval)),
            flipParams(std::move(flipP)),
            dataRotationParams(std::move(dataRotP)),
            modulusOfElasticity(modulus),
            poissonRatio(poisson),
            yieldStrength(yieldStren),
            numPerturbations(numPer),
            perturbProbability(perProb),
            widthPerturb(std::move(widPer)),
            vectorPerturb(std::move(vecPer)),
            terminalPerturb(std::move(termPer)),
            numDisplacements(numDisp),
            percentYieldStrength(perStrength),
            displaceProbability(dispProb),
            meshSize(meshS),
            order(o),
            seed(s),
            rng(seed)
    {
        LF_ASSERT_MSG(datasetSize >= 1, "Dataset size must be greater or equal to 1, but is instead " << datasetSize);
        LF_ASSERT_MSG(numBranches == 1 or numBranches >= 3, "Invalid number of branches sent to "
            "GenerationParams, being " << numBranches);
        LF_ASSERT_MSG(lengthInterval.second >= lengthInterval.first and lengthInterval.first > 1e-6, "Incorrect "
            "length interval sent to constructor of GenerationParams, [" <<
            lengthInterval.first << ", " << lengthInterval.second << "]");
        LF_ASSERT_MSG(widthInterval.second >= widthInterval.first and widthInterval.first > 1e-6, "Incorrect "
            "width interval sent to constructor of GenerationParams, [" <<
            widthInterval.first << ", " << widthInterval.second << "]");
        LF_ASSERT_MSG(flipParams.first >= 0 and flipParams.first <= 1 and flipParams.second >= 0, "Incorrect "
            "flip params sent to constructor of GenerationParams, [" <<
            flipParams.first << ", " << flipParams.second << "]");
        LF_ASSERT_MSG(dataRotationParams.first >= 1e-6 and dataRotationParams.first <= 360 and
            dataRotationParams.second >= 0, "Incorrect rotation params sent to constructor of GenerationParams, [" <<
            dataRotationParams.first << ", " << dataRotationParams.second << "]");
        LF_ASSERT_MSG(modulusOfElasticity > 0 and poissonRatio > 0 and yieldStrength > 0, "Physical parameters "
            "sent to GenerationParams incorrect, E: " << modulusOfElasticity << ", nu: " << poissonRatio <<
            ", sigma_y: " << yieldStrength);
        LF_ASSERT_MSG(numPerturbations > 0, "Invalid number of perturbations sent to Generation params");
        LF_ASSERT_MSG(perturbProbability >= 0 and perturbProbability <= 1, "Perturb probability is not on the "
            "interval [0, 1], it is instead " << perturbProbability);
        LF_ASSERT_MSG(widthPerturb.first >= 0 and widthPerturb.first <= 1 and widthPerturb.second >= 0 and
            widthPerturb.second <= 1, "Parameters sent to width_perturb in Generation params invalid");
        LF_ASSERT_MSG(vectorPerturb.first >= 0 and widthPerturb.second < 45 and widthPerturb.second >= 0,
            "Parameters sent to vector_perturb in Generation params invalid");
        LF_ASSERT_MSG(terminalPerturb.first >= 0 and terminalPerturb.first <= 1 and terminalPerturb.second >= 0 and
            terminalPerturb.second <= 1, "Parameters sent to terminal_perturb in Generation params invalid");
        LF_ASSERT_MSG(numDisplacements > 0, "Invalid number of displacements sent to Generation params");
        LF_ASSERT_MSG(percentYieldStrength.first >= 0 and percentYieldStrength.first <= 1 and
            percentYieldStrength.second >= 0 and percentYieldStrength.second <= 1 and
            percentYieldStrength.first <= percentYieldStrength.second, "Percent yield strength is not on the "
            "interval [0, 1], it is instead (" << percentYieldStrength.first << ", " << percentYieldStrength.second
            << ")\n");
        LF_ASSERT_MSG(displaceProbability >= 0 and displaceProbability <= 1, "Perturb probability is not on the "
            "interval [0, 1], it is instead " << displaceProbability);
        LF_ASSERT_MSG(meshSize > 1e-6, "Inavlid mesh size sent to Generation Params, " << meshSize);
        LF_ASSERT_MSG(order == 1 or order == 2, "Invalid order sent to Generation Params, " << order);
    }

    // The following two methods are included so that we can save the generation params when creating a neural network
    /**
    * @brief Converts parameters to Python dictionary
    * @return Dictionary containing all parameter values
    */
    [[nodiscard]] pybind11::dict to_dict() const {
        pybind11::dict d;
        d["datasetSize"] = datasetSize;
        d["numBranches"] = numBranches;
        d["lengthInterval"] = lengthInterval;
        d["widthInterval"] = widthInterval;
        d["flipParams"] = flipParams;
        d["dataRotationParams"] = dataRotationParams;
        d["modulusOfElasticity"] = modulusOfElasticity;
        d["poissonRatio"] = poissonRatio;
        d["yieldStrength"] = yieldStrength;
        d["numPerturbations"] = numPerturbations;
        d["perturbProbability"] = perturbProbability;
        d["widthPerturb"] = widthPerturb;
        d["vectorPerturb"] = vectorPerturb;
        d["terminalPerturb"] = terminalPerturb;
        d["numDisplacements"] = numDisplacements;
        d["percentYieldStrength"] = percentYieldStrength;
        d["displaceProbability"] = displaceProbability;
        d["meshSize"] = meshSize;
        d["order"] = order;
        d["seed"] = seed;
        return d;
    }

    /**
    * @brief Creates GenerationParams from Python dictionary
    * @param d Dictionary containing parameter values
    * @return GenerationParams instance
    */
    static GenerationParams from_dict(const pybind11::dict& d) {
        return {
            d["datasetSize"].cast<int>(),
            d["numBranches"].cast<int>(),
            d["lengthInterval"].cast<std::pair<double, double>>(),
            d["widthInterval"].cast<std::pair<double, double>>(),
            d["flipParams"].cast<std::pair<double, int>>(),
            d["dataRotationParams"].cast<std::pair<double, double>>(),
            d["modulusOfElasticity"].cast<double>(),
            d["poissonRatio"].cast<double>(),
            d["yieldStrength"].cast<double>(),
            d["numPerturbations"].cast<int>(),
            d["perturbProbability"].cast<double>(),
            d["widthPerturb"].cast<std::pair<double, double>>(),
            d["vectorPerturb"].cast<std::pair<double, double>>(),
            d["terminalPerturb"].cast<std::pair<double, double>>(),
            d["numDisplacements"].cast<int>(),
            d["percentYieldStrength"].cast<std::pair<double, double>>(),
            d["displaceProbability"].cast<double>(),
            d["meshSize"].cast<double>(),
            d["order"].cast<int>(),
            d["seed"].cast<unsigned>()
        };
    }
};

// The following struct helps to create perturbations in a base parametrization so we have non-linear parametrizations
/**
* @brief Parameters for generating perturbations for the parametrizations
* @details Contains settings for creating non-linear variations of base parametrizations
*          through controlled perturbations of geometry
*/
struct PerturbationParams {
    int numPerturbations;  ///< Number of perturbations to generate
    double perturbProbability; ///< Probability of applying perturbation [0,1]
    std::pair<double, double> width_perturb; ///< (% current width, % length)
    std::pair<double, double> vector_perturb; ///< (max angle change in degrees, reference branch length)
    std::pair<double, double> terminal_perturb; ///< (% branch length, % width)
    std::mt19937 &rng; ///< Reference to random number generator from @link GenerationParams @endlink

    /**
    * @brief Constructs from GenerationParams
    * @param params Source generation parameters
    */
    explicit PerturbationParams(GenerationParams &params)
        : numPerturbations(params.numPerturbations),
    perturbProbability(params.perturbProbability),
    width_perturb(params.widthPerturb),
    vector_perturb(params.vectorPerturb),
    terminal_perturb(params.terminalPerturb),
    rng(params.rng) {}

    /**
    * @brief Constructs perturbation parameters with various checks
    * @param num Number of perturbations (> 0)
    * @param perturb Perturbation probability [0,1]
    * @param width Width perturbation parameters
    * @param vector Vector perturbation parameters (angle < 45°)
    * @param terminal Terminal perturbation parameters
    * @param rng Reference to random number generator
    */
    PerturbationParams(int num, double perturb, std::pair<double, double> width, std::pair<double, double> vector,
        std::pair<double, double> terminal, std::mt19937 &rng)
        : numPerturbations(num),
        perturbProbability(perturb),
        width_perturb(std::move(width)),
        vector_perturb(std::move(vector)),
        terminal_perturb(std::move(terminal)),
        rng(rng)
    {
        LF_ASSERT_MSG(width_perturb.first >= 0 and width_perturb.first <= 1 and width_perturb.second >= 0 and
            width_perturb.second <= 1, "Parameters sent to width_perturb in Perturbation Generators invalid");
        LF_ASSERT_MSG(vector_perturb.first >= 0 and width_perturb.second < 45 and width_perturb.second >= 0,
            "Parameters sent to vector_perturb in Perturbation Generators invalid");
        LF_ASSERT_MSG(num > 0, "Invalid number of perturbations sent to Perturbation Generators");
    }
};

// The following struct includes parameters used to provide displacements to the parametrizations, and is a
// subset of GenerationParams
/**
* @brief Parameters for generating displacement boundary conditions
*/
struct DisplacementParams {
    int numDisplacements;   ///< Number of displacement BCs to generate
    double modulusOfElasticity; ///< Young's modulus (E)
    double poissonRatio;    ///< Poisson's ratio (ν)
    double yieldStrength;   ///< Material yield strength
    std::pair<double, double> percentYieldStrength; ///< Range for displacement scaling
    double meshSize; ///< Size parameter for mesh generation
    int order;  ///< Order of finite element calculation
    double displacementProbability; ///< Probability of applying displacement [0, 1]
    std::mt19937 &rng; ///< Reference to random number generator from @link GenerationParams @endlink

    /**
    * @brief Constructs from GenerationParams
    * @param params Source generation parameters
    */
    explicit DisplacementParams(GenerationParams &params)
        : numDisplacements(params.numDisplacements),
        modulusOfElasticity(params.modulusOfElasticity),
        poissonRatio(params.poissonRatio),
        yieldStrength(params.yieldStrength),
        percentYieldStrength(params.percentYieldStrength),
        meshSize(params.meshSize),
        order(params.order),
        displacementProbability(params.displaceProbability),
        rng(params.rng) {}

    /**
    * @brief Constructs displacement parameters (testing constructor)
    * @param num Number of displacements
    * @param E Young's modulus
    * @param nu Poisson's ratio
    * @param yS Yield strength
    * @param per Yield strength percentage range [0, 1]
    * @param mS Mesh size
    * @param o Element order
    * @param prob Displacement probability
    * @param r Reference to random number generator
    */
    DisplacementParams(int num, double E, double nu, double yS, std::pair<double, double> per, double mS, int o,
        double prob, std::mt19937 &r)
        : numDisplacements(num),
        modulusOfElasticity(E),
        poissonRatio(nu),
        yieldStrength(yS),
        percentYieldStrength(per),
        meshSize(mS),
        order(o),
        displacementProbability(prob),
        rng(r) {}
};

// The following struct contains random number generators used when building the displacement boundary conditions
// for the multi-branch case, index 0 corresponds to branch 0, and so on
/**
* @brief Random number generators for multi-branch displacement BCs
* @details Contains probability distributions and parameters for generating
*          displacement boundary conditions in multi-branch configurations
*/
struct DisplacementGenerators {
    double displacementProbability; ///< Probability of applying displacement
    std::vector<double> maxDistances; ///< Maximum displacement per branch
    std::mt19937 &rng;              ///< Reference to random number generator
    std::uniform_real_distribution<> uniformPerturb; ///< Uniform distribution [0,1]
    std::normal_distribution<> displacementPerturb;  ///< Normal distribution for perturbations

    /**
    * @brief Constructs displacement generators
    * @param prob Displacement probability
    * @param maxD Maximum distances per branch
    * @param rng Reference to random number generator
    */
    DisplacementGenerators(double prob, std::vector<double> maxD, std::mt19937 &rng)
                           : displacementProbability(prob), maxDistances(std::move(maxD)), rng(rng),
                           uniformPerturb(0.0, 1.0), displacementPerturb(0.0, 1.0) {}
};

// This namespace will hold the methods related to generating, saving, and loading the data sets that will be
// used to train the neural networks
/**
 * @brief Namespace containing all data generation, storage, and loading operations
 * @details Provides functionality for generating, transforming, saving, and loading
 *          parametrization datasets used in neural network training, and can then
 *          convert parametrization datasets to point datasets
 */
namespace DataOperations {

    /**
         * @brief Converts an unsigned integer to a boolean vector, with 1st index being the ones place
         * @param number Number to convert
         * @param bits Number of bits in the output vector
         * @return Vector of booleans representing the binary form of number
    */
    std::vector<bool> numberToBoolVector(unsigned int number, int bits);

    /**
         * @brief Samples n unique integers from [min, max] without replacement
         * @param min Minimum value (inclusive)
         * @param max Maximum value (inclusive)
         * @param n Number of samples to draw
         * @param rng Random number generator
         * @return Vector of n unique integers
         * @throws LFException if n > (max - min + 1) or max > min
    */
    std::vector<int> sampleWithoutReplacement(int min, int max, int n, std::mt19937 &rng);

    /**
     * @brief Generates flipped variations of a parametrization entry
     * @param param_entry Base parametrization entry
     * @param flip_params Pair of (flip probability, linear scale factor)
     * @param rng Random number generator from @link GenerationParams @endlink
     * @return Dataset containing flipped variations of param_entry, inclusive of param_entry
     * @see ParametrizationDataSet
     */
    ParametrizationDataSet flipVectorEntry(ParametrizationEntry &param_entry,
        std::pair<double, int> &flip_params, std::mt19937 &rng);

    /**
     * @brief Generates angles for rotating parametrizations
     * @param rotation_granularity Base rotation angle increment
     * @param random_proportion Random perturbation factor for angles
     * @param rng Random number generator
     * @return Vector of rotation angles in degrees
     */
    std::vector<double> generateRotationAngles(double rotation_granularity, double random_proportion, std::mt19937 &rng);

    /**
     * @brief Rotates a 2D point around a reference point     *
     * @param point Point to rotate
     * @param reference Center of rotation
     * @param angle_degrees Rotation angle in degrees
     * @return Rotated point
     * @see rotateParametrization variations for use of rotatePoint
    */
    Eigen::Vector2d rotatePoint(const Eigen::Vector2d& point, const Eigen::Vector2d& reference, double angle_degrees);

    /**
     * @brief Finds the geometric centroid of a mesh parametrization
     * @details For a single-branch parametrization, return the middle terminal, and for a
     * multi-branch parametrization, return the geometric centroid
     *
     * @param params Mesh parametrization data
     * @return Center point as 2D vector
     */
    Eigen::Vector2d findCenter(const MeshParametrizationData &params);

    /**
     * @brief Generates rotated variations of a @link MeshParametrizationData @endlink object
     * @param params Original mesh parametrization
     * @param rotation_params Pair of (rotation granularity, random proportion)
     * @param rng Random number generator
     * @return Vector of rotated mesh parametrizations
     * @see rotatePoint for rotation of individual points
     */
    std::vector<MeshParametrizationData> rotateParametrization(const MeshParametrizationData &params,
        const std::pair<double, double> &rotation_params, std::mt19937 &rng);

    /**
     * @brief Generates rotated variations of a @link ParametrizationEntry @endlink object
     * @param param_entry Original parametrization entry
     * @param rotation_params Pair of (rotation granularity, random proportion)
     * @param rng Random number generator
     * @return Dataset containing rotated variations
     */
    ParametrizationDataSet rotateParametrization(ParametrizationEntry &param_entry,
        std::pair<double, double> &rotation_params, std::mt19937 &rng);


    /**
     * @brief Generates midpoints and vectors for multi-branch construction
     * @details Use the vector of side lengths provided, an iterative algorithm is used to find the vertices of a
     * shape with the given side lengths
     * @param side_lengths Vector of branch lengths
     * @return Tuple of matrices containing (midpoints, outward-pointing vectors, sideways-pointing vectors, vertices)
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> generateMidPointsAndVectors(
        const Eigen::VectorXd &side_lengths);

    /**
     * @brief Generates displacement boundary conditions for a single branch parametrization
     * @param param_gens Displacement generation parameters
     * @return Matrix of displacement vectors
     * @see DisplacementGenerators for parameters used
     */
    Eigen::MatrixXd singleBranchDisplacements(DisplacementGenerators &param_gens);

    /**
     * @brief Generates displacement boundary conditions for a multi-branch parametrization
     * @param num Number of branches
     * @param param_gens Displacement generation parameters
     * @return Matrix of displacement vectors
     * @see DisplacementGenerators for parameters used
     */
    Eigen::MatrixXd multiBranchDisplacements(int num, DisplacementGenerators &param_gens);

    /**
     * @brief Generates perturbed variations of a base parametrization
     * @param base_param Base parametrization
     * @param perturb_params Perturbation parameters
     * @return Vector of perturbed parametrizations
     * @see PerturbationParams for parameters used
     */
    std::vector<MeshParametrizationData> generatePeturbedParametrizations(MeshParametrizationData &base_param,
        PerturbationParams &perturb_params);

    /**
     * @brief Generates displacement boundary conditions for a parametrization
     * @details Takes a parametrization and generates a dataset where all entries
     *          are in the linear elastic region of the material
     *
     * @param base_param Base parametrization to generate displacements for
     * @param params Displacement generation parameters
     * @return Dataset containing entries with different displacement BCs
     * @see DisplacementParams For parameter details
     * @see ParametrizationDataSet For return type structure
     */
    ParametrizationDataSet generateDisplacementBCs(MeshParametrizationData &base_param, DisplacementParams &params);

    /**
     * @brief Generates a single branch parametrization
     * @param num number of branches
     * @param width Width of the branch
     * @param length Length of the branch
     * @return Mesh parametrization for single branch
     */
    MeshParametrizationData generateSingleBranchParametrization(int num, double width, double length);

    /**
     * @brief Generates a multi-branch parametrization
     * @param num Number of branches
     * @param widths Vector of branch widths
     * @param lengths Vector of branch lengths
     * @return Mesh parametrization for multiple branches
     */
    MeshParametrizationData generateMultiBranchParametrization(int num,
        const Eigen::VectorXd &widths, const Eigen::VectorXd &lengths);

    /**
     * @brief Generates a complete parametrization dataset, according to Algorithm 1 of the report
     * @details First, the algorithm determines how many branches this dataset for its parametrizations and
     * initializes the relevant object
     * Second, it generates a base parametrization and multiple perturbations to induce some curvature
     * Third, it generates displacement BCs for each of these perturbations
     * Fourth, it rotates these entries according to the given rotation parameters, then adds these to the dataset
     *
     * @param gen_params Generation parameters
     * @param verbose Enable verbose output
     * @return Dataset of parametrizations
     */
    ParametrizationDataSet generateParametrizationDataSet(GenerationParams &gen_params, bool verbose = false);

    /**
     * @brief Converts a parametrization dataset containing @link ParametrizationEntry @endlink objects to a
     * dataset containing @link PointEntry @endlink
     * @param param_dataset Original parametrization dataset
     * @return Dataset in point representation
     */
    PointDataSet parametrizationToPoint(ParametrizationDataSet &param_dataset);

    /**
     * @brief Converts vector of @link MeshParametrizationData @endlink objects to a vector of
     * @link ParametrizationPoints @endlink objects
     * @param param_vector Vector of mesh parametrizations
     * @return Vector of point representations
     */
    std::vector<ParametrizationPoints> parametrizationToPoint(const std::vector<MeshParametrizationData>& param_vector);

    /**
     * @brief Saves a @link MeshParametrizationData @endlink object to a binary file
     * @param file Output file stream
     * @param param Mesh parametrization to save
     */
    void saveMeshParametrizationData(std::ofstream &file, const MeshParametrizationData &param);

    /**
     * @brief Loads a @link MeshParametrizationData @endlink object from a binary file
     * @param file Input file stream
     * @return Loaded mesh parametrization
     */
    MeshParametrizationData loadMeshParametrizationData(std::ifstream &file);

    /**
     * @brief Saves a @link ParametrizationPoints @endlink object to a binary file
     * @param file Output file stream
     * @param points Parametrization points to save
     */
    void saveParametrizationPoints(std::ofstream &file, const ParametrizationPoints &points);

    /**
     * @brief Loads a @link ParametrizationPoints @endlink object from a binary file
     * @param file Input file stream
     * @return Loaded parametrization points
     */
    ParametrizationPoints loadParametrizationPoints(std::ifstream &file);

    /**
     * @brief Saves a dataset of either type to a binary file
     * @param data_set Dataset to save
     * @param file_name Output file path (automatically loads the data from the data folder of the repo,
     * no .bin extension needed)
     */
    void saveDataSet(const DataSet &data_set, const std::string &file_name);

    /**
     * @brief Loads a dataset from a binary file
     * @param file_name Input file path (automatically loads the data from the data folder of the repo,
     * no .bin extension needed)
     * @return Loaded dataset
     */
    DataSet loadDataSet(const std::string &file_name);

};

#endif //METALFOAMS_DATA_OPERATIONS_H

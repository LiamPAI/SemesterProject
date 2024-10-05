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
struct VectorHash {
    size_t operator()(const std::vector<int>& v) const {
        std::hash<int> hasher;
        size_t seed = 0;
        for (int i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2); // "Fowler-Noll-Vo" hash function
        }
        return seed;
    }
};

constexpr size_t NUM_TAGS = 4; // Amount of tags

// Meaning behind each of the tags, they indicate the index we use within each entry for a datset
enum TagIndex {
    BASE = 0,
    BASE_PERTURBATION = 1,
    DISPLACEMENT = 2,
    ROTATION = 3,
};

// Another struct that contains the exact same information as MeshParametrization butter, but only the 6 points of
// each branch, to see how well the NN performs on both
struct ParametrizationPoints {
    int numBranches;
    Eigen::MatrixXd points;
    ParametrizationPoints(int num, Eigen::MatrixXd p) : numBranches(num), points(std::move(p)) {
        LF_ASSERT_MSG(numBranches == points.rows() / 4,
                      "Number of branches is not equal to the correct number of rows when initializing "
                      "ParametrizationPoints, num is " << numBranches <<
                      " and points has " << points.rows() << " rows");
    }
};

// Instead of the parametrization, the actual polynomial points represented by the parametrization are used here, this
// is for the purpose of testing both methods to see which works better when training the NN
struct PointEntry {
    int numBranches;
    ParametrizationPoints points;
    Eigen::MatrixXd displacements;
    double energyDifference;
    std::vector<int> tags;

    PointEntry(int num, ParametrizationPoints p, Eigen::MatrixXd disp, double eD, std::vector<int> tags)
        : numBranches(num), points(std::move(p)), displacements(std::move(disp)), energyDifference(eD),
        tags(std::move(tags)) {}
};

// This data entry simply uses the mesh parametrizations as defined in mesh_parametrization.h
struct ParametrizationEntry {
    int numBranches;
    MeshParametrizationData param;
    Eigen::MatrixXd displacements;
    double energyDifference;
    std::vector<int> tags;

    ParametrizationEntry(int num, MeshParametrizationData p, Eigen::MatrixXd disp, double eD)
        : numBranches(num), param(std::move(p)), displacements(std::move(disp)), energyDifference(eD),
        tags(NUM_TAGS, -1) {}
};

// The datasets are simply vectors of the original entries and their types
using ParametrizationDataSet = std::vector<ParametrizationEntry>;
using PointDataSet = std::vector<PointEntry>;

// enum for storing the dataset in a binary file and identifying which is which
enum class DataSetType {
    Parametrization,
    Point
};

// A dataset can contain either type of entry
using DataSet = std::variant<ParametrizationDataSet, PointDataSet>;

// A struct that will contain all the necessary parameters to construct a complete dataset
struct GenerationParams {
    int datasetSize;
    int numBranches;
    std::pair<double, double> lengthInterval; // min to max length interval to sample from (depends on mesh)
    std::pair<double, double> widthInterval; // min to max width interval
    std::pair<double, int> flipParams; // flip probability, and linear scale factor (linear scale factor determines
                                        // how much smaller the equivalent point data set will be)
    std::pair<double, double> dataRotationParams; // Rotation granularity + random proportion
    double modulusOfElasticity;
    double poissonRatio;
    double yieldStrength;
    int numPerturbations;
    double perturbProbability;
    std::pair<double, double> width_perturb; // percent current width, percent length
    std::pair<double, double> vector_perturb; // max vector change (in degrees), reference length
    std::pair<double, double> terminal_perturb; // percent branch length, percent width
    int numDisplacements;
    double percentYieldStrength;
    double displaceProbability;
    double meshSize;
    int order;
    unsigned seed;
    std::mt19937 rng;

    GenerationParams(int size, int numB, std::pair<double, double> lenInterval, std::pair<double, double> widInterval,
                     std::pair<double, int> flipP, std::pair<double, double> dataRotP, double modulus, double poisson,
                     double yieldStren, int numPer, double perProb, std::pair<double, double> widPer,
                     std::pair<double, double> vecPer, std::pair<double, double> termPer,
                     int numDisp, double perStrength, double dispProb, double meshS, int o,
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
            width_perturb(std::move(widPer)),
            vector_perturb(std::move(vecPer)),
            terminal_perturb(std::move(termPer)),
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
        LF_ASSERT_MSG(width_perturb.first >= 0 and width_perturb.first <= 1 and width_perturb.second >= 0 and
            width_perturb.second <= 1, "Parameters sent to width_perturb in Generation params invalid");
        LF_ASSERT_MSG(vector_perturb.first >= 0 and width_perturb.second < 45 and width_perturb.second >= 0,
            "Parameters sent to vector_perturb in Generation params invalid");
        LF_ASSERT_MSG(terminal_perturb.first >= 0 and terminal_perturb.first <= 1 and terminal_perturb.second >= 0 and
            terminal_perturb.second <= 1, "Parameters sent to terminal_perturb in Generation params invalid");
        LF_ASSERT_MSG(numDisplacements > 0, "Invalid number of displacements sent to Generation params");
        LF_ASSERT_MSG(percentYieldStrength >= 0 and percentYieldStrength <= 1, "Percent yield strength is not on the "
            "interval [0, 1], it is instead " << percentYieldStrength);
        LF_ASSERT_MSG(displaceProbability >= 0 and displaceProbability <= 1, "Perturb probability is not on the "
            "interval [0, 1], it is instead " << displaceProbability);
        LF_ASSERT_MSG(meshSize > 1e-6, "Inavlid mesh size sent to Generation Params, " << meshSize);
        LF_ASSERT_MSG(order == 1 or order == 2, "Invalid order sent to Generation Params, " << order);
    }
};

// The following struct helps to create perturbations in a base parametrization so we have non-linear parametrizations
struct PerturbationParams {
    int numPerturbations;
    double perturbProbability;
    std::pair<double, double> width_perturb; // percent current width, percent length
    std::pair<double, double> vector_perturb; // max vector change (in degrees), reference length
    std::pair<double, double> terminal_perturb; // percent branch length, percent width
    std::mt19937 &rng;

    explicit PerturbationParams(GenerationParams &params)
        : numPerturbations(params.numPerturbations),
    perturbProbability(params.perturbProbability),
    width_perturb(params.width_perturb),
    vector_perturb(params.vector_perturb),
    terminal_perturb(params.terminal_perturb),
    rng(params.rng) {}

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
struct DisplacementParams {
    int numDisplacements;
    double modulusOfElasticity;
    double poissonRatio;
    double yieldStrength;
    double percentYieldStrength; // Heuristic to see how much we move our displacement vectors
    double meshSize;
    int order;
    double displacementProbability;
    std::mt19937 &rng;

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

    // This constructor is for testing
    DisplacementParams(int num, double E, double nu, double yS, double per, double mS, int o,
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
struct DisplacementGenerators {
    double displacementProbability;
    std::vector<double> maxDistances;
    std::mt19937 &rng;
    std::uniform_real_distribution<> uniformPerturb;
    std::normal_distribution<> displacementPerturb;

    DisplacementGenerators(double prob, std::vector<double> maxD, std::mt19937 &rng)
                           : displacementProbability(prob), maxDistances(std::move(maxD)), rng(rng),
                           uniformPerturb(0.0, 1.0), displacementPerturb(0.0, 1.0) {}
};

// This namespace will hold the methods related to generating, saving, and loading the data sets that will be
// used to train the neural networks
namespace DataOperations {

    std::vector<bool> numberToBoolVector(unsigned int number, int bits);
    std::vector<int> sampleWithoutReplacement(int min, int max, int n, std::mt19937 &rng);

    ParametrizationDataSet flipVectorEntry(ParametrizationEntry &param_entry,
        std::pair<double, int> &flip_params, std::mt19937 &rng);
    std::vector<double> generateRotationAngles(double rotation_granularity, double random_proportion, std::mt19937 &rng);
    Eigen::Vector2d rotatePoint(const Eigen::Vector2d& point, const Eigen::Vector2d& reference, double angle_degrees);
    Eigen::Vector2d findCenter(const MeshParametrizationData &params);
    std::vector<MeshParametrizationData> rotateParametrization(const MeshParametrizationData &params,
        const std::pair<double, double> &rotation_params, std::mt19937 &rng);
    ParametrizationDataSet rotateParametrizationEntry(ParametrizationEntry &param_entry,
        std::pair<double, double> &rotation_params, std::mt19937 &rng);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>generateMidPointsAndVectors(
        const Eigen::VectorXd &side_lengths);

    Eigen::MatrixXd singleBranchDisplacements(DisplacementGenerators &param_gens);
    Eigen::MatrixXd multiBranchDisplacements(int num, DisplacementGenerators &param_gens);
    std::vector<MeshParametrizationData> generatePeturbedParametrizations(MeshParametrizationData &base_param,
        PerturbationParams &perturb_params);
    ParametrizationDataSet generateDisplacementBCs(MeshParametrizationData &base_param, DisplacementParams &params);
    MeshParametrizationData generateSingleBranchParametrization(int num, double width, double length);
    MeshParametrizationData generateMultiBranchParametrization(int num,
        const Eigen::VectorXd &widths, const Eigen::VectorXd &lengths);

    ParametrizationDataSet generateParametrizationDataSet(GenerationParams &gen_params);
    PointDataSet parametrizationToPoint(ParametrizationDataSet &param_dataset);

    void saveMeshParametrizationData(std::ofstream &file, const MeshParametrizationData &param);
    MeshParametrizationData loadMeshParametrizationData(std::ifstream &file);
    void saveParametrizationPoints(std::ofstream &file, const ParametrizationPoints &points);
    ParametrizationPoints loadParametrizationPoints(std::ifstream &file);

    void saveDataSet(const DataSet &data_set, const std::string &file_name);
    DataSet loadDataSet(const std::string &file_name);

};

#endif //METALFOAMS_DATA_OPERATIONS_H

//
// Created by Liam Curtis on 2024-09-12.
//

#ifndef METALFOAMS_DATA_OPERATIONS_H
#define METALFOAMS_DATA_OPERATIONS_H

#include "mesh_parametrization.h"
#include <Eigen/Dense>
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
// customizable so I can observe what aspects the NN handles well or not

// This is another struct I intend to use to represent the same information as a regular MeshParametrization, but it
// simply contains the 6 points required to define this "part" of the mesh, or many more points if it is a node type
// with more branches
// TODO: Consider putting asserts here to make sure that the entries and data we generate are valid
// TODO: Decide on how to handle the ordering of the points so that for node parts it is easily identifiable
//  which points coincide, either include a method here or implement it in polyPoints
constexpr size_t NUM_TAGS = 4; // Amount of tags

// Meaning behind each of the tags, they indicate the index we use within each entry
enum TagIndex {
    BASE = 0,
    BASE_ROTATION = 1,
    PERTURBATION = 2,
    FLIP_VECTOR = 3,
};

struct ParametrizationPoints {
    int numBranches;
    Eigen::MatrixXd points;

    ParametrizationPoints(int num, Eigen::MatrixXd p) : numBranches(num), points(std::move(p)) {
        LF_ASSERT_MSG(num == p.rows() / 4,
                      "Number of branches is not equal to the correct number of rows when initializing "
                      "ParametrizationPoints");
    }
};

// Instead of the parametrization, the actual polynomial points represented by the parametrization are used here, this
// is for the purpose of testing both methods to see which works better when training the NN
struct PointEntry {
    int numBranches;
    ParametrizationPoints points1;
    ParametrizationPoints points2;
    double energyDifference;
    std::vector<int> tags;

    PointEntry(int num, ParametrizationPoints p1, ParametrizationPoints p2, double eD)
        : numBranches(num), points1(std::move(p1)), points2(std::move(p2)), energyDifference(eD), tags(NUM_TAGS, -1) {}


    PointEntry(int num, ParametrizationPoints p1, ParametrizationPoints p2, double eD, std::vector<int> tags)
        : numBranches(num), points1(std::move(p1)), points2(std::move(p2)), energyDifference(eD),
        tags(std::move(tags)) {}

};

// This data entry simply uses the mesh parametrizations as defined in mesh_parametrization.h
struct ParametrizationEntry {
    int numBranches; // TODO: add this allowed num of branches to an assert
    MeshParametrizationData param1;
    MeshParametrizationData param2;
    double energyDifference;
    std::vector<int> tags;

    ParametrizationEntry(int num, MeshParametrizationData p1, MeshParametrizationData p2, double eD)
        : numBranches(num), param1(std::move(p1)), param2(std::move(p2)), energyDifference(eD), tags(NUM_TAGS, -1) {}

};

// The datasets are simply vectors of the original entries, it should be very easy to convert between the two
using ParametrizationDataSet = std::vector<ParametrizationEntry>;
using PointDataSet = std::vector<PointEntry>;

// enum for storing the dataset in a binary file
enum class DataSetType {
    Parametrization,
    Point
};

// A dataset can contain either type of entry
using DataSet = std::variant<ParametrizationDataSet, PointDataSet>;

// TODO: Consider making some of these parameters within a certain range (e.g. flip probability between 0 and 1) in constructor
struct GenerationParams {
    int datasetSize;
    int numBranches;
    std::pair<double, double> lengthInterval; // min to max length interval to sample from (depends on mesh)
    std::pair<double, double> widthInterval; // min to max width interval
    std::pair<double, int> flipParams; // flip probability, and linear scale factor
    std::pair<double, double> baseRotationParams; // Rotation granularity + random proportion
    std::pair<double, double> dataRotationParams; // Rotation granularity + random proportion
    double modulusOfElasticity; // TODO: Ensure these are the correct parameters I would like for energy calculation
    double poissonRatio;
    double yieldStrength;
    int numPerturbations;
    double percentYieldStrain;
    double percentTerminal;
    double maxRotationAngle; // In radians
    double rotationFactor;
    double perturbProbability;
    unsigned seed;
    std::mt19937 rng;

    GenerationParams(int size, int numB, std::pair<double, double> lenInterval, std::pair<double, double> widthInterval,
                     std::pair<double, int> flipP, std::pair<double, double> baseRotP,
                     std::pair<double, double> dataRotP, double modulus, double poisson, double yield, int numPer,
                     double perYield, double perTerm, double maxAng, double rotFac, double perProb,
                     unsigned s = std::chrono::system_clock::now().time_since_epoch().count())
            : datasetSize(size),
            numBranches(numB),
            lengthInterval(std::move(lenInterval)),
            widthInterval(std::move(widthInterval)),
            flipParams(std::move(flipP)),
            baseRotationParams(std::move(baseRotP)),
            dataRotationParams(std::move(dataRotP)),
            modulusOfElasticity(modulus),
            poissonRatio(poisson),
            yieldStrength(yield),
            numPerturbations(numPer),
            percentYieldStrain(perYield),
            percentTerminal(perTerm),
            maxRotationAngle(maxAng),
            rotationFactor(rotFac),
            perturbProbability(perProb),
            seed(s),
            rng(seed) {}
};

// TODO: Update the structure of this and the GenerationParams if I decide the change the heuristics to use for perturbations
// TODO: Decide if these should be part of GenerationParams, which I think they should be
// The following struct are parameters used to perturb the parametrizations, and is a subset of GenerationParams
struct PerturbationParams {
    int numPerturbations;
    double modulusOfElasticity;
    double yieldStrength;
    double percentYieldStrain; // Used in a heuristic to determine amount by which we can perturb the width
    double percentTerminal; // Used in heuristic to determine the amount by which we can perturb the terminals
    double maxRotationAngle; // In radians
    double rotationFactor;
    double perturbProbability;
    std::mt19937 &rng;

    explicit PerturbationParams(GenerationParams &params)
        : numPerturbations(params.numPerturbations),
        modulusOfElasticity(params.modulusOfElasticity),
        yieldStrength(params.yieldStrength),
        percentYieldStrain(params.percentYieldStrain),
        percentTerminal(params.percentTerminal),
        maxRotationAngle(params.maxRotationAngle),
        rotationFactor(params.rotationFactor),
        perturbProbability(params.perturbProbability),
        rng(params.rng) {}
};

// The following struct contains random number generators used when building the perturbations;
// for the multi-branch case, index 0 corresponds to branch 0, and so on
struct PerturbationGenerators {
    double perturbProbability;
    std::uniform_real_distribution<> uniformPerturb;
    std::vector<std::normal_distribution<>> widthDistributions;
    std::vector<std::normal_distribution<>> terminalDistributions;
    std::vector<std::normal_distribution<>> vectorDistributions;
    std::mt19937 &rng;

    PerturbationGenerators(double prob,
                           std::vector<std::normal_distribution<>> w_dists,
                           std::vector<std::normal_distribution<>> t_dists,
                           std::vector<std::normal_distribution<>> v_dists,
                           std::mt19937 &rng)
                           : perturbProbability(prob),
                           uniformPerturb(0.0, 1.0),
                           widthDistributions(std::move(w_dists)),
                           terminalDistributions(std::move(t_dists)),
                           vectorDistributions(std::move(v_dists)),
                           rng(rng) {}
};

// This class will hold the methods to generate the datasets required to train the neural networks
// TODO: Consider making this just a namespace considering the class doesn't store any private variables
class DataOperations {
public:
    DataOperations() = default;
    // TODO: Consider adding a function that actually "flips" the parametrizations, which I believe will be different from a rotation
    std::vector<double> generateRotationAngles(double rotation_granularity, double random_proportion);
    Eigen::Vector2d rotatePoint(const Eigen::Vector2d& point, const Eigen::Vector2d& reference, double angle_radians);
    Eigen::Vector2d findCenter(MeshParametrizationData &params);
    std::vector<bool> numberToBoolVector(unsigned int number, int bits);
    std::vector<int> sampleWithoutReplacement(int min, int max, int n);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> generateMidPointsAndVectors(Eigen::VectorXd &side_lengths);
    // TODO: For rotateParametrization, consider implementing a version for the Point version
    std::vector<MeshParametrizationData> rotateParametrization(MeshParametrizationData &params, std::pair<double, double> &rotation_params);
    ParametrizationDataSet rotateParametrizationEntry(ParametrizationEntry &param, std::pair<double, double> &rotation_params);
    ParametrizationDataSet flipVectorEntry(ParametrizationEntry &param, std::pair<double, int> &flip_params, std::mt19937 &rng);

    MeshParametrizationData generateSingleBranchParametrization(int num, double width, double length);
    MeshParametrizationData generateMultiBranchParametrization(int num, Eigen::VectorXd &widths, Eigen::VectorXd &lengths);
    MeshParametrizationData perturbSingleBranchParametrization(const MeshParametrizationData &base_param, PerturbationGenerators &param_gens);
    MeshParametrizationData perturbMultiBranchParametrization(const MeshParametrizationData &base_param, PerturbationGenerators &param_gens);
    ParametrizationDataSet generatePerturbedParametrizations(MeshParametrizationData &base_param, PerturbationParams &params);
    ParametrizationDataSet generateParametrizationDataSet(GenerationParams &params);
    PointDataSet parametrizationToPoint(ParametrizationDataSet &paramSet);

    // TODO: Implement these two functions, and decide if this current signature is enough, static or no?
    static void saveDataSet(const DataSet &data_set, const std::string &file_name);
    static DataSet loadDataSet(const std::string &file_name);

    // Helper functions for saveDataSet and loadDataSet
    static void saveMeshParametrizationData(std::ofstream &file, const MeshParametrizationData &param);
    static MeshParametrizationData loadMeshParametrizationData(std::ifstream &file);
    static void saveParametrizationPoints(std::ofstream &file, const ParametrizationPoints &points);
    static ParametrizationPoints loadParametrizationPoints(std::ifstream &file);


    PointDataSet generatePointData(); // TODO: Note, might not be needed if I just convert from the parametrization type

private:

};



#endif //METALFOAMS_DATA_OPERATIONS_H

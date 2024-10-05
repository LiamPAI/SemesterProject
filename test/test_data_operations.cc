//
// Created by Liam Curtis on 30.09.2024.
//

#include "../include/data_operations.h"
#include "../include/mesh_parametrization.h"

void printPointDataSet(const PointDataSet& dataset) {
    std::cout << "PointDataSet Contents:" << std::endl;
    std::cout << "=======================" << std::endl;

    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& entry = dataset[i];

        std::cout << "Entry " << i + 1 << ":" << std::endl;
        std::cout << "  Number of Branches: " << entry.numBranches << std::endl;

        std::cout << "  Parametrization Points:" << std::endl;
        std::cout << "    Number of Branches: " << entry.points.numBranches << std::endl;
        std::cout << "    Points Matrix:" << std::endl;
        std::cout << entry.points.points << std::endl;

        std::cout << "  Displacements Matrix:" << std::endl;
        std::cout << entry.displacements << std::endl;

        std::cout << "  Energy Difference: " << std::fixed << std::setprecision(7) << entry.energyDifference << std::endl;

        std::cout << "  Tags: ";
        for (const auto& tag : entry.tags) {
            std::cout << tag << " ";
        }
        std::cout << std::endl;

        std::cout << std::endl;
    }
}

void printMeshParametrizationData(const MeshParametrizationData& data) {
    std::cout << std::fixed << std::setprecision(7);  // Set floating-point precision for output

    std::cout << "MeshParametrizationData:\n";
    std::cout << "  Number of Branches: " << data.numBranches << "\n\n";

    auto printMatrix = [](const std::string& name, const Eigen::MatrixXd& matrix) {
        std::cout << "  " << name << " (" << matrix.rows() << "x" << matrix.cols() << "):\n";
        std::cout << matrix << "\n\n";
    };

    printMatrix("Widths", data.widths);
    printMatrix("Terminals", data.terminals);
    printMatrix("Vectors", data.vectors);

    std::cout << std::string(50, '-') << "\n";  // Separator line
}

void printParametrizationDataSet(const ParametrizationDataSet& dataset) {
    std::cout << std::fixed << std::setprecision(7);

    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& entry = dataset[i];

        std::cout << "Entry " << i + 1 << ":\n";
        std::cout << "  Number of Branches: " << entry.numBranches << "\n";
        std::cout << "  Energy Difference: " << entry.energyDifference << "\n";

        std::cout << "  Tags: ";
        for (int tag : entry.tags) {
            std::cout << tag << " ";
        }
        std::cout << "\n";

        std::cout << "  Parametrization Data:\n";
        std::cout << "    Widths (" << entry.param.widths.rows() << "x" << entry.param.widths.cols() << "):\n"
                  << entry.param.widths << "\n\n";

        std::cout << "    Terminals (" << entry.param.terminals.rows() << "x" << entry.param.terminals.cols() << "):\n"
                  << entry.param.terminals << "\n\n";

        std::cout << "    Vectors (" << entry.param.vectors.rows() << "x" << entry.param.vectors.cols() << "):\n"
                  << entry.param.vectors << "\n\n";

        std::cout << "  Displacements (" << entry.displacements.rows() << "x" << entry.displacements.cols() << "):\n"
                  << entry.displacements << "\n";

        std::cout << std::string(50, '-') << "\n";
    }
}

void printDataSet(const DataSet& dataset) {
    std::visit([](const auto& data) {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, PointDataSet>) {
            printPointDataSet(data);
        } else if constexpr (std::is_same_v<T, ParametrizationDataSet>) {
            printParametrizationDataSet(data);
        } else {
            std::cerr << "Error: Unknown dataset type" << std::endl;
        }
    }, dataset);
}

ParametrizationEntry basicSingleBranchParametrizationEntry() {
    Eigen::MatrixXd vectors(2,3);
    Eigen::MatrixXd terminals(2, 3);
    Eigen::MatrixXd widths(1, 3);

    vectors << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    terminals << -1, 0, 1, 0, 0, 0;
    widths << 1, 1, 1;

    MeshParametrizationData single_branch {1, widths, terminals, vectors};

    Eigen::MatrixXd displacements(4, 2);
    displacements << 0, 0, 0, 0, 0, 0, 0, 0;

    ParametrizationEntry basic_entry {1, single_branch, displacements, 0};
    return basic_entry;
}

ParametrizationEntry basicMultiBranchParametrizationEntry() {
    Eigen::MatrixXd multi_vectors(8,3);
    Eigen::MatrixXd multi_terminals(8, 3);
    Eigen::MatrixXd multi_widths(4, 3);

    multi_widths << 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8, 1.0, 0.9, 0.8;
    multi_terminals << 1.0, 2.0, 3.0, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.0, -1.0, -2.0,
                        0.5, 0.5, 0.5, 1.0, 2.0, 3.0,
                        0.0, -1.0, -2.0, 0.5, 0.5, 0.5;
    multi_vectors << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0;

    MeshParametrizationData multi_branch {4, multi_widths, multi_terminals, multi_vectors};

    Eigen::MatrixXd displacements(16, 1);
    displacements.setZero();

    ParametrizationEntry basic_entry {4, multi_branch, displacements, 0};
    return basic_entry;
}

void test_numberToBoolVector() {
    unsigned int num = 3;

    const auto result = DataOperations::numberToBoolVector(num, 3);

    const std::vector correct = {true, true, false};
    assert(correct == result);
}

void test_sampleWithoutReplacement() {
    constexpr int min = 0;
    constexpr int max = 7;
    constexpr int n = 4;
    std::mt19937 rng(0);

    const auto result = DataOperations::sampleWithoutReplacement(min, max, n, rng);

    const std::vector correct = {2, 5, 6, 7};
    assert(correct == result);
}

void test_flipVectorEntry() {
    auto basic_single_entry = basicSingleBranchParametrizationEntry();
    std::pair<double, int> flip_params = {1, 8}; // Vary this and call printParametrizationDataSet to test
    std::mt19937 rng(0);

    auto basic_single_dataset = DataOperations::flipVectorEntry(basic_single_entry, flip_params, rng);

    printParametrizationDataSet(basic_single_dataset);

    auto basic_multi_entry = basicMultiBranchParametrizationEntry();

    auto basic_multi_dataset = DataOperations::flipVectorEntry(basic_multi_entry, flip_params, rng);

    printParametrizationDataSet(basic_multi_dataset);
}

void test_generateRotationAngles() {
    double rotation_granularity = 360;
    double random_proportion = 0.25;
    std::mt19937 rng(0);

    auto angles = DataOperations::generateRotationAngles(rotation_granularity, random_proportion, rng);

    std::cout << "Value of angles are: ";
    for (auto angle : angles) {
        std::cout << angle << ", ";
    }
    std::cout << "\n";
}

void test_rotatePoint() {

    Eigen::Vector2d zero = Eigen::Vector2d::Zero();
    Eigen::Vector2d test_point {1, 2};
    Eigen::Vector2d test_reference_point {1, 1};

    double rotation_granularity = 90;
    double random_proportion = 0.25;
    std::mt19937 rng(0);

    auto angles = DataOperations::generateRotationAngles(rotation_granularity, random_proportion, rng);


    for (auto angle : angles) {
        auto rotated_point = DataOperations::rotatePoint(test_point, test_reference_point, angle);
        std::cout << "Value of angle is: " << angle << " and previously the point was at " << test_point.transpose() <<
            " with reference " << test_reference_point.transpose() << " but is rotated to " << rotated_point.transpose() << "\n";
    }

    rotation_granularity = 360;
    random_proportion = 0;

    angles = DataOperations::generateRotationAngles(rotation_granularity, random_proportion, rng);

    for (auto angle : angles) {
        auto rotated_point = DataOperations::rotatePoint(zero, zero, angle);
        std::cout << "Value of angle is: " << angle << " and previously the point was at " << zero.transpose() <<
            " with reference " << zero.transpose() << " but is rotated to " << rotated_point.transpose() << "\n";
    }
}

void test_findCenter() {
    auto basic_single_entry = basicSingleBranchParametrizationEntry();
    auto basic_multi_entry = basicMultiBranchParametrizationEntry();

    Eigen::Vector2d single_center = DataOperations::findCenter(basic_single_entry.param);
    Eigen::Vector2d multi_center = DataOperations::findCenter(basic_multi_entry.param);

    std::cout << "Value of single_center: " << single_center.transpose() << "\n";
    std::cout << "Value of multi_center: " << multi_center.transpose() << "\n";

    Eigen::Vector2d correct_single_center;
    Eigen::Vector2d correct_multi_center;

    correct_single_center << 0, 0;
    correct_multi_center << 0.5, 0.5;

    assert(correct_single_center.isApprox(single_center));
    assert(correct_multi_center.isApprox(correct_multi_center));
}

void test_rotateParametrization() {
    auto basic_single_entry = basicSingleBranchParametrizationEntry();
    auto basic_multi_entry = basicMultiBranchParametrizationEntry();

    double rotation_granularity = 360;
    double random_proportion = 0;
    std::pair rand_params {rotation_granularity, random_proportion};
    std::mt19937 rng(0);

    auto rotated_single_params = DataOperations::rotateParametrization(
        basic_single_entry.param, rand_params, rng);
    for (auto &param : rotated_single_params) {
        printMeshParametrizationData(param);
    }

    auto rotated_multi_params = DataOperations::rotateParametrization(
        basic_multi_entry.param, rand_params, rng);
    for (auto &param : rotated_multi_params) {
        printMeshParametrizationData(param);
    }
}

void test_rotateParametrizationEntry() {
    auto basic_single_entry = basicSingleBranchParametrizationEntry();
    auto basic_multi_entry = basicMultiBranchParametrizationEntry();

    double rotation_granularity = 90;
    double random_proportion = 0.25;
    std::pair rand_params {rotation_granularity, random_proportion};
    std::mt19937 rng(0);

    Eigen::MatrixXd displacements_single(4, 2);
    displacements_single << 1, -1, 1, -1, 1, 0, -1, 0;
    basic_single_entry.displacements = displacements_single;

    Eigen::MatrixXd displacements_multi(16, 1);
    displacements_multi << 1, -1, 1, -1, 1, 0, -1, 0, 1, -1, 1, -1, 1, 0, -1, 0;
    basic_multi_entry.displacements = displacements_multi;

    auto single_dataset = DataOperations::rotateParametrizationEntry(basic_single_entry, rand_params, rng);
    auto multi_dataset = DataOperations::rotateParametrizationEntry(basic_multi_entry, rand_params, rng);

    printParametrizationDataSet(single_dataset);
    printParametrizationDataSet(multi_dataset);
}

void test_generateMidPointsAndVectors() {
    Eigen::VectorXd triangle_sides(3);
    triangle_sides << 3, 4, 5;
    Eigen::VectorXd triangle_branch_lengths(3);
    triangle_branch_lengths << 1.0, 1.0, 1.0;

    auto triangle_multi_param = DataOperations::generateMultiBranchParametrization(3,
        triangle_sides, triangle_branch_lengths);

    bool check_triangle = MeshParametrization::meshParamValidator(triangle_multi_param);

    Eigen::VectorXd rhombus_sides(4);
    rhombus_sides << 2, 4, 3, 2;
    Eigen::VectorXd rhombus_branch_lengths(4);
    rhombus_branch_lengths << 1.0, 1.0, 1.0, 1.0;

    auto rhombus_multi_param = DataOperations::generateMultiBranchParametrization(4,
        rhombus_sides, rhombus_branch_lengths);

    bool check_rhombus = MeshParametrization::meshParamValidator(rhombus_multi_param);

    Eigen::VectorXd pentagon_side_lengths(5);
    pentagon_side_lengths << 2, 2, 3, 4, 5;
    Eigen:: VectorXd pentagon_branch_lengths(5);
    pentagon_branch_lengths << 3, 3, 3, 3, 3;

    auto pentagon_multi_param = DataOperations::generateMultiBranchParametrization(
        5, pentagon_side_lengths, pentagon_branch_lengths);

    bool check_pentagon = MeshParametrization::meshParamValidator(pentagon_multi_param);

    assert(check_triangle and check_rhombus and check_pentagon);
}

void test_singleBranchDisplacements() {
    std::mt19937 rng(0);
    DisplacementGenerators param_gens {0.75, {0.1}, rng};

    auto displacements = DataOperations::singleBranchDisplacements(param_gens);

    std::cout << "Value of displacements: \n" << displacements << "\n";
}

void test_multiBranchDisplacements() {
    std::mt19937 rng(0);
    DisplacementGenerators param_gens {1, {0.1, 1, 10, 100}, rng};

    auto displacements = DataOperations::multiBranchDisplacements(4, param_gens);

    std::cout << "Value of displacements: \n" << displacements << "\n";
}

void test_generatePeturbedParametrizations() {
    int num = 10;
    double prob = 1;
    std::pair width (0.1, 0.05);
    std::pair vector(22.5, 1.0);
    std::pair terminal(0.1, 0.05);
    std::mt19937 rng(0);
    PerturbationParams perturb_params {num, prob, width, vector, terminal, rng};

    auto basic_sigle_param = basicSingleBranchParametrizationEntry().param;

    auto basic_single_perturbed = DataOperations::generatePeturbedParametrizations(
        basic_sigle_param, perturb_params);

    for (auto single_perturbed : basic_single_perturbed) {
        printMeshParametrizationData(single_perturbed);
    }

    auto basic_multi_param = basicMultiBranchParametrizationEntry().param;

    auto basic_multi_perturbed = DataOperations::generatePeturbedParametrizations(basic_multi_param, perturb_params);

    for (auto multi_perturbed : basic_multi_perturbed) {
        printMeshParametrizationData(multi_perturbed);
    }
}

void test_generateDisplacementBCs() {
    auto basic_single_param = basicSingleBranchParametrizationEntry().param;

    printMeshParametrizationData(basic_single_param);

    int num_perturbations = 5;
    double youngs_modulus = 1000000000;
    double poisson = 0.33;
    double yield_stress = 5000000;
    double percent = 0.25; // This and perturb probability are the main parameters to modify
    double mesh_size = 0.1; // This as well
    int order = 1;
    double perturb_probability = 1;
    std::mt19937 rng(0);

    DisplacementParams perturb_params {num_perturbations, youngs_modulus, poisson, yield_stress, percent,
        mesh_size, order, perturb_probability, rng};

    auto perturbed_single_data_set = DataOperations::generateDisplacementBCs(basic_single_param, perturb_params);

    auto basic_multi_param = basicMultiBranchParametrizationEntry().param;

    auto perturbed_multi_data_set = DataOperations::generateDisplacementBCs(basic_multi_param, perturb_params);

    printParametrizationDataSet(perturbed_single_data_set);
    printParametrizationDataSet(perturbed_multi_data_set);
}

void test_generateSingleBranchParametrization() {
    int num = 1;
    double width = 0.5;
    double length = 7;

    auto test_single_branch = DataOperations::generateSingleBranchParametrization(num, width, length);
    auto check = MeshParametrization::meshParamValidator(test_single_branch);
    assert(check);
}

void test_generateMultiBranchParametrization() {
    Eigen::VectorXd widths_tri(3);
    Eigen::VectorXd lengths_tri(3);
    widths_tri << 3, 4, 5;
    lengths_tri << 1, 2, 3;
    auto test_triangle = DataOperations::generateMultiBranchParametrization(3, widths_tri, lengths_tri);
    auto check_tri = MeshParametrization::meshParamValidator(test_triangle);
    assert(check_tri);

    Eigen::VectorXd widths_quad(4);
    Eigen::VectorXd lengths_quad(4);
    widths_quad << 1, 2, 3, 4;
    lengths_quad << 1, 2, 3, 4;
    auto test_quad = DataOperations::generateMultiBranchParametrization(4, widths_quad, lengths_quad);
    auto check_quad = MeshParametrization::meshParamValidator(test_quad);
    assert(check_quad);

    Eigen::VectorXd widths_pent(5);
    Eigen::VectorXd lengths_pent(5);
    widths_pent << 1, 2, 2, 4, 3;
    lengths_pent << 2, 2, 2, 2, 2;
    auto test_pent = DataOperations::generateMultiBranchParametrization(5, widths_pent, lengths_pent);
    auto check_pent = MeshParametrization::meshParamValidator(test_pent);
    assert(check_pent);
}

void test_generateParametrizationDataSet() {
    int data_size = 20;
    int num_branches = 1;
    std::pair<double, double> len_interval(3, 5);
    std::pair<double, double> width_interval(0.5, 1);
    std::pair<double, int> flip_params(1, 0);
    std::pair<double, double> data_rotation(360, 1);
    double youngs_modulus = 1000000000;
    double poisson = 0.33;
    double yield_stress = 5000000;
    int num_perturb = 2;
    double per_prob = 1;
    std::pair width (0.1, 0.05);
    std::pair vector(22.5, 1.0);
    std::pair terminal(0.1, 0.05);
    int num_displace = 2;
    double percent = 0.5;
    double prob = 0.5;
    double mesh_size = 0.1;
    int o = 1;
    unsigned seed = 0;

    GenerationParams gen_params_single {data_size, num_branches, len_interval, width_interval, flip_params,
        data_rotation, youngs_modulus, poisson, yield_stress, num_perturb, per_prob,
        width, vector, terminal, num_displace, percent, prob, mesh_size, o, seed};

    auto single_dataset = DataOperations::generateParametrizationDataSet(gen_params_single);

    printParametrizationDataSet(single_dataset);

    GenerationParams gen_params_multi {data_size, 4, len_interval, width_interval, flip_params,
        data_rotation, youngs_modulus, poisson, yield_stress, num_perturb, per_prob,
        width, vector, terminal, num_displace, percent, prob, mesh_size, o, seed};

    auto multi_dataset = DataOperations::generateParametrizationDataSet(gen_params_multi);

    printParametrizationDataSet(multi_dataset);
}

void test_parametrizationToPoint() {
    int data_size = 20;
    int num_branches = 1;
    std::pair<double, double> len_interval(3, 5);
    std::pair<double, double> width_interval(0.5, 1);
    std::pair<double, int> flip_params(1, 0);
    std::pair<double, double> data_rotation(360, 1);
    double youngs_modulus = 1000000000;
    double poisson = 0.33;
    double yield_stress = 5000000;
    int num_perturb = 2;
    double per_prob = 1;
    std::pair width (0.1, 0.05);
    std::pair vector(22.5, 1.0);
    std::pair terminal(0.1, 0.05);
    int num_displace = 2;
    double percent = 0.5;
    double prob = 0.5;
    double mesh_size = 0.1;
    int o = 1;
    unsigned seed = 0;

    GenerationParams gen_params_single {data_size, num_branches, len_interval, width_interval, flip_params,
        data_rotation, youngs_modulus, poisson, yield_stress, num_perturb, per_prob,
        width, vector, terminal, num_displace, percent, prob, mesh_size, o, seed};

    auto single_dataset_params = DataOperations::generateParametrizationDataSet(gen_params_single);

    printParametrizationDataSet(single_dataset_params);

    auto single_dataset_points = DataOperations::parametrizationToPoint(single_dataset_params);

    printPointDataSet(single_dataset_points);

    // GenerationParams gen_params_multi {data_size, 4, len_interval, width_interval, flip_params,
    //     data_rotation, youngs_modulus, poisson, yield_stress, num_perturb, per_prob,
    //     width, vector, terminal, num_displace, percent, prob, mesh_size, o, seed};
    //
    // auto multi_dataset_params = DataOperations::generateParametrizationDataSet(gen_params_multi);
    //
    // printParametrizationDataSet(multi_dataset_params);
    //
    // auto multi_dataset_points = DataOperations::parametrizationToPoint(multi_dataset_params);
    //
    // printPointDataSet(multi_dataset_points);
}

void test_saveDataSet() {
    int data_size = 20;
    int num_branches = 1;
    std::pair<double, double> len_interval(3, 5);
    std::pair<double, double> width_interval(0.5, 1);
    std::pair<double, int> flip_params(1, 1);
    std::pair<double, double> data_rotation(360, 1);
    double youngs_modulus = 1000000000;
    double poisson = 0.33;
    double yield_stress = 5000000;
    int num_perturb = 2;
    double per_prob = 1;
    std::pair width (0.1, 0.05);
    std::pair vector(22.5, 1.0);
    std::pair terminal(0.1, 0.05);
    int num_displace = 2;
    double percent = 0.5;
    double prob = 0.5;
    double mesh_size = 0.1;
    int o = 1;
    unsigned seed = 0;

    GenerationParams gen_params_single {data_size, num_branches, len_interval, width_interval, flip_params,
        data_rotation, youngs_modulus, poisson, yield_stress, num_perturb, per_prob,
        width, vector, terminal, num_displace, percent, prob, mesh_size, o, seed};

    auto single_dataset_params = DataOperations::generateParametrizationDataSet(gen_params_single);

    printParametrizationDataSet(single_dataset_params);
    auto file_name_single = "test_saveDataSet_single";

    DataOperations::saveDataSet(single_dataset_params, file_name_single);

    auto single_dataset_points = DataOperations::parametrizationToPoint(single_dataset_params);
    auto file_name_points_single = "test_saveDataSet_single_points";

    DataOperations::saveDataSet(single_dataset_points, file_name_points_single);

    GenerationParams gen_params_multi {data_size, 4, len_interval, width_interval, flip_params,
        data_rotation, youngs_modulus, poisson, yield_stress, num_perturb, per_prob,
        width, vector, terminal, num_displace, percent, prob, mesh_size, o, seed};

    auto multi_dataset_params = DataOperations::generateParametrizationDataSet(gen_params_multi);

    printParametrizationDataSet(multi_dataset_params);

    auto file_name_multi = "test_saveDataSet_multi";
    DataOperations::saveDataSet(multi_dataset_params, file_name_multi);

    auto multi_dataset_points = DataOperations::parametrizationToPoint(multi_dataset_params);
    auto file_name_points_multi = "test_saveDataSet_multi_points";

    DataOperations::saveDataSet(single_dataset_points, file_name_points_multi);
}

void test_loadDataSet() {
    auto file_name_single = "test_saveDataSet_single";
    auto single_dataset_params = DataOperations::loadDataSet(file_name_single);
    printDataSet(single_dataset_params);

    auto file_name_multi = "test_saveDataSet_multi";
    auto multi_dataset_params = DataOperations::loadDataSet(file_name_multi);
    printDataSet(multi_dataset_params);

    auto file_name_single_points = "test_saveDataSet_single_points";
    auto single_dataset_points = DataOperations::loadDataSet(file_name_single_points);
    printDataSet(single_dataset_points);

    auto file_name_multi_points = "test_saveDataSet_multi_points";
    auto multi_dataset_points = DataOperations::loadDataSet(file_name_multi_points);
    printDataSet(multi_dataset_points);

}

int main() {

    //test_numberToBoolVector();
    //test_sampleWithoutReplacement();
    //test_flipVectorEntry();
    //test_generateRotationAngles();
    //test_rotatePoint();
    //test_findCenter();
    //test_rotateParametrization();
    //test_rotateParametrizationEntry();
    //test_generateMidPointsAndVectors();
    //test_singleBranchDisplacements();
    //test_multiBranchDisplacements();
    //test_generatePeturbedParametrizations();
    //test_generateDisplacementBCs();
    //test_generateSingleBranchParametrization();
    //test_generateMultiBranchParametrization();
    //test_generateParametrizationDataSet();
    test_parametrizationToPoint();
    //test_saveDataSet();
    //test_loadDataSet();

    return 0;
}
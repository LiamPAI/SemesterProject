//
// Created by Liam Curtis on 2024-09-12.
//

#include "../include/data_operations.h"

// The following methods will contain the implementations of the methods declared in data_operations.h, their purpose
// is to build the datasets for the NN to be trained later

// TODO: In the LF_asserts, consider putting the value of variables so we can easily test later
// TODO: Change the signature of any function so that all functions involving random number generation use fields from the GenerationParams struct
// TODO: Decide if it is possible to always have the "first" parametrization be centered around the origin when
//  training a NN, or if they will have to be shifted around and I need to include parametrizations close to the origin
// TODO: IDEA: when taking in a mesh, have some processing that determines min and max "length" + "width" of branches for future training
// TODO change return and initialization statements with std::move if possible, it's more efficient
// TODO: Change this to reflect the fact I created the geometries folder


// TODO: test this function, and determine the order of the bits
// The purpose of the following function is to mimic what a bitset is, it converts a number to its binary form in vector
std::vector<bool> DataOperations::numberToBoolVector(unsigned int number, int bits) {
    std::vector<bool> result(bits);
    for (int i = 0; i < bits; ++i) {
        result[i] = (number & (1 << i)) != 0;
    }
    return result;
}

// TODO: test this function (make sure it handles the case where n = 0)
// The purpose of the following function is to sample between min and max n times without replacement and provide the
// vector of samples, this function is used when determined how many of the potential vectors we would like to flip
// in a parametrization
std::vector<int> DataOperations::sampleWithoutReplacement(int min, int max, int n) {

    int range_size = max - min + 1;
    LF_ASSERT_MSG(max > min, "Incorrect parameters sent to sampleWithoutReplacement, max less than min");
    LF_ASSERT_MSG(n <= range_size, "Incorrect parameters sent to sampleWihtoutReplacement, n greater than range");

    // Initialize the vector of potential number to sample from
    std::vector<int> population(range_size);
    std::iota(population.begin(), population.end(), min);

    // TODO: Consider changing this to a more consistent seed for reproducible results (e.g. the seed in the generation params)
    // Initialize our random number generator for the sampling
    std::random_device rd;
    std::mt19937 gen(rd());

    // Initialize and fill our vector with the samples in the population vector
    std::vector<int> sample(n);
    std::sample(population.begin(), population.end(), sample.begin(), n, gen);

    return sample;
}

// TODO: test this function
// TODO: this function currently doesn't return every single possible combination, it is more of a heuristic,
//  decide later if that is ok or if I want to change that
// TODO: Adjust this function since ParametrizationEntry has now changed
// The purpose of this function is to flip the vectors given within a ParametrizationEntry and return the corresponding
// dataset, the total potential number of flips is 2^((3 + 3) * numBranches) since there are two parametrizations per
// entry, so we use a flip probability and a linear scaling factor so that the number of expected flipped vectors is
// flip_probability * (2 * scaling_factor * numBranches), note that the original params sent to the function is also
// always included in the return vector
ParametrizationDataSet DataOperations::flipVectorEntry(ParametrizationEntry &param, std::pair<double, int> &flip_params, std::mt19937 &rng) {

    // Initialize the data set we will return and add our initial unflipped entry
    ParametrizationDataSet flipped_data_set;
    flipped_data_set.emplace_back(param);

    // Obtain the vector of samples
    int num_potential_flips = std::min(param.numBranches * flip_params.second, int(pow(2, 3 * param.numBranches)));
    auto samples = sampleWithoutReplacement(0, 7, num_potential_flips);
    std::uniform_real_distribution<> probability(0.0, 1.0);

    // Start a loop for the amount of flips we would like to perform
    for (int sample : samples) {
        // Obtain the indices to flip by converting the sampled number to binary
        auto flip_indices = numberToBoolVector(sample, 3);
        bool flipped = false;
        Eigen::VectorXd new_vectors(param.param.vectors);

        for (int i = 0; i < param.numBranches; ++i) {
            // With probability flip_probability, we decide if we flip the vectors of this branch of this
            // parametrization according to the drawn sample
            // Flip vectors for branch i of param1
            if (probability(rng) < flip_params.first) {
                flipped = true;
                for (int j = 0; j < 3; ++j) {
                    if (flip_indices[j]) {
                        new_vectors(2 * i, j) = -new_vectors(2 * i, j);
                        new_vectors(2 * i + 1, j) = -new_vectors(2 * i + 1, j);
                    }
                }
            }
        }
        // If we flipped vectors, we add the flipped ParametrizationEntry to the data_set
        if (flipped) {
            MeshParametrizationData new_param {param.numBranches, param.param.widths,
                                                param.param.terminals, new_vectors};
            flipped_data_set.emplace_back(param.numBranches, new_param,
                                          param.displacements, param.energyDifference);
        }
    }
    return flipped_data_set;
}

// TODO: test this method
// TODO: Make sure this method also generates the angle 0 given "correct" params
// The purpose of the function is generate angles at which we want to rotate a parametrization in order to robustly
// train the NN
std::vector<double> DataOperations::generateRotationAngles(double rotation_granularity, double random_proportion) {

    LF_ASSERT_MSG(rotation_granularity >= 0 and rotation_granularity <= 2 * M_PI,
                  "Invalid rotation_granularity sent to generateRotationAngles, " << rotation_granularity);
    LF_ASSERT_MSG(random_proportion >= 0,
                  "Invalid random_proportion sent to generateRotationAngles, " << random_proportion);

    // Declare vector that will store all angles and calculate number of rotations
    std::vector<double> angles;
    int num_rotations = std::ceil(2 * M_PI / rotation_granularity);

    // Uniform rotations, note that this also includes a "no-rotation" parametrization
    for (int i = 0; i < num_rotations; ++i) {
        angles.push_back(i * rotation_granularity);
    }

    // TODO: Consider putting a flat seed here in order to have reproducible results (perhaps the seed from generation params)
    // Declare random number generator necessary to add a few more random rotations
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 2 * M_PI);

    // Initialize number of desired random rotations, generate random samples, and add to these to the return vector
    int num_random = int(num_rotations * random_proportion);
    for (int i = 0; i < num_random; ++i) {
        angles.push_back(dis(gen));
    }

    return angles;
}


// TODO: test this function, making sure that if the function was given a reference point as its point, it returns the same thing
// The purpose of the following function is to rotate a point by a certain angle, given a reference point, so that we
// can easily rotate branches
Eigen::Vector2d DataOperations::rotatePoint(const Eigen::Vector2d& point, const Eigen::Vector2d& reference, double angle_radians) {
    // Translate point so that reference point is at origin
    Eigen::Vector2d translated = point - reference;

    // Rotate the point around the origin
    double cos_theta = std::cos(angle_radians);
    double sin_theta = std::sin(angle_radians);

    Eigen::Matrix2d rotation_matrix;
    rotation_matrix << cos_theta, -sin_theta,
            sin_theta,  cos_theta;

    Eigen::Vector2d rotated = rotation_matrix * translated;

    // Translate point back to original reference
    return rotated + reference;
}

// TODO: test this function
// The purpose of this function is to return the "center" point of a parametrization, so that we have a reference point
// with which to rotate our parametrizations. In the 1-branch case, I simply return the middle terminal. In the
// multi-branch case, I return the geometric centroid, assuming a convex polygon
Eigen::Vector2d DataOperations::findCenter(MeshParametrizationData &params) {

    // If the amount of branches is 1, we simply return the middle terminal
    if (params.numBranches == 1) {
        return Eigen::Vector2d(params.terminals.block<2, 1>(0, 1));
    }

    // If the amount of branches is 3+, we calculate the geometric centroid of the shape, assuming correct ordering of
    // the terminals within the params struct, and our assumption of a convex polygon
    else if (params.numBranches >= 3) {
        Eigen::Vector2d center = Eigen::Vector2d::Zero();
        for (int i = 0; i < params.numBranches; ++i) {
            center += params.terminals.block<2, 1>(2 * i, 0);
        }
        return center / params.numBranches;
    }

    else {
        LF_ASSERT_MSG(false, "The parametrization params sent to findCenter have an "
                             "incorrect number of branches, " << params.numBranches);
    }
}

// TODO: test this function with both single and multi branch parametrizations (which I think it will by design)
// TODO: make sure this function returns the base parametrization only if the rotation_params demand it
// TODO: Fix this function, as the vectors of a parametrization are currently not rotating with it
// The purpose of the following function is to rotate a parametrization given certain rotation parameters (rotation
// granularity, and an additional random proportion) and a return a vector of all these rotated parametrizations,
// note additionally that the original parametrization will also be included in the return vector due to the
// implementation of generateRotationAngles
std::vector<MeshParametrizationData> DataOperations::rotateParametrization(MeshParametrizationData &params, std::pair<double, double> &rotation_params) {

    std::vector<MeshParametrizationData> rotated_data_set;

    // Initialize angles vector and the center point for which to rotate from
    std::vector<double> angles = generateRotationAngles(rotation_params.first, rotation_params.second);
    Eigen::Vector2d center = findCenter(params);

    // Iterate through all angles and rotate the param according to that angle
    for (double angle : angles) {
        Eigen::MatrixXd rotated_terminals (params.terminals.rows(), params.terminals.cols());
        rotated_terminals.setZero();

        // Iterate through all branches and rotate each terminal of that branch
        for (int j = 0; j < params.numBranches; ++j) {
            rotated_terminals.block<2, 1>(2 * j, 0) =
                    rotatePoint(params.terminals.block<2, 1>(2 * j, 0), center, angle);
            rotated_terminals.block<2, 1>(2 * j, 1) =
                    rotatePoint(params.terminals.block<2, 1>(2 * j, 1), center, angle);
            rotated_terminals.block<2, 1>(2 * j, 2) =
                    rotatePoint(params.terminals.block<2, 1>(2 * j, 2), center, angle);
        }

        // TODO: Double check that with std::move this constructor doesn't mess up my original parametrization
        // Add new parametrization to our vector
        rotated_data_set.emplace_back(params.numBranches, Eigen::VectorXd(params.widths),
                                      rotated_terminals, Eigen::VectorXd(params.vectors));

    }
    return rotated_data_set;

}

// TODO: test this function
// TODO: make sure this function returns the base parametrization only if the rotation_params demand it
// This function has the exact same functionality as rotateParametrization but does it for type ParametrizationEntry,
// where both parametrizations and the vector of displacements are rotated an equal amount, and the energy is kept
// the same. This helps for the NN to be rotation-invariant
ParametrizationDataSet DataOperations::rotateParametrizationEntry(ParametrizationEntry &param_entry,
                                                                  std::pair<double, double> &rotation_params) {
    ParametrizationDataSet rotated_data_set;

    // Initialize angles vector and find the center point of the first parametrization
    std::vector<double> angles = generateRotationAngles(rotation_params.first, rotation_params.second);
    Eigen::Vector2d center = findCenter(param_entry.param);
    Eigen::Vector2d origin;
    origin.setZero();

    // Iterate through all angles and rotate the param_entry according to that angle
    for (double angle : angles) {
        // Declare the new matrices for this rotated parametrization
        Eigen::MatrixXd rotated_terminals(param_entry.param.terminals.rows(), param_entry.param.terminals.cols());
        Eigen::MatrixXd rotated_vectors(param_entry.param.vectors.rows(), param_entry.param.vectors.cols());
        rotated_terminals.setZero();
        rotated_vectors.setZero();

        // Iterate through all branches and rotate each terminal and vector of that branch
        for (int j = 0; j < param_entry.numBranches; ++j) {
            rotated_terminals.block<2, 1>(2 * j, 0) =
                    rotatePoint(param_entry.param.terminals.block<2, 1>(2 * j, 0), center, angle);
            rotated_terminals.block<2, 1>(2 * j, 1) =
                    rotatePoint(param_entry.param.terminals.block<2, 1>(2 * j, 1), center, angle);
            rotated_terminals.block<2, 1>(2 * j, 2) =
                    rotatePoint(param_entry.param.terminals.block<2, 1>(2 * j, 2), center, angle);

            rotated_vectors.block<2, 1>(2 * j, 0) =
                    rotatePoint(param_entry.param.vectors.block<2, 1>(2 * j, 0), origin, angle);
            rotated_vectors.block<2, 1>(2 * j, 1) =
                    rotatePoint(param_entry.param.vectors.block<2, 1>(2 * j, 1), origin, angle);
            rotated_vectors.block<2, 1>(2 * j, 2) =
                    rotatePoint(param_entry.param.vectors.block<2, 1>(2 * j, 2), origin, angle);
        }

        // Declare the new matrix for the displacement vectors
        Eigen::MatrixXd rotated_displacements(param_entry.displacements.rows(), param_entry.displacements.cols());
        rotated_displacements.setZero();

        // Iterate through the displacement vectors and rotate them
        for (int i = 0; i < param_entry.displacements.rows() / 2; ++i) {
            for (int j = 0; j < param_entry.displacements.cols(); ++j) {
                rotated_displacements.block<2, 1>(2 * i, j) =
                        rotatePoint(param_entry.displacements.block<2, 1>(2 * i, j), origin, angle);
            }
        }

        // Add new parametrization to our vector
        MeshParametrizationData new_param {param_entry.numBranches, Eigen::MatrixXd(param_entry.param.widths),
                                            rotated_terminals, rotated_vectors};

        rotated_data_set.emplace_back(param_entry.numBranches, new_param,
                                      rotated_displacements, param_entry.energyDifference);
    }
    return rotated_data_set;

}

// TODO: test this function, especially with respect to the ordering of the final midpoints and vectors matrix
// The purpose of this function is to take in a vector of side lengths, and return the midpoints of the sides of a
// convex polygon with the given side lengths, along with the corresponding outward pointing unit normal vector, and
// unit normal vectors between vertices, which is useful for generating multi-branch parametrizations
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> DataOperations::generateMidPointsAndVectors(Eigen::VectorXd &side_lengths) {

    // Declare/initialize helper variables
    int n = side_lengths.size();
    int prev, next;
    Eigen::MatrixXd vertices(2, n);
    double total_angle = 0;
    double x = 0, y = 0;
    double a, b, c, angle;

    // TODO: Decide if I should put the calculate angle functionality in a different function, though I think it's fine to keep here
    for (int i = 0; i < n; ++i) {
        // Determine the previous and next indices around index i
        prev = (i - 1 + n) % n;
        next = (i + 1) % n;

        // Obtain the side lengths at the previous, current, and next index
        a = side_lengths[prev];
        b = side_lengths[i];
        c = side_lengths[next];

        // Law of cosines to determine the angle at this vertex using the side lengths
        angle = std::acos((a * a + b * b - c * c) / (2 * a * b));
        total_angle += angle;

        x += side_lengths[i] * std::cos(total_angle);
        y += side_lengths[i] * std::sin(total_angle);

        vertices.col(i) << x, y;
    }

    // Center the vertices around the origin using the geometric centroid
    Eigen::Vector2d centroid = vertices.rowwise().mean();
    vertices.colwise() -= centroid;

    // Initialize return matrices and helper vectors
    Eigen::MatrixXd midpoints(2, n);
    Eigen::MatrixXd outward_vectors(2, n);
    Eigen::MatrixXd side_vectors(2, n);

    // This loop will help initialize our return matrices
    for (int i = 0; i < n; ++i) {
        // Find the next index of the vector, considering a "circular" polygon
        next = (i + 1) % n;

        // Find midpoints, side_vector, and out_vector
        midpoints.col(i) = (vertices.col(i) + vertices.col(next)) / 2.0;
        side_vectors.col(i) = vertices.col(next) - vertices.col(i);
        side_vectors.col(i).normalize();
        outward_vectors.col(i) << -side_vectors.col(i).y(), side_vectors.col(i).x();

        // Ensure the vector points outward
        if (outward_vectors.col(i).dot(midpoints.col(i)) < 0) {
            outward_vectors.col(i) = -outward_vectors.col(i);
        }
    }

    // Return the tuple of matrices for parametrization generation
    return {midpoints, outward_vectors, side_vectors};
}

// TODO: test this function
// This function takes in a single-branch parametrization and returns a matrix of displacement vectors to perturb it
// according to our perturbation parameters
Eigen::MatrixXd DataOperations::singleBranchDisplacements(PerturbationGenerators &param_gens) {
    // Declare matrix that will hold our displacement vectors
    Eigen::MatrixXd displacements(4, 2);
    displacements.setZero();

    // Iterate through the 4 corners of this branch and generate displacement vectors
    for (int i = 0; i < 4; ++i) {

        // Add a displacement boundary condition with probability perturbProbability for each coordinate
        if (param_gens.uniformPerturb(param_gens.rng) < param_gens.perturbProbability) {
            displacements(i % 2, i / 2) += param_gens.displacementPerturb(param_gens.rng)
                    * param_gens.maxDistances[0];
        }

        if (param_gens.uniformPerturb(param_gens.rng) < param_gens.perturbProbability) {
            displacements(i % 2 + 1, i / 2) += param_gens.displacementPerturb(param_gens.rng)
                    * param_gens.maxDistances[0];
        }
    }

    // Return the ordered displacement vectors
    return displacements;
}

// TODO: test this function
// TODO: Change this function to reflect the fact that we just want displacement vectors as the second thing
// This function takes in a multi-branch parametrization and returns a matrix of displacement vectors to perturb it
// according to our perturbation parameters
Eigen::MatrixXd DataOperations::multiBranchDisplacements(int num, PerturbationGenerators &param_gens) {
    // Declare matrix that will hold our displacement vectors
    Eigen::MatrixXd displacements(num * 4, 1);
    displacements.setZero();

    // Iterate through all branches to add displacement boundary conditions one at a time
    for (int i = 0; i < num; ++i) {

        // Iterate through the 2 points for which we'll have displacement BCs (4 total with 2D x and y)
        for (int j = 0; j < 4; ++j) {

            // There can only be displacement BCs at the ends of each branch, the remaining sides have Neumann BCs
            if (param_gens.uniformPerturb(param_gens.rng) < param_gens.perturbProbability) {
                displacements(4 * i + j, 0) += param_gens.displacementPerturb(param_gens.rng)
                                           * param_gens.maxDistances[i];
            }
        }
    }

    // Return the ordered displacement vectors
    return displacements;
}

// TODO: test this function
// TODO: IDEA: Perhaps include a print statement that indicates the percentage of generated perturbations that are in linear elastic region
// TODO: Ensure that widths remains positive, perhaps check using meshParamValidator
// TODO: make sure this function returns the base parametrization only if the perturbation_params demand it
// The purpose of this function is to take in a parametrization, ideally unperturbed (though it can be), and return
// a vector of parametrizations that are very slightly altered from the original (e.g. bent, elongated, etc.) that
// are all within the linear elastic region of the material
ParametrizationDataSet DataOperations::generatePerturbedParametrizations(MeshParametrizationData &base_param,
                                                                       PerturbationParams &params) {
    // Declare vector of max distances, which will be the variance when generating displacements
    std::vector<double> max_distances;
    ParametrizationDataSet perturbed_entries;

    if (base_param.numBranches == 1) {

        // Use the following heuristic to determine the maximum allowable strain for calculating displacement vectors
        double max_stress = params.percentYieldStrength * params.yieldStrength;
        double max_strain = max_stress / params.modulusOfElasticity;

        // Obtain the std deviation we'll be using when rotating the vectors
        // TODO: Modify this heuristic if needed after testing, i.e. consider using the width over the length, since
        //  the width will be significantly smaller
        double approx_length = (base_param.terminals.col(1) - base_param.terminals.col(0)).norm() +
                (base_param.terminals.col(2) - base_param.terminals.col(1)).norm();

        max_distances.emplace_back(approx_length * max_strain);

        PerturbationGenerators perturb_params = {params.perturbProbability, max_distances, params.rng};

        calculationParams calc_energy_params {params.yieldStrength, params.modulusOfElasticity,
            params.poissonRatio,params.meshSize, params.order};

        // Keep adding data entries until we reach our desired size for this base_param
        // TODO: Implement something to prevent an infinite loop here, and perhaps add a print statement for testing purposes
        while (perturbed_entries.size() < params.numPerturbations) {
            auto displacements = singleBranchDisplacements(perturb_params);

            std::pair<bool, double> energy_check = MeshParametrization::displacementEnergy(base_param,
                displacements, calc_energy_params);

            if (energy_check.first) {
                perturbed_entries.emplace_back(1, base_param, displacements, energy_check.second);
            }
        }
    }

    // Same as in the single branch case, but slightly modified to handle multiple branches
    else if (base_param.numBranches >= 3) {

        // Same calculation as in the single branch case, calculate a maximum allowable strain heuristic
        // for displacement vectors
        double max_stress = params.percentYieldStrength * params.yieldStrength;
        double max_strain = max_stress / params.modulusOfElasticity;

        for (int i = 0; i < base_param.numBranches; ++i) {
            // Obtain the length and use this to find the maximum allowable distance for this branch
            // TODO: Modify this heuristic if needed after testing
            double approx_length = (base_param.terminals.block<2, 1>(2 * i, 1) - base_param.terminals.block<2, 1>(2 * i, 0)).norm() +
                                   (base_param.terminals.block<2, 1>(2 * i, 2) - base_param.terminals.block<2, 1>(2 * i, 1)).norm();
            max_distances.emplace_back(approx_length * max_strain);
        }

        PerturbationGenerators perturb_params = {params.perturbProbability, max_distances, params.rng};

        calculationParams calc_energy_params {params.yieldStrength, params.modulusOfElasticity,
            params.poissonRatio,params.meshSize, params.order};

        // Keep adding data entries until we reach our desired size for this base_param
        // TODO: Implement something to prevent an infinite loop here, and perhaps add a print statement for testing purposes
        while (perturbed_entries.size() < params.numPerturbations) {
            auto displacements = multiBranchDisplacements(base_param.numBranches, perturb_params);

            std::pair<bool, double> energy_check = MeshParametrization::displacementEnergy(base_param,
                displacements, calc_energy_params);

            if (energy_check.first) {
                perturbed_entries.emplace_back(1, base_param, displacements, energy_check.second);
            }
        }
    }

    else {
        LF_ASSERT_MSG(false, "Wrong number of branches sent to generatePerturbedParametrizations, "
        << base_param.numBranches);
    }

    return perturbed_entries;
}

// TODO: test this function
// The purpose of this function is to take the number of branches, a width, and a length, and output a base
// parametrization to add to the dataset. The base parametrization is a straight horizontal branch in this case
MeshParametrizationData DataOperations::generateSingleBranchParametrization(int num, double width, double length) {

    if (num == 1) {
        // Declare necessary matrices to initialize a parametrization
        Eigen::MatrixXd widths(1, 3);
        Eigen::MatrixXd terminals(2, 3);
        Eigen::MatrixXd vectors (2, 3);

        // Initialize necessary matrices with a non-varying width, length exactly as expected, and all vectors pointing up
        widths << width, width, width;
        terminals << -length / 2, 0.0, length / 2, 0.0, 0.0, 0.0;
        vectors << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

        // Return the MeshParametrization Data we want
        return {num, widths, terminals, vectors};
    }
    else {
        LF_ASSERT_MSG(false, "Incorrect number of branches sent to generateSingleBranchParametrization, " << num);
    }

}

// TODO: test this function
// The purpose of this function is to take the number of branches, a vector of widths, a vector of lengths, and output
// a base multi-branch parametrization to add to the dataset. The branch will likely be modified afterwards. The base
// parametrization will be centered around the origin, with straight branches protruding out of it of constant width
MeshParametrizationData DataOperations::generateMultiBranchParametrization(int num, Eigen::VectorXd &widths,
                                                                           Eigen::VectorXd &lengths) {
    if (num >= 3) {
        // Declare matrices that will take part in the final parametrization
        Eigen::MatrixXd terminals(num * 2, 3);
        Eigen::MatrixXd vectors(num * 2, 3);
        Eigen::MatrixXd new_widths = widths.replicate(1, 3);

        // The following is a tuple containing the midpoints, unit outward pointing vectors, and unit side vectors
        auto midpoints_and_vectors = generateMidPointsAndVectors(widths);

        // Iterate through all branches and return desired vector
        for (int i = 0; i < num; ++i) {
            // Initialize the 3 vectors for this branch, which are the same since we assume a straight branch
            vectors.block<2, 3>(i * 2, 0) << std::get<2>(midpoints_and_vectors).col(i).replicate(1, 3);

            // Initialize the terminals using the given length and outward pointing vectors
            terminals.block<2, 1>(i * 2, 0) << std::get<0>(midpoints_and_vectors).block<2, 1>(0, i);
            terminals.block<2, 1>(i * 2, 1) = terminals.block<2, 1>(i * 2, 0) +
                    std::get<1>(midpoints_and_vectors).block<2, 1>(0, i) * lengths[i] / 2;
            terminals.block<2, 1>(i * 2, 2) = terminals.block<2, 1>(i * 2, 1) +
                    std::get<1>(midpoints_and_vectors).block<2, 1>(0, i) * lengths[i] / 2;
        }
        return {num, new_widths, terminals, vectors};
    }

    else {
        LF_ASSERT_MSG(false, "Incorrect number of branches sent to generateMultiBranchParametrization, " << num);
    }
}

// TODO: test this function
// TODO: IDEA: If it leaves the linear elastic region, make the energy difference high such that the NN wants to avoid it
// TODO: Consider adding print statements (perhaps with a bool of verbose) to see where we are within the parametrization generation process
// TODO: Find a way to perturb the widths of the base parametrizations, otherwise they will likely always be constant
// The purpose of this function is to generate an entire data set full of the "MeshParametrizationData" type using the
// parameters given in params and the corresponding GenerationParams struct
ParametrizationDataSet DataOperations::generateParametrizationDataSet(GenerationParams &params) {

    // Declare the dataset and reserve space for it
    ParametrizationDataSet data_set;
    data_set.reserve(params.datasetSize);

    // Initialize necessary random number generators for width and length, and perturbation parameters
    std::uniform_real_distribution<> length_dist(params.lengthInterval.first, params.lengthInterval.second);
    std::uniform_real_distribution<> width_dist(params.widthInterval.first, params.widthInterval.second);
    PerturbationParams perturb_params {params};

    // This vector will hold the correct value of the tags throughout the loop
    std::vector<int> tagCounter(NUM_TAGS, -1);

    // Iterate until we have our desired dataset size
    while (data_set.size() < params.datasetSize) {

        MeshParametrizationData base_param;

        if (params.numBranches == 1) {

            // Generate a base parametrization with the generated length and width
            double length = length_dist(params.rng);
            double width = width_dist(params.rng);
            base_param = generateSingleBranchParametrization(1, width, length);

        }

        else if (params.numBranches >= 3) {

            // Initialize vector of widths and lengths for each branch
            Eigen::VectorXd lengths(params.numBranches);
            Eigen::VectorXd widths(params.numBranches);
            for (int i = 0; i < params.numBranches; ++i) {
                lengths(i) = length_dist(params.rng);
                widths(i) = width_dist(params.rng);
            }
            // Generate base multi-branch parametrization
            base_param = generateMultiBranchParametrization(params.numBranches, widths, lengths);

        }

        else {
            LF_ASSERT_MSG(false, "params has wrong number of branches sent to generateParametrizationDataset, " << params.numBranches);
        }
        // With a base_param now initialized, we update the index in tags
        tagCounter[TagIndex::BASE]++;

        // Generate rotated versions of the base parametrization
        std::vector<MeshParametrizationData> rotated_base = rotateParametrization(base_param,
                                                                                  params.baseRotationParams);

        // Iterate through the rotated versions of base and create perturbations of each, update tag
        for (MeshParametrizationData rotated_base_param : rotated_base) {
            tagCounter[TagIndex::BASE_ROTATION]++;

            // Create a perturbed dataset for each of these rotated parametrizations
            ParametrizationDataSet perturbed_rotated_base = generatePerturbedParametrizations(rotated_base_param,
                                                                                              perturb_params);

            // Iterate through each perturbed entry and rotate, update tags as well
            for (ParametrizationEntry perturb_rot_base : perturbed_rotated_base) {
                tagCounter[TagIndex::PERTURBATION]++;
                ParametrizationDataSet rotated_twice = rotateParametrizationEntry(perturb_rot_base,
                                                                                  params.dataRotationParams);

                // Iterate through each twice rotated + perturbed entry, flip them according to flip params,
                for (ParametrizationEntry rot_perturb_rot : rotated_twice) {
                    tagCounter[TagIndex::FLIP_VECTOR]++;
                    ParametrizationDataSet flip_perturb_rot = flipVectorEntry(rot_perturb_rot,
                                                                              params.flipParams, params.rng);

                    // Add the tags to each entry and add to the overall dataset
                    std::transform(flip_perturb_rot.begin(), flip_perturb_rot.end(),
                                   std::back_inserter(data_set),
                                   [&tagCounter] (ParametrizationEntry entry) {
                                        entry.tags = tagCounter;
                                        return entry;
                    });
                }
            }
        }
    }

    // Trim the dataset to the exact size we want and return it
    if (data_set.size() > params.datasetSize) {
        data_set.erase(data_set.begin() + params.datasetSize, data_set.end());
    }
    return data_set;
}

// TODO: test this function
// TODO: for testing purposes, perhaps have a counter of the number of time we see a tag twice
// This function takes in a parametrization data set and converts it into a point data set using the functionality
// in polynomialPoints. Since the point parametrization doesn't depend on vector orientations, if the vector flip tag
// is a tag we've already seen, we skip over it rather than add the same entry twice
PointDataSet DataOperations::parametrizationToPoint(ParametrizationDataSet &paramSet) {
    PointDataSet data_set;
    data_set.reserve(paramSet.size());

    std::unordered_set<int> seen_tags;

    // Iterate through all entries in the initial dataset
    for (ParametrizationEntry param_entry: paramSet) {

        // If we can't find this tag within our set, we can add this entry, since it is "unique"
        if (seen_tags.find(param_entry.tags[TagIndex::FLIP_VECTOR]) == seen_tags.end()) {

            // Use polynomialPoints to convert each parametrization correctly then add to the data set
            ParametrizationPoints points{param_entry.numBranches,
                                          MeshParametrization::polynomialPoints(param_entry.param)};
            data_set.emplace_back(param_entry.numBranches, points,
                                  param_entry.displacements, param_entry.energyDifference);

            // Add this tag to our set of seen tags since this is new
            seen_tags.insert(param_entry.tags[TagIndex::FLIP_VECTOR]);
        }
    }
    return data_set;
}

// TODO: test this function
// This function takes in a file and MeshParametrizationData object and saves the object to the file
void DataOperations::saveMeshParametrizationData(std::ofstream &file, const MeshParametrizationData &param) {
    file.write(reinterpret_cast<const char*>(&param.numBranches), sizeof(int));

    // Save widths, current shape is numBranches
    file.write(reinterpret_cast<const char*>(param.widths.data()), 3 * param.numBranches * sizeof(double));

    // Save terminals, current shape will be (2 * numBranches) by 3
    file.write(reinterpret_cast<const char*>(param.terminals.data()), 6 * param.numBranches * sizeof(double));

    // Save vectors, current shape will be (2 * numBranches) by 3
    file.write(reinterpret_cast<const char*>(param.vectors.data()), 6 * param.numBranches * sizeof(double));
}

// TODO: test this function
// This function takes in a file we're reading to return a MeshParametrization object
MeshParametrizationData DataOperations::loadMeshParametrizationData(std::ifstream &file) {
    int numBranches;
    file.read(reinterpret_cast<char*>(&numBranches), sizeof(int));

    Eigen::MatrixXd widths(numBranches, 3);
    file.read(reinterpret_cast<char*>(widths.data()), 3 * numBranches * sizeof(double));

    // Load the points into the matrix
    Eigen::MatrixXd terminals(2 * numBranches, 3);
    file.read(reinterpret_cast<char*>(terminals.data()), 6 * numBranches * sizeof(double));

    // Load the points into the matrix
    Eigen::MatrixXd vectors(2 * numBranches, 3);
    file.read(reinterpret_cast<char*>(vectors.data()), 6 * numBranches * sizeof(double));

    return {numBranches, std::move(widths), std::move(terminals), std::move(vectors)};
}

// TODO: test this function
// This function takes in a file and ParametrizationPoints object and saves the object to the file
void DataOperations::saveParametrizationPoints(std::ofstream &file, const ParametrizationPoints &points) {
    file.write(reinterpret_cast<const char*>(&points.numBranches), sizeof(int));

    // Save points, current shape is (4 * numBranches) by 3
    file.write(reinterpret_cast<const char*>(points.points.data()), 12 * points.numBranches * sizeof(double));
}

// TODO: test this function
// This function takes in a file we're reading to return a ParametrizationPoints object
ParametrizationPoints DataOperations::loadParametrizationPoints(std::ifstream &file) {
    int numBranches;
    file.read(reinterpret_cast<char*>(&numBranches), sizeof(int));

    // Load the points into the matrix
    Eigen::MatrixXd points(4 * numBranches, 3);
    file.read(reinterpret_cast<char*>(points.data()), 12 * numBranches * sizeof(double));

    return {numBranches, std::move(points)};
}

// TODO: test this function
// TODO: Change this to retrieve/store the data in the data folder once files are restructured
// This function takes in a DataSet, identifies whether it's filled with Parametrization entries or Point entries,
// and saves the corresponding structure into a binary file for later use with a given file name
void DataOperations::saveDataSet(const DataSet &data_set, const std::string &file_name) {
    // TODO: Find out if the path works for this
    // Declare the file
    std::ofstream file(file_name, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file for writing within saveDataSet: " + file_name);
    }

    // Find the type of this dataset and write it to the file
    DataSetType type = std::holds_alternative<ParametrizationDataSet>(data_set) ?
            DataSetType::Parametrization : DataSetType::Point;
    file.write(reinterpret_cast<const char*>(&type), sizeof(DataSetType));

    // Check which type this data set holds in order to see how to save the data set
    if (std::holds_alternative<ParametrizationDataSet>(data_set)) {

        // Write the size of the data set at the beginning of the file
        const auto &param_data_set = std::get<ParametrizationDataSet>(data_set);
        size_t size = param_data_set.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

        // Iterate through each entry and save it
        for (const auto &entry : param_data_set) {
            file.write(reinterpret_cast<const char*>(&entry.numBranches), sizeof(int));
            saveMeshParametrizationData(file, entry.param);

            // Calculate number of values used in the displacement matrix
            int num_displacements = entry.numBranches == 1 ? 8 : 4 * entry.numBranches;
            file.write(reinterpret_cast<const char *>(&entry.displacements), num_displacements * sizeof(double));

            file.write(reinterpret_cast<const char*>(&entry.energyDifference), sizeof(double));
            file.write(reinterpret_cast<const char*>(entry.tags.data()), NUM_TAGS * sizeof(int));
        }
    }

    else {

        // Write the size of the data set at the beginning of the file
        const auto &point_data_set = std::get<PointDataSet>(data_set);
        size_t size = point_data_set.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

        // Iterate through each entry and save it
        for (const auto &entry : point_data_set) {
            file.write(reinterpret_cast<const char*>(&entry.numBranches), sizeof(int));
            saveParametrizationPoints(file, entry.points);

            // Calculate number of values used in the displacement matrix
            int num_displacements = entry.numBranches == 1 ? 8 : 4 * entry.numBranches;
            file.write(reinterpret_cast<const char *>(&entry.displacements), num_displacements * sizeof(double));

            file.write(reinterpret_cast<const char*>(&entry.energyDifference), sizeof(double));
            file.write(reinterpret_cast<const char*>(entry.tags.data()), NUM_TAGS * sizeof(int));
        }
    }
}

// TODO: test this function
// TODO: Change this to retrieve/store the data in the data folder once files are restructured
// This function is given a file name, identifies whether it contains Parametrization entires or Point entires,
// and returns the corresponding dataset
DataSet DataOperations::loadDataSet(const std::string &file_name) {
    // TODO: Find out if the path works for this
    std::ifstream file(file_name, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file for reading within loadDataSet: " + file_name);
    }

    // Read the type of this data set so that we know how to read it
    DataSetType type;
    file.read(reinterpret_cast<char*>(&type), sizeof(DataSetType));

    if (type == DataSetType::Parametrization) {

        // Obtain the size of this data set
        ParametrizationDataSet data_set;
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size_t));

        // Iterate through all entries given the size we just acquired, and load each field of the entry
        for (size_t i= 0; i < size; i++) {
            int num_branches;
            file.read(reinterpret_cast<char*>(&num_branches), sizeof(int));

            MeshParametrizationData param = loadMeshParametrizationData(file);

            int num_cols = num_branches == 1 ? 2 : 1;
            Eigen::MatrixXd displacements(4 * num_branches, num_cols);
            file.read(reinterpret_cast<char*>(&displacements), 4 * num_branches * num_cols * sizeof(double));

            double energy_diff;
            file.read(reinterpret_cast<char*>(&energy_diff), sizeof(double));

            ParametrizationEntry entry(num_branches, std::move(param), std::move(displacements), energy_diff);

            file.read(reinterpret_cast<char*>(entry.tags.data()), NUM_TAGS * sizeof(int));

            data_set.push_back(std::move(entry));
        }
        return data_set;
    }

    else {

        // Obtain the size of this data set
        PointDataSet data_set;
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size_t));

        // Iterate through all entries given the size we just acquired
        for (size_t i= 0; i < size; i++) {
            int num_branches;
            file.read(reinterpret_cast<char*>(&num_branches), sizeof(int));

            ParametrizationPoints points = loadParametrizationPoints(file);

            int num_cols = num_branches == 1 ? 2 : 1;
            Eigen::MatrixXd displacements(4 * num_branches, num_cols);
            file.read(reinterpret_cast<char*>(&displacements), 4 * num_branches * num_cols * sizeof(double));

            double energy_diff;
            file.read(reinterpret_cast<char*>(&energy_diff), sizeof(double));

            PointEntry entry(num_branches, std::move(points), std::move(displacements), energy_diff);

            file.read(reinterpret_cast<char*>(entry.tags.data()), NUM_TAGS * sizeof(int));

            data_set.push_back(std::move(entry));
        }
        return data_set;
    }
}
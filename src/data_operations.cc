//
// Created by Liam Curtis on 2024-09-12.
//

#include "../include/data_operations.h"

// The following methods will contain the implementations of the methods declared in data_operations.h, their purpose
// is to build the datasets for the NN to be trained later

// This function mimics what a bitset is, it converts a number to its binary form in a vector, where the first index
// is in the ones place
std::vector<bool> DataOperations::numberToBoolVector(unsigned int number, int bits) {
    std::vector<bool> result(bits);
    for (int i = 0; i < bits; ++i) {
        result[i] = (number & (1 << i)) != 0;
    }
    return result;
}

// The purpose of the following function is to sample between min and max n times without replacement and provide the
// vector of samples, this function is used when determined how many of the potential vectors we would like to flip
// in a parametrization
std::vector<int> DataOperations::sampleWithoutReplacement(int min, int max, int n, std::mt19937 &rng) {

    int range_size = max - min + 1;
    LF_ASSERT_MSG(max > min, "Incorrect parameters sent to sampleWithoutReplacement, max less than min");
    LF_ASSERT_MSG(n <= range_size, "Incorrect parameters sent to sampleWihtoutReplacement, n greater than range");

    // Initialize the vector of potential numbers to sample from
    std::vector<int> population(range_size);
    std::iota(population.begin(), population.end(), min);

    // Initialize and fill our vector with the samples in the population vector
    std::vector<int> sample(n);
    std::sample(population.begin(), population.end(), sample.begin(), n, rng);

    return sample;
}

// The purpose of this function is to flip the vectors given within a ParametrizationEntry and return the corresponding
// dataset, the total potential number of flips is 2^(3 * numBranches), so we use a flip probability and a
// linear scaling factor so that the number of expected flipped vectors is
// flip_probability * scaling_factor * numBranches, note that the original parametrization entry sent to the
// function is also always included in the return vector
ParametrizationDataSet DataOperations::flipVectorEntry(ParametrizationEntry &param_entry,
    std::pair<double, int> &flip_params, std::mt19937 &rng) {

    // Initialize the data set we will return and add our initial un-flipped entry
    ParametrizationDataSet flipped_data_set;
    flipped_data_set.emplace_back(param_entry);

    // Obtain the vector of samples
    int num_flips = std::min(param_entry.numBranches * flip_params.second,
        int(pow(2, 3 * param_entry.numBranches)) - 1);
    int num_potential_flips = int(pow(2, 3 * param_entry.numBranches)) - 1;
    auto samples = sampleWithoutReplacement(1, num_potential_flips, num_flips, rng);

    std::uniform_real_distribution<> probability(0.0, 1.0);

    // Start a loop for the amount of flips we would like to perform
    for (int sample : samples) {

        // Obtain the indices to flip by converting the sampled number to binary
        auto flip_indices = numberToBoolVector(sample, 3 * param_entry.numBranches);
        bool flipped = false;
        Eigen::MatrixXd new_vectors(param_entry.param.vectors);

        // Flip the vectors according to the 1's in the vector and our flip probability
        if (probability(rng) < flip_params.first) {
            flipped = true;
            for (int j = 0; j < 3 * param_entry.numBranches; ++j) {
                if (flip_indices[j]) {
                    new_vectors(2 * (j / 3), j % 3) = -new_vectors(2 * (j / 3), j % 3);
                    new_vectors(2 * (j / 3) + 1, j % 3) = -new_vectors(2 * (j / 3) + 1, j % 3);
                }
            }
        }

        // If we flipped vectors, we add the flipped ParametrizationEntry to the data_set, this is to make sure we
        // don't have a repeat of the entry we received
        if (flipped) {
            MeshParametrizationData new_param {param_entry.numBranches, param_entry.param.widths,
                                                param_entry.param.terminals, new_vectors};
            flipped_data_set.emplace_back(param_entry.numBranches, new_param,
                                          param_entry.displacements, param_entry.energyDifference);
        }
    }
    return flipped_data_set;
}

// The purpose of the function is generate angles at which we want to rotate a parametrization in order to robustly
// train the NN, note this method can return only the angle 0 if given params 360, 0, respectively
std::vector<double> DataOperations::generateRotationAngles(double rotation_granularity, double random_proportion, std::mt19937 &rng) {

    // Declare vector that will store all angles and calculate number of rotations
    std::vector<double> angles;
    int num_rotations = std::ceil(360.0 / rotation_granularity);

    // Uniform rotations, note that this also includes a "no-rotation" parametrization
    for (int i = 0; i < num_rotations; ++i) {
        angles.push_back(i * rotation_granularity);
    }

    // Declare random number generator necessary to add a few more random rotations
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 360);

    // Initialize number of desired random rotations, generate random samples, and add to these to the return vector
    int num_random = int(num_rotations * random_proportion);
    for (int i = 0; i < num_random; ++i) {
        angles.push_back(dis(gen));
    }
    return angles;
}

// The purpose of the following function is to rotate a point by a certain angle, given a reference point, so that we
// can easily rotate branches, note that this functions rotates the point counter-clockwise on the 2D-plane
Eigen::Vector2d DataOperations::rotatePoint(const Eigen::Vector2d& point, const Eigen::Vector2d& reference, double angle_degrees) {
    // Translate point so that reference point is at origin
    Eigen::Vector2d translated = point - reference;

    // Rotate the point around the origin
    double cos_theta = std::cos(angle_degrees * M_PI / 180.0);
    double sin_theta = std::sin(angle_degrees * M_PI / 180.0);

    Eigen::Matrix2d rotation_matrix;
    rotation_matrix << cos_theta, -sin_theta,
            sin_theta,  cos_theta;

    Eigen::Vector2d rotated = rotation_matrix * translated;

    // Translate point back to original reference
    return rotated + reference;
}

// The purpose of this function is to return the "center" point of a parametrization, so that we have a reference point
// with which to rotate our parametrizations. In the 1-branch case, I simply return the middle terminal. In the
// multi-branch case, I return the geometric centroid, assuming a convex polygon
Eigen::Vector2d DataOperations::findCenter(const MeshParametrizationData &params) {

    // If the amount of branches is 1, we simply return the middle terminal
    if (params.numBranches == 1) {
        return Eigen::Vector2d(params.terminals.block<2, 1>(0, 1));
    }

    // If the amount of branches is 3+, we calculate the geometric centroid of the shape, assuming correct ordering of
    // the terminals within the params struct, and our assumption of a convex polygon
    if (params.numBranches >= 3) {
        Eigen::Vector2d center = Eigen::Vector2d::Zero();
        for (int i = 0; i < params.numBranches; ++i) {
            center += params.terminals.block<2, 1>(2 * i, 0);
        }
        return center / params.numBranches;
    }

    LF_ASSERT_MSG(false, "The parametrization params sent to findCenter have an "
                  "incorrect number of branches, " << params.numBranches);
}

// The purpose of the following function is to rotate a parametrization given certain rotation parameters (rotation
// granularity, and an additional random proportion) and a return a vector of all these rotated parametrizations,
// note additionally that the original parametrization will also be included in the return vector due to the
// implementation of generateRotationAngles
std::vector<MeshParametrizationData> DataOperations::rotateParametrization(const MeshParametrizationData &params,
    const std::pair<double, double> &rotation_params, std::mt19937 &rng) {

    std::vector<MeshParametrizationData> rotated_data_set;

    // Initialize angles vector and the center points for which to rotate from
    std::vector<double> angles = generateRotationAngles(rotation_params.first, rotation_params.second, rng);
    Eigen::Vector2d center = findCenter(params);
    Eigen::Vector2d origin = Eigen::Vector2d::Zero(); // for the vectors

    // Iterate through all angles and rotate the param according to that angle
    for (double angle : angles) {
        Eigen::MatrixXd rotated_terminals (params.terminals.rows(), params.terminals.cols());
        Eigen::MatrixXd rotated_vectors (params.vectors.rows(), params.vectors.cols());
        rotated_terminals.setZero();
        rotated_vectors.setZero();

        // Iterate through all branches and rotate each terminal and vector of that branch
        for (int j = 0; j < params.numBranches; ++j) {
            rotated_terminals.block<2, 1>(2 * j, 0) =
                    rotatePoint(params.terminals.block<2, 1>(2 * j, 0), center, angle);
            rotated_terminals.block<2, 1>(2 * j, 1) =
                    rotatePoint(params.terminals.block<2, 1>(2 * j, 1), center, angle);
            rotated_terminals.block<2, 1>(2 * j, 2) =
                    rotatePoint(params.terminals.block<2, 1>(2 * j, 2), center, angle);

            rotated_vectors.block<2, 1>(2 * j, 0) =
                     rotatePoint(params.vectors.block<2, 1>(2 * j, 0), origin, angle);
            rotated_vectors.block<2, 1>(2 * j, 1) =
                    rotatePoint(params.vectors.block<2, 1>(2 * j, 1), origin, angle);
            rotated_vectors.block<2, 1>(2 * j, 2) =
                    rotatePoint(params.vectors.block<2, 1>(2 * j, 2), origin, angle);
        }

        // Add new parametrization to our vector
        rotated_data_set.emplace_back(params.numBranches, Eigen::MatrixXd(params.widths),
                                      rotated_terminals, rotated_vectors);
    }
    return rotated_data_set;
}

// This function has the exact same functionality as rotateParametrization but does it for type ParametrizationEntry,
// where the parametrization and the vector of displacements are rotated an equal amount, and the energy is kept
// the same. This helps for the NN to be rotation-invariant
ParametrizationDataSet DataOperations::rotateParametrizationEntry(ParametrizationEntry &param_entry,
        std::pair<double, double> &rotation_params, std::mt19937 &rng) {
    ParametrizationDataSet rotated_data_set;

    // Initialize angles vector and find the center point of the first parametrization
    std::vector<double> angles = generateRotationAngles(rotation_params.first, rotation_params.second, rng);
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
        for (int i = 0; i < param_entry.numBranches; ++i) {
            for (int j = 0; j < param_entry.displacements.cols(); ++j) {
                rotated_displacements.block<2, 1>(4 * i, j) =
                    rotatePoint(param_entry.displacements.block<2, 1>(4 * i, j), origin, angle);
                rotated_displacements.block<2, 1>(4 * i + 2, j) =
                    rotatePoint(param_entry.displacements.block<2, 1>(4 * i + 2, j), origin, angle);
            }
        }

        // Add new parametrization to our vector
        MeshParametrizationData new_param {param_entry.numBranches, Eigen::MatrixXd(param_entry.param.widths),
                                            rotated_terminals, rotated_vectors};

        rotated_data_set.emplace_back(param_entry.numBranches, std::move(new_param),
                                      std::move(rotated_displacements), param_entry.energyDifference);
    }
    return rotated_data_set;
}

// The purpose of this function is to take in a vector of side lengths, and return the midpoints of the sides of a
// convex polygon with the given side lengths, along with the corresponding outward pointing unit normal vector, and
// unit normal vectors between vertices, as well as the vertices of the corresponding polygon, this is useful for
// generating a multi-branch parametrization
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> DataOperations::generateMidPointsAndVectors(
    const Eigen::VectorXd &side_lengths) {

    // Declare return variables
    int n = side_lengths.size();
    Eigen::MatrixXd vertices(2, n);
    Eigen::MatrixXd outward_vectors(2, n);
    Eigen::MatrixXd midpoints(2, n);
    Eigen::MatrixXd side_vectors(2, n);

    // Place vertices in an initial circular shape corresponding to the side lengths
    double angle = 0;
    vertices.col(0) = Eigen::Vector2d(0, 0);
    for (int i = 1; i < n; ++i) {
        angle += 2 * M_PI / n;
        vertices.col(i) = vertices.col(i-1) +
            side_lengths(i-1) * Eigen::Vector2d(std::cos(angle), std::sin(angle));
    }

    // Iterative adjustment until all side lengths are exactly as we desired
    const int max_iterations = 1000;
    const double tolerance = 1e-6;
    for (int iter = 0; iter < max_iterations; ++iter) {
        bool converged = true;
        for (int i = 0; i < n; ++i) {
            int next = (i + 1) % n;
            Eigen::Vector2d edge = vertices.col(next) - vertices.col(i);
            double current_length = edge.norm();
            double target_length = side_lengths(i);

            // We adjust the vertices so that the side length is closer to what it should be
            if (std::abs(current_length - target_length) > tolerance) {
                converged = false;
                double scale = target_length / current_length;
                Eigen::Vector2d mid_point = 0.5 * (vertices.col(i) + vertices.col(next));
                vertices.col(i) = mid_point + 0.5 * scale * (vertices.col(i) - mid_point);
                vertices.col(next) = mid_point + 0.5 * scale * (vertices.col(next) - mid_point);
            }
        }
        if (converged) break;
    }

    // If the polygon doesn't match our initial side lengths, we re-scale
    double avg_scale = 0;
    for (int i = 0; i < n; ++i) {
        int next = (i + 1) % n;
        Eigen::Vector2d edge = vertices.col(next) - vertices.col(i);
        double current_length = edge.norm();
        avg_scale += side_lengths(i) / current_length;
    }
    avg_scale /= n;
    vertices *= avg_scale;

    // Center the polygon
    Eigen::Vector2d centroid = vertices.rowwise().mean();
    vertices.colwise() -= centroid;

    // Calculate outward vectors, midpoints, and side vectors, and return
    for (int i = 0; i < n; ++i) {
        int next = (i + 1) % n;
        Eigen::Vector2d edge = vertices.col(next) - vertices.col(i);
        midpoints.col(i) = (vertices.col(i) + vertices.col(next)) / 2.0;
        outward_vectors.col(i) = Eigen::Vector2d(edge.y(), -edge.x()).normalized();
        side_vectors.col(i) = edge.normalized();
    }
    return {midpoints, outward_vectors, side_vectors, vertices};
}

// This function takes in a single-branch parametrization and returns a matrix of displacement vectors to perturb it
// according to our perturbation parameters
Eigen::MatrixXd DataOperations::singleBranchDisplacements(DisplacementGenerators &param_gens) {
    // Declare matrix that will hold our displacement vectors
    Eigen::MatrixXd displacements(4, 2);
    displacements.setZero();

    // Iterate through the 4 corners of this branch and generate displacement vectors
    for (int i = 0; i < 4; ++i) {
        // Add a displacement boundary condition with probability displaceProbability for each coordinate
        if (param_gens.uniformPerturb(param_gens.rng) < param_gens.displacementProbability) {
            displacements((i % 2) * 2, i / 2) = param_gens.displacementPerturb(param_gens.rng)
                    * param_gens.maxDistances[0];
        }

        if (param_gens.uniformPerturb(param_gens.rng) < param_gens.displacementProbability) {
            displacements((i % 2) * 2 + 1, i / 2) = param_gens.displacementPerturb(param_gens.rng)
                    * param_gens.maxDistances[0];
        }
    }
    return displacements;
}

// This function takes in a multi-branch parametrization and returns a matrix of displacement vectors to perturb it
// according to our perturbation parameters
Eigen::MatrixXd DataOperations::multiBranchDisplacements(int num, DisplacementGenerators &param_gens) {
    LF_ASSERT_MSG(param_gens.maxDistances.size() == num, "Vector of maxDistances and num given to "
    "multiBranchDisplacements incorrect, num = " << num << ", and size is " << param_gens.maxDistances.size() << "\n");
    // Declare matrix that will hold our displacement vectors
    Eigen::MatrixXd displacements(num * 4, 1);
    displacements.setZero();

    // Iterate through all branches to add displacement boundary conditions one at a time
    for (int i = 0; i < num; ++i) {
        // Iterate through the 2 points for which we'll have displacement BCs (4 total with 2D x and y)
        for (int j = 0; j < 4; ++j) {
            // There can only be displacement BCs at the ends of each branch, the remaining sides have Neumann BCs
            // Add a displacement boundary condition with probability displaceProbability for each coordinate
            if (param_gens.uniformPerturb(param_gens.rng) < param_gens.displacementProbability) {
                displacements(4 * i + j, 0) += param_gens.displacementPerturb(param_gens.rng)
                                           * param_gens.maxDistances[i];
            }
        }
    }
    return displacements;
}

// This function takes in a base_parametrization and some perturbation parameters and returns a vector of perturbed
// parametrizations, with altered widths, terminals, and vector orientations
std::vector<MeshParametrizationData> DataOperations::generatePeturbedParametrizations(MeshParametrizationData &base_param,
        PerturbationParams &perturb_params) {

    std::vector<MeshParametrizationData> perturbed_vector;
    // If the current parametrization is invalid, we return the empty vector in order to generate a new one
    if (!MeshParametrization::meshParamValidator(base_param)) {
        return perturbed_vector;
    }
    perturbed_vector.emplace_back(base_param);

    std::uniform_real_distribution<double> uniform (-1, 1);

    // We perturb the parametrization for the 1-branch case
    if (base_param.numBranches == 1) {
        // Initialize the heuristic parameters to perturb this branch
        double approx_length = (base_param.terminals.col(1) - base_param.terminals.col(0)).norm() +
                                (base_param.terminals.col(2) - base_param.terminals.col(1)).norm();
        Eigen::Vector2d direction = (base_param.terminals.col(2) - base_param.terminals.col(0)).normalized();
        double start_dot = std::abs(base_param.vectors.col(0).dot(direction));
        double end_dot = std::abs(base_param.vectors.col(2).dot(direction));
        double curvature = (start_dot + end_dot) / 2.0;

        double length_factor = std::min(approx_length / perturb_params.vector_perturb.second, 2.0);
        double max_angle = perturb_params.vector_perturb.first * length_factor *
            (1 - curvature / 2.0);

        double max_width = std::min(perturb_params.width_perturb.first * base_param.widths.mean(),
            perturb_params.width_perturb.second * approx_length);

        double max_terminal = std::min(perturb_params.terminal_perturb.first * base_param.widths.mean(),
            perturb_params.terminal_perturb.second * approx_length);

        int count = 0;
        while (perturbed_vector.size() < perturb_params.numPerturbations) {
            MeshParametrizationData new_param {base_param.numBranches, base_param.widths,
                base_param.terminals, base_param.vectors};
            // Perturb each "parameter" with probability displaceProbability
            for (int i = 0; i < 3; ++i) {
                // perturb the width
                if (uniform(perturb_params.rng) <= perturb_params.perturbProbability) {
                    new_param.widths(i) += uniform(perturb_params.rng) * max_width;
                }

                // perturb the terminal
                if (uniform(perturb_params.rng) <= perturb_params.perturbProbability) {
                    new_param.terminals(0, i) += uniform(perturb_params.rng) * max_terminal;
                    new_param.terminals(1, i) += uniform(perturb_params.rng) * max_terminal;
                }

                // perturb the vector
                if (uniform(perturb_params.rng) <= perturb_params.perturbProbability) {
                    double angle = uniform(perturb_params.rng) * max_angle;
                    Eigen::Vector2d origin = Eigen::Vector2d::Zero();
                    new_param.vectors.col(i) = rotatePoint(new_param.vectors.col(i), origin, angle);
                }
            }

            if (MeshParametrization::meshParamValidator(new_param)) {
                perturbed_vector.emplace_back(new_param);
                count = 0;
                continue;
            }
            count++;
            if (count > 50) {
                return perturbed_vector;
            }
        }
        return perturbed_vector;
    }

    // Initialize vector of approximate lengths of branches
    std::vector<double> approx_lengths(base_param.numBranches);
    for (int i = 0; i < base_param.numBranches; i++) {
        approx_lengths[i] = (base_param.terminals.block<2, 1>(2 * i, 1) - base_param.terminals.block<2, 1>(2 * i, 0)).norm() +
                                (base_param.terminals.block<2, 1>(2 * i, 2) - base_param.terminals.block<2, 1>(2 * i, 1)).norm();
    }

    // Initialize vector of curvatures per branch
    std::vector<double> curvatures(base_param.numBranches);
    for (int i = 0; i < base_param.numBranches; i++) {
        Eigen::Vector2d direction = (base_param.terminals.block<2, 1>(2 * i, 2) -
            base_param.terminals.block<2, 1>(2 * i, 0)).normalized();
        double start_dot = std::abs(base_param.vectors.block<2, 1>(2 * i, 0).dot(direction));
        double end_dot = std::abs(base_param.vectors.block<2, 1>(2 * i, 2).dot(direction));
        curvatures[i] = (start_dot + end_dot) / 2.0;
    }

    // Initialize vector of maximum angle changes per branch
    std::vector<double> max_angles(base_param.numBranches);
    for (int i = 0; i < base_param.numBranches; i++) {
        double length_factor = std::min(approx_lengths[i] / perturb_params.vector_perturb.second, 2.0);
        max_angles[i] = perturb_params.vector_perturb.first * length_factor *
        (1 - curvatures[i] / 2.0);
    }

    // Initialize vector of maximum width changes per branch
    std::vector<double> max_widths(base_param.numBranches);
    for (int i = 0; i < base_param.numBranches; i++) {
        max_widths[i] = std::min(perturb_params.width_perturb.first * base_param.widths.row(i).mean(),
                perturb_params.width_perturb.second * approx_lengths[i]);
    }

    // Initialize vector of maximum terminal changes per branch
    std::vector<double> max_terminals(base_param.numBranches);
    for (int i = 0; i < base_param.numBranches; i++) {
        max_terminals[i] = std::min(perturb_params.terminal_perturb.first * base_param.widths.row(i).mean(),
            perturb_params.terminal_perturb.second * approx_lengths[i]);
    }

    int count = 0;
    // We perturb the parametization in the 3+ branch case, ensuring that the width, terminal, and vector positions
    // for the first "columns" always remain the same due to the constraints at the center
    while (perturbed_vector.size() < perturb_params.numPerturbations) {
        MeshParametrizationData new_param {base_param.numBranches, base_param.widths,
                        base_param.terminals, base_param.vectors};

        // Perturb each "parameter" with probability displaceProbability for each branch
        for (int i = 0; i < base_param.numBranches; ++i) {

            // Skip over the first column, due to geometric constraints on the center
            for (int j = 1; j < 3; ++j) {
                // perturb the width
                if (uniform(perturb_params.rng) <= perturb_params.perturbProbability) {
                    new_param.widths(i, j) += uniform(perturb_params.rng) * max_widths[i];
                }

                // perturb the terminal
                if (uniform(perturb_params.rng) <= perturb_params.perturbProbability) {
                    new_param.terminals(2 * i, j) += uniform(perturb_params.rng) * max_terminals[i];
                    new_param.terminals(2 * i + 1, j) += uniform(perturb_params.rng) * max_terminals[i];
                }

                // perturb the vector
                if (uniform(perturb_params.rng) <= perturb_params.perturbProbability) {
                    double angle = uniform(perturb_params.rng) * max_angles[i];
                    Eigen::Vector2d origin = Eigen::Vector2d::Zero();
                    new_param.vectors.block(2 * i, j, 2, 1) = rotatePoint(
                        new_param.vectors.block(2 * i, j, 2, 1), origin, angle);
                }
            }
        }

        if (MeshParametrization::meshParamValidator(new_param)) {
            perturbed_vector.emplace_back(new_param);
            count = 0;
            continue;
        }
        count++;
        if (count > 50) {
            return perturbed_vector;
        }
    }
    return perturbed_vector;
}

// The purpose of this function is to take in a parametrization, and return a ParametrizationDataset where all entries
// are in the linear elastic region of the material
ParametrizationDataSet DataOperations::generateDisplacementBCs(MeshParametrizationData &base_param,
                                                                       DisplacementParams &params) {
    // We check the given base parametrization to make sure it is valid
    if (!MeshParametrization::meshParamValidator(base_param)) {
        return {};
    }

    // Declare vector of max distances, which will be the variance when generating displacements
    std::vector<double> max_distances;
    ParametrizationDataSet displaced_entries;

    if (base_param.numBranches == 1) {
        // Use the following heuristic to determine the maximum allowable strain for calculating displacement vectors
        double max_stress = params.percentYieldStrength * params.yieldStrength;
        double max_strain = max_stress / params.modulusOfElasticity;

        // Obtain the std deviation we'll be using when rotating the vectors
        double approx_length = (base_param.terminals.col(1) - base_param.terminals.col(0)).norm() +
                (base_param.terminals.col(2) - base_param.terminals.col(1)).norm();
        double approx_width = base_param.widths.mean();
        max_distances.emplace_back(std::min(approx_width, approx_length) * max_strain);

        DisplacementGenerators perturb_params = {params.displacementProbability, max_distances, params.rng};

        calculationParams calc_energy_params {params.yieldStrength, params.modulusOfElasticity,
            params.poissonRatio,params.meshSize, params.order};

        int count = 0;
        // Keep adding data entries until we reach our desired size for this base_param
        while (displaced_entries.size() < params.numDisplacements) {
            auto displacements = singleBranchDisplacements(perturb_params);
            std::pair<bool, double> energy_check = MeshParametrization::displacementEnergy(base_param,
                displacements, calc_energy_params);

            // If we are still in the linear elastic region, we reset count and add to our dataset,
            // else we increment count
            if (energy_check.first) {
                count = 0;
                displaced_entries.emplace_back(1, base_param, displacements, energy_check.second);
            }
            else {
                count++;
                if (count >= 50) return displaced_entries;
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
            double approx_length = (base_param.terminals.block<2, 1>(2 * i, 1) - base_param.terminals.block<2, 1>(2 * i, 0)).norm() +
                                   (base_param.terminals.block<2, 1>(2 * i, 2) - base_param.terminals.block<2, 1>(2 * i, 1)).norm();
            double approx_width = base_param.widths.row(i).mean();
            max_distances.emplace_back(std::min(approx_length, approx_width) * max_strain);
        }

        DisplacementGenerators perturb_params = {params.displacementProbability, max_distances, params.rng};

        calculationParams calc_energy_params {params.yieldStrength, params.modulusOfElasticity,
            params.poissonRatio,params.meshSize, params.order};

        int count = 0;
        // Keep adding data entries until we reach our desired size for this base_param
        while (displaced_entries.size() < params.numDisplacements) {
            auto displacements = multiBranchDisplacements(base_param.numBranches, perturb_params);

            std::pair<bool, double> energy_check = MeshParametrization::displacementEnergy(base_param,
                displacements, calc_energy_params);

            // If we are still in the linear elastic region, we reset count and add to our dataset,
            // else we increment count
            if (energy_check.first) {
                count = 0;
                displaced_entries.emplace_back(base_param.numBranches, base_param, displacements, energy_check.second);
            }
            else {
                count++;
                if (count >= 50) return displaced_entries;
            }
        }
    }

    else {
        LF_ASSERT_MSG(false, "Wrong number of branches sent to generatePerturbedParametrizations, "
        << base_param.numBranches);
    }
    return displaced_entries;
}

// The purpose of this function is to take the number of branches, a width, and a length, and output a base
// parametrization to add to the dataset. The base parametrization is a straight horizontal branch in this
// case on the x-axis
MeshParametrizationData DataOperations::generateSingleBranchParametrization(int num, double width, double length) {
    LF_ASSERT_MSG(width > 1e-5 and length > 1e-5, "Width and length are not valid in "
        "generateSingleBranchParametrization, width: " << width << " and length: " << length);
    LF_ASSERT_MSG(num == 1, "Incorrect number of branches sent to generateSingleBranchParametrization, " << num);

    // Declare necessary matrices to initialize a parametrization
    Eigen::MatrixXd widths(1, 3);
    Eigen::MatrixXd terminals(2, 3);
    Eigen::MatrixXd vectors (2, 3);

    // Initialize necessary matrices with a non-varying width, length exactly as expected, and all vectors pointing up
    widths << width, width, width;
    terminals << -length / 2, 0.0, length / 2, 0.0, 0.0, 0.0;
    vectors << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;

    // Return the MeshParametrization Data we want
    return {num, widths, terminals, vectors};
}

// The purpose of this function is to take the number of branches, a vector of widths, a vector of lengths, and output
// a base multi-branch parametrization to add to the dataset. The branch will likely be modified afterwards. The base
// parametrization will be centered around the origin, with straight branches protruding out of it of constant width
MeshParametrizationData DataOperations::generateMultiBranchParametrization(int num, const Eigen::VectorXd &widths,
                                                                           const Eigen::VectorXd &lengths) {
    LF_ASSERT_MSG(widths.size() == lengths.size() and num == widths.size(), "Sizes of widths and lengths do not match in "
        "generateMultiBranchParametrization, widths size: " << widths.size() << ", lengths size: " << lengths.size());

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
    LF_ASSERT_MSG(false, "Incorrect number of branches sent to generateMultiBranchParametrization, " << num);
}

// The purpose of this function is to generate an entire data set full of the "MeshParametrizationData" type using the
// parameters given in params and the corresponding GenerationParams struct
ParametrizationDataSet DataOperations::generateParametrizationDataSet(GenerationParams &gen_params) {
    // Declare the dataset and reserve space for it
    ParametrizationDataSet data_set;
    data_set.reserve(gen_params.datasetSize);

    // Initialize necessary random number generators for width and length, and perturbation parameters
    std::uniform_real_distribution<> length_dist(gen_params.lengthInterval.first, gen_params.lengthInterval.second);
    std::uniform_real_distribution<> width_dist(gen_params.widthInterval.first, gen_params.widthInterval.second);
    DisplacementParams displacement_params {gen_params};
    PerturbationParams perturbation_params {gen_params};

    // This vector will hold the correct value of the tags throughout the loop
    std::vector<int> tagCounter(NUM_TAGS, -1);

    // Initialize counters so we don't run into any infinite loops
    int count_base = 0;
    int base_limit = 10;
    int count_displace = 0;
    int displace_limit = 10;

    // Iterate until we have our desired dataset size
    while (data_set.size() < gen_params.datasetSize) {

        MeshParametrizationData base_param;

        if (gen_params.numBranches == 1) {

            // Generate a base parametrization with the generated length and width
            double length = length_dist(gen_params.rng);
            double width = width_dist(gen_params.rng);
            base_param = generateSingleBranchParametrization(1, width, length);

        }

        else if (gen_params.numBranches >= 3) {

            // Initialize vector of widths and lengths for each branch
            Eigen::VectorXd lengths(gen_params.numBranches);
            Eigen::VectorXd widths(gen_params.numBranches);
            for (int i = 0; i < gen_params.numBranches; ++i) {
                lengths(i) = length_dist(gen_params.rng);
                widths(i) = width_dist(gen_params.rng);
            }
            // Generate base multi-branch parametrization
            base_param = generateMultiBranchParametrization(gen_params.numBranches, widths, lengths);

        }
        // With a base_param now initialized, we update the indices in tags
        tagCounter[TagIndex::BASE]++;
        tagCounter[TagIndex::BASE_PERTURBATION] = -1;

        // Generate perturbed versions of the base parametrization
        std::vector<MeshParametrizationData> perturbed_base = generatePeturbedParametrizations(
            base_param, perturbation_params);

        // If the returned vector is empty, we make sure not to increment the base tag
        if (perturbed_base.empty()) tagCounter[TagIndex::BASE]--;

        // Increment count of base if the desired size is not reached, otherwise reset it
        if (perturbed_base.size() < perturbation_params.numPerturbations) {
            count_base++;
            if (count_base >= base_limit) {
                std::cout << "The perturbation params are not ideal, dataset returned early \n";
                return data_set;
            }
        }
        else count_base = 0;

        // Iterate through the perturbed versions of base and create displacements of each, update tag
        for (MeshParametrizationData perturbed_base_param : perturbed_base) {
            tagCounter[TagIndex::BASE_PERTURBATION]++;
            tagCounter[TagIndex::DISPLACEMENT] = -1;

            // Provide displacement boundary conditions for each of these perturbed parametrizations
            ParametrizationDataSet displaced_perturbed_base = generateDisplacementBCs(
                perturbed_base_param, displacement_params);

            // Reset the tag counter if the returned dataset is empty
            if (displaced_perturbed_base.empty()) tagCounter[TagIndex::BASE_PERTURBATION]--;

            // Increment count of base if the desired size is not reached, otherwise reset it
            if (displaced_perturbed_base.size() < displacement_params.numDisplacements) {
                count_displace++;
                if (count_displace >= displace_limit) {
                    std::cout << "The displacement params are not ideal, dataset returned early \n";
                    return data_set;
                }
            }
            else count_displace = 0;

            // Iterate through each displaced and perturbed entry and rotate, update tags as well
            for (ParametrizationEntry disp_perturbed_base : displaced_perturbed_base) {
                tagCounter[TagIndex::DISPLACEMENT]++;
                tagCounter[TagIndex::ROTATION] = -1;
                ParametrizationDataSet rotated = rotateParametrizationEntry(
                    disp_perturbed_base, gen_params.dataRotationParams, gen_params.rng);

                // Iterate through each rotated, displaced, and perturbed entry, flip them according to flip params
                for (ParametrizationEntry rot_perturb : rotated) {
                    tagCounter[TagIndex::ROTATION]++;
                    ParametrizationDataSet flip_perturb_rot = flipVectorEntry(
                        rot_perturb, gen_params.flipParams, gen_params.rng);

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
    if (data_set.size() > gen_params.datasetSize) {
        data_set.erase(data_set.begin() + gen_params.datasetSize, data_set.end());
    }
    return data_set;
}

// This function takes in a parametrization data set and converts it into a point data set using the functionality
// in polynomialPoints. Since the point parametrization doesn't depend on vector orientations, if the vector flip tag
// is a tag we've already seen, we skip over it rather than add the same entry twice
PointDataSet DataOperations::parametrizationToPoint(ParametrizationDataSet &param_dataset) {
    PointDataSet point_data_set;
    point_data_set.reserve(param_dataset.size());

    std::unordered_map<std::vector<int>, bool, VectorHash> seen_tags;

    // Iterate through all entries in the initial dataset
    for (ParametrizationEntry param_entry: param_dataset) {

        // If we can't find this tag within our set, we can add this entry, since it is "unique"
        if (!seen_tags.contains(param_entry.tags)) {

            // Use polynomialPoints to convert each parametrization correctly then add to the data set
            ParametrizationPoints points{param_entry.numBranches,
                                          MeshParametrization::polynomialPoints(param_entry.param)};
            point_data_set.emplace_back(param_entry.numBranches, points,
                                  param_entry.displacements, param_entry.energyDifference, param_entry.tags);

            // Add this tag to our set of seen tags since this is new
            seen_tags[param_entry.tags] = true;
        }
    }
    return point_data_set;
}

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

    return {numBranches, widths, terminals, vectors};
}

// This function takes in a file and ParametrizationPoints object and saves the object to the file
void DataOperations::saveParametrizationPoints(std::ofstream &file, const ParametrizationPoints &points) {
    file.write(reinterpret_cast<const char*>(&points.numBranches), sizeof(int));

    // Save points, current shape is (4 * numBranches) by 3
    file.write(reinterpret_cast<const char*>(points.points.data()), 12 * points.numBranches * sizeof(double));
}

// This function takes in a file we're reading to return a ParametrizationPoints object
ParametrizationPoints DataOperations::loadParametrizationPoints(std::ifstream &file) {
    int numBranches;
    file.read(reinterpret_cast<char*>(&numBranches), sizeof(int));

    // Load the points into the matrix
    Eigen::MatrixXd points(4 * numBranches, 3);
    file.read(reinterpret_cast<char*>(points.data()), 12 * numBranches * sizeof(double));

    return {numBranches, std::move(points)};
}

// This function takes in a DataSet, identifies whether it's filled with Parametrization entries or Point entries,
// and saves the corresponding structure into a binary file for later use with a given file name
void DataOperations::saveDataSet(const DataSet &data_set, const std::string &file_name) {
    // Find the path to where we want to save the data set
    std::filesystem::path project_root = std::filesystem::current_path().parent_path();
    std::filesystem::path data_dir = project_root / "data";
    std::filesystem::path full_path = data_dir / (file_name + ".bin");

    // Declare the file
    std::ofstream file(full_path, std::ios::binary);
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

            file.write(reinterpret_cast<const char *>(entry.displacements.data()), entry.displacements.size() * sizeof(double));

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
            file.write(reinterpret_cast<const char *>(entry.displacements.data()), num_displacements * sizeof(double));

            file.write(reinterpret_cast<const char*>(&entry.energyDifference), sizeof(double));
            file.write(reinterpret_cast<const char*>(entry.tags.data()), NUM_TAGS * sizeof(int));
        }
    }
}

// This function is given a file name, identifies whether it contains Parametrization entries or Point entires,
// and returns the corresponding dataset
DataSet DataOperations::loadDataSet(const std::string &file_name) {
    // Find the path to where we want to save the data set
    std::filesystem::path project_root = std::filesystem::current_path().parent_path();
    std::filesystem::path data_dir = project_root / "data";
    std::filesystem::path full_path = data_dir / (file_name + ".bin");

    std::ifstream file(full_path, std::ios::binary);
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
            if (!file.read(reinterpret_cast<char*>(&num_branches), sizeof(int)))
                throw std::runtime_error("Failed to read num_branches");

            MeshParametrizationData param = loadMeshParametrizationData(file);

            int num_cols = num_branches == 1 ? 2 : 1;
            Eigen::MatrixXd displacements(4 * num_branches, num_cols);
            if (!file.read(reinterpret_cast<char*>(displacements.data()), 4 * num_branches * num_cols * sizeof(double))) {
                throw std::runtime_error("Failed to read displacement data");
            };

            double energy_diff;
            if (!file.read(reinterpret_cast<char*>(&energy_diff), sizeof(double)))
                throw std::runtime_error("Failed to read energy_diff");
            ParametrizationEntry entry(num_branches, std::move(param), std::move(displacements), energy_diff);

            if (!file.read(reinterpret_cast<char*>(entry.tags.data()), NUM_TAGS * sizeof(int)))
                throw std::runtime_error("Failed to read tags");

            data_set.push_back(entry);
        }
        return data_set;
    }

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
        file.read(reinterpret_cast<char*>(displacements.data()), 4 * num_branches * num_cols * sizeof(double));

        double energy_diff;
        file.read(reinterpret_cast<char*>(&energy_diff), sizeof(double));

        std::vector<int> tags(NUM_TAGS);
        file.read(reinterpret_cast<char*>(tags.data()), NUM_TAGS * sizeof(int));

        PointEntry entry(num_branches, std::move(points), std::move(displacements), energy_diff, tags);

        data_set.push_back(std::move(entry));
    }
    return data_set;
}

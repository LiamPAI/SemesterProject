## @package calculate_displacements
#  Functions for optimizing displacement vectors in metal foam structures
#
#  This module provides functionality for optimizing displacement vectors to minimize
#  the total energy in a metal foam toy structure, subject to compatibility and boundary conditions.

import os.path
import random
import sys

from tornado.options import options

sys.path.append("/Users/liamcurtis/Documents/ETH Classes/Winter 2024/Semester Project/Sem_Project_LC/cmake-build-debug/python")
import metal_foams

from neural_network import (
    NeuralNetwork,
    ParamPointDataset,
    NeuralNetworkTrainer,
    DataType,
    TagIndex,
    normalize_input,
    denormalize_input,
    use_model_for_inference,
    generate_dataset,
    load_dataset,
    save_dataset,
    print_dataset,
    parametrization_to_point
)
import numpy as np
import torch
from scipy.optimize import minimize, Bounds
from scipy.stats import pearsonr
import copy

## @brief Flatten a MeshParametrizationData object into a 1D array
#  @param param Parametrization object to flatten
#  @return Flattened array containing widths, terminals, and vectors
def flatten_parametrization(param):
    flattened_param = np.concatenate([
        param.widths.flatten(),
        param.terminals.flatten(),
        param.vectors.flatten()
    ])
    return flattened_param

## @brief Flatten a ParametrizationPoints object into a 1D array
#  @param points Point object to flatten
#  @return Flattened array of points
def flatten_point(points):
    return points.points.flatten()

## @brief Rescale the optimization problem for better numerical stability
#  @param models_dict Dictionary of neural network models
#  @param parametrizations List of parametrizations
#  @param initial_displacement_vectors Initial displacement vectors
#  @param scale_factor Factor to scale displacements (default: 1000)
#  @return Tuple of (scaled_models_dict, scaled_initial_displacement_vectors)
def rescale_problem(models_dict, parametrizations, initial_displacement_vectors, scale_factor=1000):
    """
    Rescale the optimization problem with respect to displacement vectors to improve numerical stability

    :param models_dict: Dictionary of neural network models
    :param parametrizations: List of parametrizations
    :param initial_displacement_vectors: List of initial displacement vectors
    :param scale_factor: Factor to scale the displacements by (default 1000)
    :return: Tuple of (scaled_models_dict, scaled_initial_displacement_vectors)
    """
    scaled_models_dict = {}
    for num_branches, model in models_dict.items():
        scaled_model = copy.deepcopy(model)
        scaled_model.displacement_factor *= scale_factor
        scaled_models_dict[num_branches] = scaled_model

    scaled_displacement_vectors = [disp * scale_factor for disp in initial_displacement_vectors]
    return scaled_models_dict, scaled_displacement_vectors


# TODO: Come up with the "design" of what to do next in graph_mesh so that using this function is relatively scalable
#   e.g. do I need to create a function that generates datasets quite quickly?
#   How can I compare results from this to the FEM results
#   How to make sure that this function is easily usable, and perhaps, outputs values intermittently?

## @brief Optimize displacement vectors to minimize total energy of an overall mesh
#  @param models_dict Dictionary mapping number of branches to corresponding neural network models
#  @param parametrizations List of parametrizations or points
#  @param initial_displacement_vectors List of initial displacement vectors
#  @param compatibility_conditions List of compatibility conditions between parts
#  @param fixed_displacements List of fixed displacement boundary conditions
#  @param data_type Type of data (PARAMETRIZATION or POINT)
#  @param scale_factor Factor for scaling displacements (default: 1000)
#  @return Tuple of (parametrizations, optimized vectors, energies)
def optimize_displacement_vectors(models_dict, parametrizations, initial_displacement_vectors,
                                  compatibility_conditions, fixed_displacements, data_type, scale_factor=1000):
    """
    Optimize displacements vectors across all the parametrizations of a mesh to minimize the total energy given
    fixed displacement boundary conditions

    :param models_dict: Dictionary mapping number of branches to corresponding NeuralNetwork model
    :param parametrizations: List of all parametrizations or points involved in the calculation
    :param initial_displacement_vectors: List of initial displacement vectors for each parametrization (usually set to 0)
    :param compatibility_conditions: List of tuples (param1_idx, param2_idx), (branch1_idx, branch2_idx), (side1_idx, side2_idx)
    :param fixed_displacements: List of {(param_idx, side_idx): 4D vector of displacements}
    :param data_type: PARAMETRIZATION OR POINT
    :return: Parametrizations, the optimized displacement vectors, and the corresponding energy
    """

    # Calculate the total energy change of the mesh
    def calculate_energy(x):
        displacement_vectors = []
        start_e = 0
        for init_disp_e in initial_displacement_vectors:
            end_e = start_e + np.prod(init_disp_e.shape)
            displacement_vectors.append(x[start_e:end_e].reshape(init_disp_e.shape) / scale_factor)
            start_e = end_e

        energy_e = 0
        for param_e, disp in zip(parametrizations, displacement_vectors):
            if data_type == DataType.PARAMETRIZATION:
                num_branches_e = param_e.numBranches
                flattened_param_e = flatten_parametrization(param_e)
            else:
                num_branches_e = disp.shape[1] / 4 if disp.shape[1] == 1 else 1
                flattened_param_e = flatten_point(param_e)

            model_e = models_dict[num_branches_e]

            # print(f"Value of displacement vectors before normalization is \n{disp}")

            normalized_disp_e = disp.flatten() / model_e.displacement_factor

            # print(f"Value of displacement vectors after normalization is \n{normalized_disp_e}")

            input_data_e = np.concatenate([flattened_param_e, normalized_disp_e])
            normalized_energy_e = model_e(torch.tensor(input_data_e, dtype=torch.float32)).item()

            # print(f"Value of normalized energy is {normalized_energy_e}")

            energy_e += normalized_energy_e / model_e.energy_scale * (model_e.energy_max - model_e.energy_min) + model_e.energy_min

            # print(f"Value of de-normalized energy is {normalized_energy_e / model_e.energy_scale * (model_e.energy_max - model_e.energy_min) + model_e.energy_min}")
            # print(f"Value of total energy is {energy_e}")

        return energy_e

    def constraint_function(x):
        displacement_vectors = []
        start_c = 0
        for init_disp_c in initial_displacement_vectors:
            end_c = start_c + np.prod(init_disp_c.shape)
            displacement_vectors.append(x[start_c:end_c].reshape(init_disp_c.shape) / scale_factor)
            start_c = end_c

        constraints_c = []

        # Compatibility Conditions
        # TODO: Make sure the indexing works out here like I want it, it should, and also ensure whether is better to be in vector or scalar format here
        for condition in compatibility_conditions:
            param1_idx, param2_idx = condition.indices
            branch1_idx, side1_idx = condition.firstLocation
            branch2_idx, side2_idx = condition.secondLocation

            diff = (displacement_vectors[param1_idx][(4 * branch1_idx):(4 * branch1_idx + 4)][:, side1_idx]
                    - displacement_vectors[param2_idx][(4 * branch2_idx):(4 * branch2_idx + 4)][:, side2_idx])
            constraints_c.extend(diff)

        # Fixed displacement boundary conditions
        # TODO: Make sure the indexing works out here like I want it, it should, and also ensure whether is better to be in vector or scalar format here
        for condition in fixed_displacements:
            param_idx, side_idx = condition.indices
            fixed_value = condition.displacements
            diff = displacement_vectors[param_idx][:, side_idx] - fixed_value
            constraints_c.extend(diff)

        return np.array(constraints_c)

    def callback(xk, res=None):
        if res is None:
            # This is for methods that don't pass OptimizeResult
            energy_callback = calculate_energy(xk)
            print(f"Current energy: {energy_callback}")
        else:
            # This is for methods like 'trust-constr' that pass OptimizeResult
            print(f"Current energy: {res.fun}")
        return False  # Returning False allows the optimization to continue


    x0 = np.concatenate([disp.flatten() * scale_factor for disp in initial_displacement_vectors])
    x0 += np.random.normal(0, 1e-3, x0.shape)
    constraints = {'type' : 'eq', 'fun': constraint_function}
    bounds = Bounds(-3, 3)

    result = minimize(calculate_energy, x0, method='trust-constr', bounds=bounds ,constraints=constraints,
                      options={'maxiter': 1000,
                               'verbose': 2,
                               'xtol': 1e-6,
                               'gtol': 1e-6,
                               'barrier_tol':1e-6},
                      callback=callback)

    if result.success:
        optimized_vectors = []
        start = 0
        for init_disp in initial_displacement_vectors:
            end = start + np.prod(init_disp.shape)
            optimized_vectors.append(result.x[start:end].reshape(init_disp.shape) / scale_factor)
            start = end

        # Post-processing to have nice return values for easy comparison to FEM results
        energies = []
        for param, disp in zip(parametrizations, optimized_vectors):
            if data_type == DataType.PARAMETRIZATION:
                num_branches = param.numBranches
                flattened_param = flatten_parametrization(param)
            else:
                num_branches = disp.shape[1] / 4 if disp.shape[1] == 1 else 1
                flattened_param = flatten_point(param)

            model = models_dict[num_branches]

            normalized_disp = disp.flatten() / model.displacement_factor
            input_data = np.concatenate([flattened_param, normalized_disp])
            normalized_energy = model(torch.tensor(input_data, dtype=torch.float32)).item()
            energy = normalized_energy / model.energy_scale * (model.energy_max - model.energy_min) + model.energy_min
            energies.append(energy)

        return parametrizations, optimized_vectors, energies
    else:
        raise ValueError("Optimization failed: " + result.message)

## @brief Analyze optimization results by comparing with FEM results
#  @param optimized_vectors List of optimized displacement vectors
#  @param optimized_energies List of optimized energies
#  @param true_vectors List of true displacement vectors from FEM
#  @param true_energies List of true energies from FEM
#  @return Dictionary containing error metrics and correlations
def analyze_optimization_results (optimized_vectors, optimized_energies, true_vectors, true_energies):
    """
    Analyze the results of the NN optimization by comparing with the "true" values from the finite element calculation

    :param optimized_vectors: List of NN-optimized displacement vectors in matrix form
    :param optimized_energies: List of NN-optimized energies for each parametrization
    :param true_vectors: List of the FEM-obtained displacement vectors
    :param true_energies: List of the FEM-obtained energy values
    """
    def calculate_vector_errors(opt_vecs, true_vecs):
        opt_flat = np.concatenate([o.flatten() for o in opt_vecs])
        true_flat = np.concatenate([t.flatten() for t in true_vecs])

        mse = np.mean((opt_flat - true_flat)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(opt_flat - true_flat))
        max_error = np.max(np.abs(opt_flat - true_flat))
        normalized_rmse = rmse / np.mean(np.abs(true_flat))
        mean_relative_error = np.mean(np.abs((opt_flat - true_flat) / true_flat))
        max_relative_error = np.max(np.abs((opt_flat - true_flat) / true_flat))

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Max_Error': max_error,
            'Normalized_RMSE': normalized_rmse,
            'Mean_Relative_Error': mean_relative_error,
            'Max_Relative_Error': max_relative_error
        }

    def calculate_energy_errors(opt_energies, true_energies):
        mse = np.mean((np.array(opt_energies) - np.array(true_energies))**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(opt_energies) - np.array(true_energies)))
        max_error = np.max(np.abs(np.array(opt_energies) - np.array(true_energies)))
        normalized_rmse = rmse / np.mean(np.abs(true_energies))
        mean_relative_error = np.mean(np.abs((np.array(opt_energies) - np.array(true_energies)) / np.array(true_energies)))
        max_relative_error = np.max(np.abs((np.array(opt_energies) - np.array(true_energies)) / np.array(true_energies)))

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Max_Error': max_error,
            'Normalized_RMSE': normalized_rmse,
            'Mean_Relative_Error': mean_relative_error,
            'Max_Relative_Error': max_relative_error
        }

    vector_errors = calculate_vector_errors(optimized_vectors, true_vectors)
    energy_errors = calculate_energy_errors(optimized_energies, true_energies)

    # Calculate correlation coefficients between optimized and "true" results
    vector_correlation = pearsonr(np.concatenate([v.flatten() for v in optimized_vectors]),
                                  np.concatenate([v.flatten() for v in true_vectors]))[0]
    energy_correlation = pearsonr(optimized_energies, true_energies)[0]

    return {
        'Displacement_Errors': vector_errors,
        'Energy_Errors' : energy_errors,
        'Vector_Correlation' : vector_correlation,
        'Energy_Correlation' : energy_correlation
    }

# TODO: Decide if I actually want this function in the first place

## @brief Print analysis results in a formatted manner
#  @param results Dictionary containing analysis results
def print_result_analysis(results):
    print("Displacement Errors:")
    for metric, value in results['Displacement_Errors'].items():
        print(f"  {metric}: {value}")

    print("\nEnergy Errors:")
    for metric, value in results['Energy_Errors'].items():
        print(f"  {metric}: {value}")

    print(f"\nVector Correlation: {results['Vector_Correlation']}")
    print(f"Energy Correlation: {results['Energy_Correlation']}")

## @brief Calculate energy for each parametrization-displacement pair
#  @param models_dict Dictionary of neural network models
#  @param parametrizations List of parametrizations
#  @param displacement_vectors List of displacement vectors
#  @param data_type Type of data (PARAMETRIZATION or POINT)
#  @return Tuple of (individual energies, total energy)
def calculate_energy_test(models_dict, parametrizations, displacement_vectors, data_type):
    """
    Calculate the energy for each parametrization-displacement pair and the total energy.

    :param models_dict: Dictionary mapping number of branches to corresponding NeuralNetwork model
    :param parametrizations: List of all parametrizations or points involved in the calculation
    :param displacement_vectors: List of displacement vectors for each parametrization
    :param data_type: PARAMETRIZATION or POINT
    :return: Tuple of (list of individual energies, total energy)
    """
    individual_energies = []
    total_energy = 0

    for param, disp in zip(parametrizations, displacement_vectors):
        if data_type == DataType.PARAMETRIZATION:
            num_branches = param.numBranches
            flattened_param = flatten_parametrization(param)
        else:
            num_branches = param.points.shape[0] // 4
            flattened_param = flatten_point(param)

        model = models_dict[num_branches]

        normalized_disp = disp.flatten() / model.displacement_factor

        input_data = np.concatenate([flattened_param, normalized_disp])
        normalized_energy = model(torch.tensor(input_data, dtype=torch.float32)).item()

        energy = normalized_energy / model.energy_scale * (model.energy_max - model.energy_min) + model.energy_min
        individual_energies.append(energy)
        total_energy += energy

    return individual_energies, total_energy

## @brief Calculate energy for point-based representations
#  @param models_dict Dictionary of neural network models
#  @param points List of point objects
#  @param displacement_vectors List of displacement vectors
#  @return Tuple of (individual energies, total energy)
def calculate_energy_results(models_dict, params, displacement_vectors, data_type):
    """
    Calculate the energy for each parametrization-displacement pair and the total energy.

    :param models_dict: Dictionary mapping number of branches to corresponding NeuralNetwork model
    :param params: List of all parametrizations or points involved in the calculation
    :param displacement_vectors: List of displacement vectors for each parametrization
    :param data_type: PARAMETRIZATION or POINT
    :return: Tuple of (list of individual energies, total energy)
    """
    individual_energies = []
    total_energy = 0

    for param, disp in zip(params, displacement_vectors):
        if data_type == DataType.PARAMETRIZATION:
            num_branches = param.numBranches
            flattened_param = flatten_parametrization(param)
        else:
            num_branches = param.points.shape[0] // 4
            flattened_param = flatten_point(param)

        model = models_dict[num_branches]

        normalized_disp = disp.flatten() / model.displacement_factor
        print(f"Value of the flattened param is: {flattened_param}")
        print(f"Value of the normalized displacements are: {normalized_disp}")

        input_data = np.concatenate([flattened_param, normalized_disp])
        normalized_energy = model(torch.tensor(input_data, dtype=torch.float32)).item()

        energy = normalized_energy / model.energy_scale * (model.energy_max - model.energy_min) + model.energy_min
        individual_energies.append(energy)
        total_energy += energy

    return individual_energies, total_energy













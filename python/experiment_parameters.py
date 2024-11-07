## @package experiment_parameters
#  Functions for hyperparameter search and experiment configurations
#
#  This module provides functions for conducting hyperparameter searches and
#  defining specific experimental configurations for the toy metal foams contained in testNE1.geo and testNE3.geo
#  It includes utilities for training models with different parameters
#  and generating datasets with various configurations.

import os.path
import sys
from itertools import combinations, product

from scipy.constants import value

from python.neural_network import print_dataset

sys.path.append("/Users/liamcurtis/Documents/ETH Classes/Winter 2024/Semester Project/Sem_Project_LC/cmake-build-debug/python")
import metal_foams
import pandas as pd
from ast import literal_eval
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Any
from torch import nn
from neural_network import (generate_dataset, save_dataset, load_dataset, DataType, NeuralNetwork,
                            NeuralNetworkTrainer, print_dataset, parametrization_to_point)
import csv

# This file will contain a ton of basic functions that return generation and training parameters for the NN depending
# on the experiment number

## @brief Perform hyperparameter search using either random or combinatorial search
#  @param cc_dataset The dataset to train on
#  @param dataset_type Type of dataset (parametrization or point)
#  @param search_space Dictionary of hyperparameters and their possible values
#  @param search_type Either 'random' or 'combinatorial'
#  @param num_trials Number of trials for random search (ignored for combinatorial)
#  @param csv_file Optional CSV file name to save results
#  @return Tuple of (best_model, best_params, best_performance, all_results)
def hyperparameter_search(cc_dataset, dataset_type, search_space: Dict[str, List[Any]],
                          search_type: str = 'random', num_trials:int = 10, csv_file: str = None):
    """
    Perform hyperparameter search using either random or combinatorial search

    :param cc_dataset: The dataset to train on
    :param dataset_type: Type of dataset (parametrization or point)
    :param search_space: Dictionary of hyperparameters and their possible values
    :param search_type: Either 'random' or 'combinatorial'
    :param num_trials: Number of trials for random search (ignored for combinatorial)
    :param csv_file: Optional CSV file name to save results, a file is created is this is not provided
    :return:
    """

    best_model = None
    best_performance = float('inf')
    best_params = None
    all_results = []

    training_data_dir = os.path.join(os.getcwd(), "training_data")
    os.makedirs(training_data_dir, exist_ok=True)

    # If no CSV file is specified, create a new one with a timestamp
    if csv_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(training_data_dir, f"hyperparameter_{timestamp}.csv")
    else:
        csv_file = os.path.join(training_data_dir, csv_file)

    # Check if the file already exists
    file_exists = os.path.isfile(csv_file)

    # Depending on the search type, we either get all possible combinations, or randomly select a few
    if search_type.lower() == 'combinatorial':
        keys, values = zip(*search_space.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        print(f"Total combinations to test: {len(combinations)}")
        trials = combinations
    elif search_type.lower() == 'random':
        trials = []
        for i in range(num_trials):
            hyperparams = {key: random.choice(value) for key, value in search_space.items()}
            trials.append(hyperparams)
    else:
        raise ValueError("search_type must be either 'random' or 'combinatorial'")

    # Run all of the trials
    for i, hyperparams in enumerate(trials):
        print(f"Running trial {i + 1}/{len(trials)}")

        print("Selected hyperparameters:")
        for k, v in hyperparams.items():
            print(f"  {k}: {v}")

        model, performance_metrics = train_and_save_model(cc_dataset, dataset_type, hyperparams)

        current_performance = performance_metrics['overall_avg_loss']
        print(f"Performance: {current_performance:.6f}")

        # Add to the results dictionary
        result = {**hyperparams, 'performance': current_performance}
        if isinstance(hyperparams['generation_params'], metal_foams.GenerationParams):
            result['generation_params'] = hyperparams['generation_params'].to_dict()
        else:
            result['generation_params'] = str(hyperparams['generation_params'])
        all_results.append(result)

        if current_performance < best_performance:
            best_performance = current_performance
            best_model = model
            best_params = hyperparams

        print("\n" + "="*50 + "\n")

    print("\nHyperparameter Search Results:")
    for i, result in enumerate(all_results):
        print(f"\nTrial {i+1}:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    print("\nBest Hyperparameters:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print(f"Best Performance: {best_performance:.6f}")

    column_order = [
        'generation_params',  # Generation params first
        # Training parameters
        'batch_size',
        'learning_rate',
        'optimizer',
        'scheduler',
        'criterion',
        'epochs',
        'split_ratios',
        # Architecture details last
        'num_branches',
        'hidden_layer_sizes',
        'activation_fn',
        'output_activation_fn',
        'performance',        # Performance metrics
    ]

    mode = 'a' if file_exists else 'w'
    with open(csv_file, mode, newline='') as csvfile:
        all_fields = set(all_results[0].keys())
        fieldnames = [col for col in column_order if col in all_fields]
        remaining_fields = sorted(list(all_fields - set(fieldnames)))
        fieldnames.extend(remaining_fields)

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        for result in all_results:
            row = {}
            for k in fieldnames:  # Iterate through fields in order
                v = result[k]
                if isinstance(v, (int, float, str)):
                    row[k] = v
                elif k == 'generation_params':
                    row[k] = str(v)  # v is already a dict from .to_dict()
                elif k == 'hidden_layer_sizes':
                    row[k] = str(v)
                elif k == 'split_ratios':
                    row[k] = str(v)
                elif k in ['activation_fn', 'output_activation_fn']:
                    row[k] = v.__name__
                else:
                    row[k] = str(v)
            writer.writerow(row)

    print(f"\nResults {'appended to' if file_exists else 'saved in'} {csv_file}")

    return best_model, best_params, best_performance, all_results

## @brief Train a neural network with given parameters and save it
#  @param cc_dataset The dataset to be trained on
#  @param dataset_type Type of the dataset (PARAMETRIZATION OR POINT)
#  @param params Dictionary containing the model training parameters
#  @param filename The model will be saved with this filename
#  @return Tuple of (trained_model, performance_metrics)
def train_and_save_model(cc_dataset, dataset_type, params, filename=None):
    """
    Train a neural network with the given training parameters, and save it

    :param cc_dataset: The dataset to be trained on
    :param dataset_type: Type of the dataset (PARAMETRIZATION OR POINT)
    :param params: Dictionary containing the model training parameters
    :param filename: The model will be saved with this filename
    :return: The trained model and its corresponding performance metrics
    """
    print("Training model with the following parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Check if generation_params is already a GenerationParams object
    if isinstance(params['generation_params'], metal_foams.GenerationParams):
        generation_params = params['generation_params']
    else:
        # If it's a dict, create a GenerationParams object from it
        generation_params = metal_foams.GenerationParams.from_dict(params['generation_params'])

    model = NeuralNetwork(
        num_branches=params['num_branches'],
        data_type=dataset_type,
        hidden_layer_sizes=params['hidden_layer_sizes'],
        generation_params=generation_params,
        activation_fn=params['activation_fn'],
        output_activation_fn=params['output_activation_fn']
    )

    trainer = NeuralNetworkTrainer(
        NN_model=model,
        cc_dataset=cc_dataset,
        dataset_type=dataset_type,
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        split_ratios=params['split_ratios'],
        optimizer_name=params['optimizer'],
        scheduler_name=params['scheduler'],
        criterion_name=params['criterion']
    )

    print("\nStarting training...")
    trainer.train(epochs=params['epochs'])

    performance_metrics = trainer.analyze_performance()
    print("\nTraining completed. Performance metrics:")
    for metric, value in performance_metrics.items():
        print(f"  {metric}: {value}")

    model.set_training_params(params)

    if filename:
        params_to_save = params.copy()
        if isinstance(params['generation_params'], metal_foams.GenerationParams):
            params_to_save['generation_params'] = params['generation_params'].to_dict()
        else:
            params_to_save['generation_params'] = params['generation_params']
        model.set_training_params(params_to_save)
        model.save(filename)
        print(f"\nModel saved as {filename}")

    return model, performance_metrics

## @brief Perform hyperparameter search across multiple datasets
#  @param datasets_dict Dictionary of datasets with their types
#  @param search_space Dictionary of hyperparameters and their possible values
#  @param search_type Either 'random' or 'combinatorial'
#  @param num_trials Number of trials
#  @return Dictionary containing results for each dataset
def conduct_hyperparameter_search(datasets_dict, search_space: Dict[str, List[Any]],
                                  search_type: str ='combinatorial', num_trials: int = 10):
    """
    Perform usually a combinatorial search across the multiple datasets provided

    :param datasets_dict: Dictionary of datasets with their types
    :param search_space: Dictionary of hyperparameters and their possible values
    :param search_type: Either 'random' or 'combinatorial'
    :param num_trials: Number of trials
    :return: Dictionary containing results for each dataset
    """
    results = {}

    for dataset_name, dataset_info in datasets_dict.items():
        print(f"\nProcessing dataset: {dataset_name}")

        # Load the dataset
        cc_dataset = metal_foams.loadDataSet(dataset_info['filename'])

        # Create the dataset_specific CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"hyperparameter_{dataset_name}_{timestamp}.csv"

        # Modify number of branches in search space
        dataset_search_space = search_space.copy()
        dataset_search_space['num_branches'] = [dataset_info['num_branches']]

        # Run the hyperparameter search for this dataset
        _, _, _, _ = hyperparameter_search(
            cc_dataset=cc_dataset,
            dataset_type=dataset_info['type'],
            search_space=dataset_search_space,
            search_type=search_type,
            num_trials=num_trials,
            csv_file=csv_file
        )

        results[dataset_name] = csv_file

    return results

def analyze_hyperparameter_search(csv_name):
    """
    Analyze hyperparameter search results from a CSV file.

    Parameters:
    csv_path (str): Path to the CSV file containing hyperparameter search results

    Returns:
    dict: Dictionary containing various analysis metrics and summaries
    """
    # Construct path to CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'training_data', csv_name)

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Clean up activation function names
    df['activation_fn'] = df['activation_fn'].str.replace('nn.', '')
    df['output_activation_fn'] = df['output_activation_fn'].str.replace('nn.', '')

    # Clean up hidden_layer_sizes strings
    df['hidden_layer_sizes'] = df['hidden_layer_sizes'].str.replace("'", "")

    # Best Configuration Analysis
    best_config = df.loc[df['performance'].idxmin()]
    worst_config = df.loc[df['performance'].idxmax()]

    # Architecture Analysis
    arch_stats = df.groupby('hidden_layer_sizes').agg({
        'performance': ['mean', 'std', 'min', 'max', 'count']
    })

    # Learning Rate Analysis
    lr_stats = df.groupby('learning_rate').agg({
        'performance': ['mean', 'std', 'min', 'max', 'count']
    })

    # Activation Function Analysis
    df['activation_combo'] = df['activation_fn'] + ' + ' + df['output_activation_fn']
    activation_stats = df.groupby('activation_combo').agg({
        'performance': ['mean', 'std', 'min', 'max', 'count']
    })

    # Loss Function Analysis
    criterion_stats = df.groupby('criterion').agg({
        'performance': ['mean', 'std', 'min', 'max', 'count']
    })

    # Cross Analysis: Learning Rate x Architecture
    lr_arch_stats = df.pivot_table(
        values='performance',
        index='learning_rate',
        columns='hidden_layer_sizes',
        aggfunc='mean'
    )

    # Cross Analysis: Activation x Architecture
    act_arch_stats = df.pivot_table(
        values='performance',
        index='activation_combo',
        columns='hidden_layer_sizes',
        aggfunc='mean'
    )

    # Top 5 Configurations
    top_5 = df.nsmallest(5, 'performance')[
        ['hidden_layer_sizes', 'activation_fn', 'output_activation_fn',
         'learning_rate', 'criterion', 'performance']
    ]

    # Calculate quartiles for context
    quartiles = df['performance'].quantile([0.25, 0.5, 0.75])

    # Prepare summary statistics
    summary_stats = {
        'total_configurations': len(df),
        'mean_performance': float(df['performance'].mean()),
        'std_performance': float(df['performance'].std()),
        'min_performance': float(df['performance'].min()),
        'max_performance': float(df['performance'].max()),
        'quartiles': {str(k): float(v) for k, v in quartiles.items()}
    }

    # Create detailed best configuration dictionary
    best_config_dict = {
        'architecture': best_config['hidden_layer_sizes'],
        'learning_rate': float(best_config['learning_rate']),
        'activation_fn': best_config['activation_fn'],
        'output_activation_fn': best_config['output_activation_fn'],
        'criterion': best_config['criterion'],
        'performance': float(best_config['performance'])
    }

    # Convert DataFrame results to nested dictionaries with proper float conversion
    def convert_stats_to_dict(stats_df):
        result = {}
        for col in stats_df.columns.levels[0]:
            result[col] = {}
            for idx in stats_df.index:
                result[col][idx] = {
                    k: float(v) if isinstance(v, (np.float64, np.float32)) else v
                    for k, v in stats_df[col].loc[idx].items()
                }
        return result

    # Apply conversions
    arch_stats_dict = convert_stats_to_dict(arch_stats)
    lr_stats_dict = convert_stats_to_dict(lr_stats)
    activation_stats_dict = convert_stats_to_dict(activation_stats)
    criterion_stats_dict = convert_stats_to_dict(criterion_stats)

    # Convert cross-analysis results
    lr_arch_dict = {
        str(idx): {str(col): float(val)
                   for col, val in row.items() if not pd.isna(val)}
        for idx, row in lr_arch_stats.iterrows()
    }

    act_arch_dict = {
        str(idx): {str(col): float(val)
                   for col, val in row.items() if not pd.isna(val)}
        for idx, row in act_arch_stats.iterrows()
    }

    # Compile all results
    results = {
        'summary_stats': summary_stats,
        'best_configuration': best_config_dict,
        'top_5_configurations': top_5.to_dict('records'),
        'architecture_statistics': arch_stats_dict,
        'learning_rate_statistics': lr_stats_dict,
        'activation_statistics': activation_stats_dict,
        'criterion_statistics': criterion_stats_dict,
        'lr_architecture_cross_analysis': lr_arch_dict,
        'activation_architecture_cross_analysis': act_arch_dict
    }

    return results

def print_analysis_report(results):
    """
    Print a formatted report of the hyperparameter search analysis.

    Parameters:
    results (dict): Output from analyze_hyperparameter_search function
    """
    print("=== Hyperparameter Search Analysis ===\n")

    # Print Summary Statistics
    print("Summary Statistics:")
    print(f"Total Configurations Tested: {results['summary_stats']['total_configurations']}")
    print(f"Mean Performance: {results['summary_stats']['mean_performance']:.4f}")
    print(f"Standard Deviation: {results['summary_stats']['std_performance']:.4f}")
    print(f"Performance Range: [{results['summary_stats']['min_performance']:.4f}, {results['summary_stats']['max_performance']:.4f}]")
    print("\nQuartiles:")
    for q, v in results['summary_stats']['quartiles'].items():
        print(f"Q{q}: {v:.4f}")
    print("\n" + "="*50 + "\n")

    # Print Best Configuration
    print("Best Configuration:")
    for key, value in results['best_configuration'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("\n" + "="*50 + "\n")

    # Print Top 5 Configurations
    print("Top 5 Configurations:")
    for i, config in enumerate(results['top_5_configurations'], 1):
        print(f"\n{i}.")
        for key, value in config.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    print("\n" + "="*50 + "\n")

    # Print Architecture Statistics
    print("Architecture Performance (Mean ± Std):")
    arch_stats = results['architecture_statistics']
    for arch in arch_stats['performance'].keys():
        mean = arch_stats['performance'][arch]['mean']
        std = arch_stats['performance'][arch]['std']
        print(f"{arch}: {mean:.4f} ± {std:.4f}")
    print("\n" + "="*50 + "\n")

    # Print Learning Rate Statistics
    print("Learning Rate Performance (Mean ± Std):")
    lr_stats = results['learning_rate_statistics']
    for lr in lr_stats['performance'].keys():
        mean = lr_stats['performance'][lr]['mean']
        std = lr_stats['performance'][lr]['std']
        print(f"LR {lr}: {mean:.4f} ± {std:.4f}")
    print("\n" + "="*50 + "\n")

    # Print Activation Function Statistics
    print("Activation Function Combinations (Mean ± Std):")
    act_stats = results['activation_statistics']
    for act in act_stats['performance'].keys():
        mean = act_stats['performance'][act]['mean']
        std = act_stats['performance'][act]['std']
        print(f"{act}: {mean:.4f} ± {std:.4f}")
    print("\n" + "="*50 + "\n")

    # Print Loss Function Statistics
    print("Loss Function Performance (Mean ± Std):")
    crit_stats = results['criterion_statistics']
    for crit in crit_stats['performance'].keys():
        mean = crit_stats['performance'][crit]['mean']
        std = crit_stats['performance'][crit]['std']
        print(f"{crit}: {mean:.4f} ± {std:.4f}")


## @brief Print all values stored in a GenerationParams object
#  @param gen_params GenerationParams object to print
def print_generation_params(gen_params):
    """
    Print all values stored in a GenerationParams object in a clean format.
    """
    print("Generation Parameters:")
    print(f"  Data Size: {gen_params.datasetSize}")
    print(f"  Number of Branches: {gen_params.numBranches}")
    print(f"  Length Interval: {gen_params.lengthInterval}")
    print(f"  Width Interval: {gen_params.widthInterval}")
    print(f"  Flip Parameters: {gen_params.flipParams}")
    print(f"  Rotation Parameters: {gen_params.dataRotationParams}")
    print(f"  Young's Modulus: {gen_params.modulusOfElasticity}")
    print(f"  Poisson Ratio: {gen_params.poissonRatio}")
    print(f"  Yield Stress: {gen_params.yieldStrength}")
    print(f"  Number of Perturbations: {gen_params.numPerturbations}")
    print(f"  Perturbation Probability: {gen_params.perturbProbability}")
    print(f"  Width Perturbation: {gen_params.widthPerturb}")
    print(f"  Vector Perturbation: {gen_params.vectorPerturb}")
    print(f"  Terminal Perturbation: {gen_params.terminalPerturb}")
    print(f"  Number of Displacements: {gen_params.numDisplacements}")
    print(f"  Percent Yield Strength: {gen_params.percentYieldStrength}")
    print(f"  Displacement Probability: {gen_params.displaceProbability}")
    print(f"  Mesh Size: {gen_params.meshSize}")
    print(f"  Order: {gen_params.order}")
    print(f"  Seed: {gen_params.seed}")

## @brief Load and print model information
#  @param filename Name of the model file to load
#  @return Loaded model
def load_and_print_model_info(filename):
    model = NeuralNetwork.load(filename)
    model.print_model_info()
    return model

# Note that for this function the filename does not desire the ".bin" ending to it
## @brief Generate and save dataset using an experiment function
#  @param experiment_function Function that returns generation parameters
#  @param filename Filename to save the dataset (without .bin extension)
def generate_save_dataset_using_experiment(experiment_function, filename):
    generation_params = experiment_function()
    cc_dataset = generate_dataset(generation_params, True)
    save_dataset(cc_dataset, filename)

# This function provides the generation parameters for the dataset of single branches for the basic experiment of
# training an NN to work on testNE1.geo
## @brief Generate parameters for single branch experiment (Experiment 0)
#  @details Provides generation parameters for the dataset of single branches
#           training an NN to work on testNE1.geo
#  @return GenerationParams object with configuration for single branch experiment
def experiment0_single_params():
    data_size = 100000
    num_branches = 1
    length_interval = (0.4, 0.9)
    width_interval = (1.2, 2.1)
    flip_params = (1, 0)
    rotation_params = (360, 2)
    youngs_modulus = 1000000000
    poisson = 0.33
    yield_stress = 5000000
    num_perturb = 2
    perturb_prob = 0.5
    width_perturb = 0.04, 0.02
    vector_perturb = 5, 0.5
    terminal_perturb = 0.025, 0.05
    num_displacements = 2
    percent_yield_strength = (0.01, 0.5)
    displace_probability = 0.75
    mesh_size = 0.1
    order = 1
    seed = 0

    gen_params = metal_foams.GenerationParams(data_size, num_branches, length_interval, width_interval, flip_params,
                    rotation_params, youngs_modulus, poisson, yield_stress, num_perturb, perturb_prob, width_perturb,
                    vector_perturb, terminal_perturb, num_displacements, percent_yield_strength, displace_probability,
                    mesh_size, order, seed)
    return gen_params

## @brief Train and save model for single branch experiment
#  @param data_type Type of dataset to use (PARAMETRIZATION or POINT)
#  @details Trains neural network on testNE1.geo single branch configuration
#           using predefined architecture and hyperparameters
def train_and_save_experiment0_single(data_type=DataType.PARAMETRIZATION):
    if data_type == DataType.PARAMETRIZATION:
        cc_dataset = load_dataset("testNE1_single")
    else:
        cc_dataset = load_dataset("testNE1_single_point_disp")
    generation_params = experiment0_single_params()
    training_params = {
        'num_branches': 1,
        'hidden_layer_sizes': [64, 32, 16],
        'generation_params': generation_params.to_dict(),
        'activation_fn': nn.ReLU,
        'batch_size': 16,
        'learning_rate': 0.01,
        'split_ratios': [0.7, 0.15, 0.15],
        'optimizer': 'adam',
        'scheduler': 'reduce_lr_on_plateau',
        'criterion': 'mse',
        'epochs': 500
    }
    if data_type == DataType.PARAMETRIZATION:
        train_and_save_model(cc_dataset, data_type, training_params, "testNE1_single.pth")
    else:
        train_and_save_model(cc_dataset, data_type, training_params, "testNE1_single_point.pth")


# This function provides the generation parameters for the dataset of multi-branches for the basic experiment of
# training an NN to work on testNE1.geo
## @brief Generate parameters for multi-branch experiment (Experiment 0)
#  @details Provides generation parameters for the dataset of multiple branches
#           training an NN to work on testNE1.geo
#  @return GenerationParams object with configuration for multi-branch experiment
def experiment0_multi_params():
    data_size = 50000
    num_branches = 3
    length_interval = (0.4, 0.9)
    width_interval = (1.2, 2.1)
    flip_params = (1, 0)
    rotation_params = (360, 2)
    youngs_modulus = 1000000000
    poisson = 0.33
    yield_stress = 5000000
    num_perturb = 2
    perturb_prob = 0.5
    width_perturb = 0.04, 0.02
    vector_perturb = 5, 0.5
    terminal_perturb = 0.025, 0.05
    num_displacements = 2
    percent_yield_strength = (0.05, 0.3)
    displace_probability = 0.9
    mesh_size = 0.12
    order = 1
    seed = 0

    gen_params = metal_foams.GenerationParams(data_size, num_branches, length_interval, width_interval, flip_params,
                    rotation_params, youngs_modulus, poisson, yield_stress, num_perturb, perturb_prob, width_perturb,
                    vector_perturb, terminal_perturb, num_displacements, percent_yield_strength, displace_probability,
                    mesh_size, order, seed)
    return gen_params

## @brief Train and save model for multi-branch experiment
#  @param data_type Type of dataset to use (PARAMETRIZATION or POINT)
#  @details Trains neural network on testNE1.geo multi-branch configuration
#           using predefined architecture and hyperparameters
def train_and_save_experiment0_multi(data_type=DataType.PARAMETRIZATION):
    if data_type == DataType.PARAMETRIZATION:
        cc_dataset = load_dataset("testNE1_multi")
    else:
        cc_dataset = load_dataset("testNE1_multi_point_disp")
    generation_params = experiment0_multi_params()
    training_params = {
        'num_branches': 3,
        'hidden_layer_sizes': [128, 64, 32, 16],
        'generation_params': generation_params.to_dict(),
        'activation_fn': nn.LeakyReLU,
        'batch_size': 16,
        'learning_rate': 0.0001,
        'split_ratios': [0.7, 0.15, 0.15],
        'optimizer': 'adam',
        'scheduler': 'reduce_lr_on_plateau',
        'criterion': 'mse',
        'epochs': 500
    }
    if data_type == DataType.PARAMETRIZATION:
        train_and_save_model(cc_dataset, data_type, training_params, "testNE1_multi.pth")
    else:
        train_and_save_model(cc_dataset, data_type, training_params, "testNE1_multi_point.pth")


# The following experiments are used to generate a dataset for testNE3.geo, which is the primary mesh
## @brief Generate parameters for Experiment 1 with small single branch dataset
#  @details Creates configuration for testNE3.geo with single branch and small dataset size (1000 samples)
#  @return GenerationParams object for small single branch experiment
def experiment1_single_params_small():
    data_size = 1000
    num_branches = 1
    length_interval = (0.8, 2.65)
    width_interval = (0.35, 0.61)
    flip_params = (1, 0)
    rotation_params = (360, 2)
    youngs_modulus = 1000000000
    poisson = 0.33
    yield_stress = 5000000
    num_perturb = 2
    perturb_prob = 0.5
    width_perturb = 0.05, 0.05
    vector_perturb = 25, 1.5
    terminal_perturb = 0.05, 0.1
    num_displacements = 2
    percent_yield_strength = (0, 0.35)
    displace_probability = 1
    mesh_size = 0.075
    order = 1
    seed = 0

    gen_params = metal_foams.GenerationParams(data_size, num_branches, length_interval, width_interval, flip_params,
                                              rotation_params, youngs_modulus, poisson, yield_stress, num_perturb, perturb_prob, width_perturb,
                                              vector_perturb, terminal_perturb, num_displacements, percent_yield_strength, displace_probability,
                                              mesh_size, order, seed)
    return gen_params

## @brief Generate parameters for Experiment 1 with medium single branch dataset
#  @details Modifies small dataset configuration to medium size (10,000 samples)
#  @return GenerationParams object for medium single branch experiment
def experiment1_single_params_medium():
    gen_params = experiment1_single_params_small()
    gen_params.datasetSize = 10000
    return gen_params

## @brief Generate parameters for Experiment 1 with large single branch dataset
#  @details Modifies small dataset configuration to large size (100,000 samples)
#  @return GenerationParams object for large single branch experiment
def experiment1_single_params_large():
    gen_params = experiment1_single_params_small()
    gen_params.datasetSize = 100000
    return gen_params

## @brief Generate parameters for Experiment 1 with small multi-branch dataset
#  @details Modifies single branch configuration to include 3 branches
#  @return GenerationParams object for small multi-branch experiment
def experiment1_multi_params_small():
    gen_params = experiment1_single_params_small()
    gen_params.numBranches = 3
    return gen_params

## @brief Generate parameters for Experiment 1 with medium multi-branch dataset
#  @details Modifies small multi-branch configuration to medium size (10,000 samples)
#  @return GenerationParams object for medium multi-branch experiment
def experiment1_multi_params_medium():
    gen_params = experiment1_single_params_small()
    gen_params.numBranches = 3
    gen_params.datasetSize = 10000
    return gen_params

## @brief Generate parameters for Experiment 1 with large multi-branch dataset
#  @details Modifies small multi-branch configuration to large size (100,000 samples)
#  @return GenerationParams object for large multi-branch experiment
def experiment1_multi_params_large():
    gen_params = experiment1_single_params_small()
    gen_params.numBranches = 3
    gen_params.datasetSize = 100000
    return gen_params

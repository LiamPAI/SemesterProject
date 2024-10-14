import os.path
import random
import sys

from torch.cuda import graph

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
import itertools
from typing import Dict, List, Any
import torch.nn as nn
from datetime import datetime
import csv

def hyperparameter_search(cc_dataset, dataset_type, search_space: Dict[str, List[Any]], num_trials:int = 10, csv_file: str = None):
    best_model = None
    best_performance = float('inf')
    best_params = None
    all_results = []

    # If no CSV file is specified, create a new one with a timestamp
    if csv_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"hyperparameter_{timestamp}.csv"

    # Check if the file already exists
    file_exists = os.path.isfile(csv_file)

    for i in range(num_trials):
        print(f"Running trial {i + 1}/{num_trials}")

        hyperparams = {key: random.choice(value) for key, value in search_space.items()}

        print("Selected hyperparameters:")
        for k, v in hyperparams.items():
            print(f"  {k}: {v}")

        model = NeuralNetwork(
            num_branches=hyperparams['num_branches'],
            data_type=dataset_type,
            hidden_layer_sizes=hyperparams['hidden_layer_sizes'],
            generation_params=hyperparams['generation_params'],
            activation_fn=hyperparams['activation_fn']
        )

        trainer = NeuralNetworkTrainer(
            NN_model=model,
            cc_dataset=cc_dataset,
            dataset_type=dataset_type,
            batch_size=hyperparams['batch_size'],
            learning_rate=hyperparams['learning_rate'],
            split_ratios=hyperparams['split_ratios'],
            optimizer_name=hyperparams['optimizer'],
            scheduler_name=hyperparams['scheduler'],
            criterion_name=hyperparams['criterion']
        )

        trainer.train(epochs=hyperparams['epochs'])

        performance_metrics = trainer.analyze_performance()
        current_performance = performance_metrics['overall_avg_loss']
        print(f"Performance: {current_performance:.6f}")

        result = {**hyperparams, 'performance': current_performance,
                  'generation_params': hyperparams['generation_params'].to_dict()}
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

    mode = 'a' if file_exists else 'w'
    with open(csv_file, mode, newline='') as csvfile:
        fieldnames = list(all_results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        for result in all_results:
            row = {}
            for k, v in result.items():
                if isinstance(v, (int, float, str)):
                    row[k] = v
                elif k == 'generation_params':
                    row[k] = str(v)  # v is already a dict from .to_dict()
                elif k == 'hidden_layer_sizes':
                    row[k] = str(v)
                elif k == 'split_ratios':
                    row[k] = str(v)
                elif k == 'activation_fn':
                    row[k] = v.__name__
                else:
                    row[k] = str(v)
            writer.writerow(row)

    print(f"\nResults {'appended to' if file_exists else 'saved in'} {csv_file}")

    return best_model, best_params, best_performance, all_results

def generate():
    data_size = 1000
    num_branches = 1
    length_interval = (0.25, 1)
    width_interval = (0.05, 0.1)
    flip_params = (1, 3)
    rotation_params = (360, 1)
    youngs_modulus = 1000000000
    poisson = 0.33
    yield_stress = 5000000
    num_perturb = 2
    perturb_prob = 0.75
    width_perturb = 0.1, 0.05
    vector_perturb = 25, 0.5
    terminal_perturb = 0.05, 0.1
    num_displacements = 2
    percent_yield_strength = 0.5
    displace_probability = 0.6
    mesh_size = 0.03
    order = 1
    seed = 0

    gen_params = metal_foams.GenerationParams(data_size, num_branches, length_interval, width_interval, flip_params, rotation_params,
                                              youngs_modulus, poisson, yield_stress, num_perturb, perturb_prob, width_perturb,
                                              vector_perturb, terminal_perturb, num_displacements, percent_yield_strength, displace_probability,
                                              mesh_size, order, seed)

    cc_dataset = generate_dataset(gen_params, True)

    save_dataset(cc_dataset, "test_NN_param1")
    return gen_params

if __name__ == "__main__":

    #  cc_dataset = load_dataset("test_NN_param0")

    #
    # search_space = {
    #     'num_branches': [1],
    #     'hidden_layer_sizes': [[64, 32, 16], [128, 64, 32, 16]],
    #     'generation_params': [params],
    #     'activation_fn': [nn.ReLU, nn.LeakyReLU],
    #     'batch_size': [16],
    #     'learning_rate': [0.01],
    #     'split_ratios': [(0.7, 0.15, 0.15)],
    #     'optimizer': ['adam'],
    #     'scheduler': ['reduce_lr_on_plateau'],
    #     'criterion': ['mse', 'mae'],
    #     'epochs': [500]
    # }
    # best_model, best_params, best_performance, all_results = hyperparameter_search(cc_dataset,
    #     DataType.PARAMETRIZATION, search_space, num_trials=8, csv_file="hyperparameter_20241011_182013.csv")

    graph_mesh = metal_foams.GraphMesh()
    graph_mesh.loadMeshFromFile("testNE1.geo")
    graph_mesh.buildGraphFromMesh()
    graph_mesh.printGraphState()
    part_graph = graph_mesh.splitMesh(0.5, 0.1)
    graph_mesh.printPartGraphState(part_graph)
    training_params = graph_mesh.getNNTrainingParams(part_graph)
    graph_mesh.printTrainingParams(training_params)
    conditions = graph_mesh.getCompatibilityConditions(part_graph)
    graph_mesh.printCompatibilityConditions(conditions)
    graph_mesh.closeMesh()

    params = generate()
    cc_dataset = load_dataset("test_NN_param1")
    print_dataset(cc_dataset)
import os.path
import random
import sys

import torch

from python.experiment_parameters import train_and_save_experiment0_single, train_and_save_experiment0_multi, \
    experiment1_single_params_small, train_and_save_model, experiment1_multi_params_large
from python.neural_network import load_dataset

sys.path.append("/Users/liamcurtis/Documents/ETH Classes/Winter 2024/Semester Project/Sem_Project_LC/cmake-build-debug/python")
import metal_foams

from experiment_parameters import (
    experiment0_single_params,
    experiment0_multi_params,
    experiment1_single_params_small,
    experiment1_single_params_medium,
    experiment1_single_params_large,
    experiment1_multi_params_small,
    experiment1_multi_params_medium,
    experiment1_multi_params_large,
    print_generation_params,
    hyperparameter_search,
    conduct_hyperparameter_search,
    train_and_save_experiment0_single,
    train_and_save_experiment0_multi,
    load_and_print_model_info,
    DataType,
    generate_dataset,
    save_dataset,
    load_dataset,
    print_dataset,
    parametrization_to_point,
    nn,
    analyze_hyperparameter_search,
    print_analysis_report
)
from calculate_displacements import (
    optimize_displacement_vectors,
    analyze_optimization_results,
    print_result_analysis,
    calculate_energy_test,
    calculate_energy_results
)
import numpy as np

if __name__ == "__main__":
    # TODO: When running the NN on the mesh, find out if certain properties of the parametrizations lead to higher
    #  energies/ especially compared to the expected zero entries
    # csv_name = 'hyperparameter_param_single_100k_20241031_115322.csv'
    # csv_name = 'hyperparameter_param_multi_100k_20241031_214647.csv'
    # csv_name = 'hyperparameter_point_single_100k_20241101_162645.csv'
    # csv_name = 'hyperparameter_point_multi_100k_20241102_053451.csv'
    # try:
    #     results = analyze_hyperparameter_search(csv_name)
    #     print("="*60)
    #     print_analysis_report(results)
    # except Exception as e:
    #     print(f"Error analyzing : {str(e)}")

    # cc_point_dataset = parametrization_to_point(cc_dataset)
    # save_dataset(cc_point_dataset, "testNE3_multi_point_large")
    # print_dataset(cc_dataset)

    # generation_params = experiment1_single_params_small()
    # cc_dataset = generate_dataset(generation_params, True)
    # save_dataset(cc_dataset, "testNE3_single_param_small_planeStrain")

    # generation_params = experiment1_single_params_medium()
    # cc_dataset = generate_dataset(generation_params, True)
    # save_dataset(cc_dataset, "testNE3_single_param_medium_planeStrain")
    #
    # generation_params = experiment1_single_params_large()
    # cc_dataset = generate_dataset(generation_params, True)
    # save_dataset(cc_dataset, "testNE3_single_param_large_planeStrain")
    #
    # generation_params = experiment1_multi_params_small()
    # cc_dataset = generate_dataset(generation_params, True)
    # save_dataset(cc_dataset, "testNE3_multi_param_small_planeStrain")
    #
    # generation_params = experiment1_multi_params_medium()
    # cc_dataset = generate_dataset(generation_params, True)
    # save_dataset(cc_dataset, "testNE3_multi_param_medium_planeStrain")
    #
    # generation_params = experiment1_multi_params_large()
    # cc_dataset = generate_dataset(generation_params, True)
    # save_dataset(cc_dataset, "testNE3_multi_param_large_planeStrain")



    # load_and_print_model_info("testNE3_multi_param.pth")
    # train_and_save_experiment0_single(DataType.POINT)

    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # search_space = {
    #     'num_branches': [1],
    #     'hidden_layer_sizes': [[128, 64, 32, 16], [64, 32, 16], [256, 128]],
    #     'generation_params': [generation_params],
    #     'activation_fn': [nn.ReLU, nn.LeakyReLU],
    #     'output_activation_fn': [nn.ReLU, nn.Softplus, nn.Mish],
    #     'batch_size': [256],
    #     'learning_rate': [0.01, 0.001, 0.0001],
    #     'split_ratios': [(0.7, 0.15, 0.15)],
    #     'optimizer': ['adam'],
    #     'scheduler': ['reduce_lr_on_plateau'],
    #     'criterion': ['mse', 'mae'],
    #     'epochs': [75]
    # }

    # gen_params = experiment1_single_params_large()
    # cc_dataset = load_dataset("testNE3_single_point_large")
    # optimal_parameters = {
    #     'num_branches': 1,
    #     'hidden_layer_sizes': [256, 128],
    #     'generation_params': gen_params,
    #     'activation_fn': nn.LeakyReLU,
    #     'output_activation_fn': nn.Mish,
    #     'batch_size': 256,
    #     'learning_rate': 0.01,
    #     'split_ratios': (0.7, 0.15, 0.15),
    #     'optimizer': 'adam',
    #     'scheduler': 'reduce_lr_on_plateau',
    #     'criterion': 'mae',
    #     'epochs': 100
    # }
    # train_and_save_model(cc_dataset, DataType.POINT, optimal_parameters, "testNE3_single_point.pth")

    gen_params = experiment1_multi_params_large()
    # cc_dataset = load_dataset("testNE3_multi_point_large")
    # optimal_parameters = {
    #     'num_branches': 3,
    #     'hidden_layer_sizes': [256, 128],
    #     'generation_params': gen_params,
    #     'activation_fn': nn.LeakyReLU,
    #     'output_activation_fn': nn.Softplus,
    #     'batch_size': 256,
    #     'learning_rate': 0.01,
    #     'split_ratios': (0.7, 0.15, 0.15),
    #     'optimizer': 'adam',
    #     'scheduler': 'reduce_lr_on_plateau',
    #     'criterion': 'mse',
    #     'epochs': 100
    # }
    # train_and_save_model(cc_dataset, DataType.POINT, optimal_parameters, "testNE3_multi_point.pth")



    # best_model, best_params, best_performance, all_results = hyperparameter_search(cc_dataset=cc_dataset,
    #             dataset_type=DataType.PARAMETRIZATION, search_space=search_space, search_type='random',
    #             num_trials=1, csv_file="hyperparameter_20241011_182013.csv")

    # datasets_dict = {
    #     'param_single_100k': {
    #         'filename': 'testNE3_single_param_large',
    #         'type': DataType.PARAMETRIZATION,
    #         'num_branches': 1
    #     },
    #     'param_multi_100k': {
    #         'filename': 'testNE3_multi_param_large',
    #         'type': DataType.PARAMETRIZATION,
    #         'num_branches': 3
    #     },
    #     'point_single_100k': {
    #         'filename': 'testNE3_single_point_large',
    #         'type': DataType.POINT,
    #         'num_branches': 1
    #     },
    #     'point_multi_100k': {
    #         'filename': 'testNE3_multi_point_large',
    #         'type': DataType.POINT,
    #         'num_branches': 3
    #     }
    # }
    # conduct_hyperparameter_search(datasets_dict=datasets_dict, search_space=search_space, search_type='combinatorial')

    single_branch_model = load_and_print_model_info("testNE3_single_point.pth")
    multi_branch_model = load_and_print_model_info("testNE3_multi_point.pth")

    model_dict = {
        1: single_branch_model,
        3: multi_branch_model
    }

    graph_mesh = metal_foams.GraphMesh()
    graph_mesh.loadMeshFromFile("testNE3.geo")

    graph_mesh.buildGraphFromMesh()

    part_graph = graph_mesh.splitMesh(1.5, 0.3)

    mesh_params = graph_mesh.getMeshParametrizations(part_graph, "testNE3.msh", gen_params.order)
    centered_mesh_params = graph_mesh.centerMeshParametrizations(mesh_params)
    centered_mesh_points = metal_foams.parametrizationToPoint(centered_mesh_params)

    initial_displacement_vectors = graph_mesh.getInitialDisplacements(part_graph)
    graph_mesh.printMeshParamsAndDisplacements(centered_mesh_params, initial_displacement_vectors)


    compat_conditions = graph_mesh.getCompatibilityConditions(part_graph)
    graph_mesh.printCompatibilityConditions(compat_conditions)

    test_displacements = []

    disp1 = np.array([[0.000], [0.000], [-0.001], [-0.001]])
    disp2 = np.array([[-0.001], [0.001], [0.000], [0.000]])
    disp3 = np.array([[0.0005], [-0.00075], [0.001], [0.001]])
    # disp4 = np.array([[0.0015], [0.002], [0.000], [0.000]])

    # disp1 = np.array([[0.000], [0.000], [0.000], [0.00]])
    # disp2 = np.array([[0.00], [0.00], [0.000], [0.000]])
    # disp3 = np.array([[0.00], [0.00], [0.00], [0.00]])
    # disp4 = np.array([[0.00], [0.00], [0.000], [0.000]])
    #
    test_displacements.append((1, disp1))
    test_displacements.append((2, disp2))
    test_displacements.append((3, disp3))
    # test_displacements.append((4, disp4))
    #
    fixed_conditions = graph_mesh.getFixedDisplacementConditions(part_graph, test_displacements)
    graph_mesh.printFixedDisplacementConditions(fixed_conditions)
    #
    calculation_params = metal_foams.CalculationParams(gen_params.yieldStrength, gen_params.modulusOfElasticity,
                                                       gen_params.poissonRatio, 0.05, gen_params.order)

    energy_list, total_energy = calculate_energy_test(model_dict, centered_mesh_points,
                                                      initial_displacement_vectors, DataType.POINT)
    # graph_mesh.printMeshParamsAndDisplacementsAndEnergies(centered_mesh_params, initial_displacement_vectors, energy_list)
    graph_mesh.printMeshPointsAndDisplacements(centered_mesh_points, initial_displacement_vectors, energy_list)
    print(f"Total energy with 0 displacements is: {total_energy}")

    # print("\nStarting to optimize the displacement vectors:")
    # parametrizations, optimized_vectors, energies = optimize_displacement_vectors(model_dict, centered_mesh_params,
    #                     initial_displacement_vectors, compat_conditions, fixed_conditions, DataType.PARAMETRIZATION)
    #
    # print(f"Optimized vectors:\n{optimized_vectors}")
    # print(f"\nOptimized energies:\n{energies}")

    points, true_displacements, true_energies = graph_mesh.meshEnergy(part_graph, "testNE3.msh", test_displacements, calculation_params)
    energy_results, total_result = calculate_energy_results(model_dict, centered_mesh_points,
                                                            true_displacements, DataType.POINT)
    graph_mesh.printMeshMatricesAndDisplacements(points, true_displacements, energy_results, true_energies)
    print(f"\nTotal energy: {total_result}")

    # new_flattened_param = np.array([ 0.6 ,  0.6,   0.6,   0.,    0.,    0.,    0.7,  0.,   -0.7, 1.,   1.,   1.,
    #                                  0.,    0. ,   0.  ])
    # new_flattened_param = np.array([ 0.6 ,  0.6,   0.6,   -0.75,    0.,    0.65,    0,  0.,   0, 0.,   0.,   0.,
    #                                  1.,    1. ,   1.  ])

    # new_flatted_disps = np.array([ 0.87995028,  0.13779895, -0.01958498,  -0.20669842, 0.87939061, 0.2755979,
    #                                 0.10894553, 0.2755979])
    # new_flatted_disps = np.array([ -0.01958498, -0.20669842, -0.87995028, -0.13779895, 0.10894553, 0.2755979,
    #                              -0.87939061,  -0.2755979])
    # new_flatted_disps = np.array([ 0, 0, 0, 0, 0, 0,
    #                                0,  0])
    #
    # model = model_dict[1]
    # input_data = np.concatenate([new_flattened_param, new_flatted_disps])
    # normalized_energy = model(torch.tensor(input_data, dtype=torch.float32)).item()
    # energy = normalized_energy / model.energy_scale * (model.energy_max - model.energy_min) + model.energy_min
    # print(f"Energy value for last param is {energy}")

    # print("Value of points is")
    # print(points)
    # print("Value of the displacements is")
    # print(true_displacements)
    # print("Value of the energies are")
    # print(true_energies)
    # print(sum(true_energies))

    # results = analyze_optimization_results(optimized_vectors, energies, true_displacements, true_energies)
    #
    # print_result_analysis(results)

    # (displacements, energy) = graph_mesh.meshFEMCalculation("testNE1.msh", test_displacements, calculation_params)
    # print(displacements)
    # print(energy)




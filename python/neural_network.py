## @package neural_network
#  Neural network implementations for metal foam analysis
#
#  This module provides classes and functions for training and using neural networks for the energy
#  minimization calculation on a metal foam
#
#  Key Components:
#  - NeuralNetwork: Main neural network architecture
#  - ParamPointDataset: Dataset handling for parameters and points
#  - NeuralNetworkTrainer: Training utilities and analysis

import os.path
import sys
sys.path.append("/Users/liamcurtis/Documents/ETH Classes/Winter 2024/Semester Project/Sem_Project_LC/cmake-build-debug/python")
import metal_foams

from collections import defaultdict
from enum import IntEnum
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt
import pickle
from enum import Enum
import random
from collections import Counter

# TODO: Comment the file more thoroughly
# TODO: Remove unnecessary import statements later
# TODO: Create a dataset that corresponds to the physical parameters of actual aluminum foam meshes to get an idea
#  of how to build one on Gmsh

## @brief Enum for indexing different types of tags in the dataset
class TagIndex(IntEnum):
    BASE = 0                ##< Base structure tag
    BASE_PERTURBATION = 1   ##< Base perturbation tag
    DISPLACEMENT = 2        ##< Displacement tag
    ROTATION = 3            ##< Rotation tag

## @brief Enum for different types of datasets
class DataType(Enum):
    PARAMETRIZATION = "parametrization" ##< Parametrization dataset type
    POINT = "point"                     ##< Point dataset type

# TODO: Test all of the below classes to make sure they make sense and output correctly
# TODO: Test the below functions with models that have more than 1 branch
# The following is a custom neural network class that holds the main elements of its architecture along with the
# parameters that generated the dataset it was trained on

## @brief Neural network model for FEM parametrization
#  @details Implements a fully connected neural network with customizable architecture
#          to approximate parametrizations and the resulting energy from prescribed displacement boundary conditions
class NeuralNetwork(nn.Module):
    ## @brief Initialize the neural network
    #  @param num_branches Number of branches in the parametrization
    #  @param data_type Type of data (PARAMETRIZATION or POINT)
    #  @param hidden_layer_sizes List of sizes for hidden layers
    #  @param generation_params Parameters used to generate the dataset
    #  @param activation_fn Activation function for hidden layers (default: ReLU)
    #  @param output_activation_fn Activation function for output layer (default: ReLU)
    def __init__(self, num_branches, data_type, hidden_layer_sizes, generation_params, activation_fn=nn.ReLU,
                 output_activation_fn=nn.ReLU):
        super(NeuralNetwork, self).__init__()

        self.num_branches = num_branches
        self.data_type = DataType(data_type)
        self.generation_params = generation_params

        # The number of branches, and the data type, determines the input size to this NN
        if self.data_type == DataType.PARAMETRIZATION:
            input_size = 19 * num_branches if num_branches >= 3 else 23
        elif self.data_type == DataType.POINT:
            input_size = 16 * num_branches if num_branches >= 3 else 20
        else:
            raise ValueError("Invalid data_type")

        layer_sizes = [input_size] + hidden_layer_sizes + [1]
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.activation = activation_fn()
        self.displacement_factor = None
        self.energy_min = None
        self.energy_max = None
        self.energy_scale = None

        self.training_params = None
        self.output_activation = output_activation_fn()

    ## @brief Forward pass through the network
    #  @param x Input tensor
    #  @return Output tensor after forward pass (scalar)
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return self.output_activation(x)

    ## @brief Set normalization factors for the model
    #  @param displacement_factor Factor for normalizing displacements
    #  @param energy_min Minimum energy value
    #  @param energy_max Maximum energy value
    #  @param energy_scale Scale factor for energy normalization
    def set_normalization_factors(self, displacement_factor, energy_min, energy_max, energy_scale):
        self.displacement_factor = displacement_factor
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.energy_scale = energy_scale

    ## @brief Set training parameters
    #  @param params Dictionary of training parameters
    def set_training_params(self, params):
        self.training_params = params

    # The save function holds the necessary parameters to train the neural network again using the same 
    # generation parameters for C++

    ## @brief Save the model to a file
    #  @details saves the model to the networks folder of the repo, extension ".pth" must be included
    #  @param filename Name of file to save the model
    #  @details Saves model state, architecture, and training parameters
    def save(self, filename):
        directory = os.path.join(os.getcwd(), "networks")
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)

        state = {
            'model_state_dict': self.state_dict(),
            'num_branches': self.num_branches,
            'data_type': self.data_type.value,
            'hidden_layer_sizes': [layer.out_features for layer in self.layers[:-1]],
            'activation_fn': type(self.activation).__name__,
            'output_activation_fn': type(self.output_activation).__name__,
            'generation_params': self.generation_params.to_dict(),
            'displacement_factor': self.displacement_factor,
            'energy_min': self.energy_min,
            'energy_max': self.energy_max,
            'energy_scale': self.energy_scale,
            'training_params': self.training_params
        }
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")

    ## @brief Load a model from a file
    #  @details The network will be searched for in the networks folder of the repo, and the ".pth" extension
    #  must be included
    #  @param filename Name of file to load the model from
    #  @return Loaded neural network model
    @classmethod
    def load(cls, filename):
        directory = os.path.join(os.getcwd(), "networks")
        filepath = os.path.join(directory, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")

        state = torch.load(filepath)
        activation_fn = getattr(nn, state['activation_fn'])
        output_activation_fn = getattr(nn, state['output_activation_fn'])
        generation_params = metal_foams.GenerationParams.from_dict(state['generation_params'])

        model = cls(
            num_branches=state['num_branches'],
            data_type=state['data_type'],
            hidden_layer_sizes=state['hidden_layer_sizes'],
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            generation_params=generation_params
        )
        model.load_state_dict(state['model_state_dict'])
        model.set_normalization_factors(state['displacement_factor'], state['energy_min'], state['energy_max'], state['energy_scale'])
        model.set_training_params(state['training_params'])
        return model

    ## @brief Print model information
    #  @details Prints architecture, parameters, and training information
    def print_model_info(self):
        print(f"Number of branches: {self.num_branches}")
        print(f"Data type: {self.data_type}")
        print(f"Hidden layer sizes: {[layer.out_features for layer in self.layers[:-1]]}")
        print(f"Activation function: {type(self.activation).__name__}")
        print(f"Output Activation function: {type(self.activation).__name__}")
        print(f"Displacement factor: {self.displacement_factor}")
        print(f"Energy min: {self.energy_min}")
        print(f"Energy max: {self.energy_max}")
        print(f"Energy scale: {self.energy_scale}")
        print("\nGeneration Parameters:")
        gen_params = self.generation_params
        attrs = [attr for attr in dir(gen_params) if not attr.startswith('__') and not callable(getattr(gen_params, attr))]
        for attr in attrs:
            value = getattr(gen_params, attr)
            if value is not None:
                print(f"  {attr}: {value}")
        if self.training_params:
            print("\nTraining Parameters:")
            for key, value in self.training_params.items():
                print(f"  {key}: {value}")


# The following class is designed to holds datasets of either type, being parametrizations, or points, and
# normalizes the displacements and energies as well
## @brief Dataset class for handling parametrization and point data
#  @details Processes and normalizes data for neural network training,
#           handling both parametrization types
class ParamPointDataset(Dataset):
    ## @brief Initialize the dataset
    #  @param cc_dataset Raw dataset from C++ code (of type Dataset)
    #  @param dataset_type Type of dataset (PARAMETRIZATION or POINT)
    #  @param energy_scale Scale factor for energy normalization (default: 100)
    def __init__(self, cc_dataset, dataset_type, energy_scale=100):
        self.data = []
        self.energies = []
        self.tags = []
        self.dataset_type = DataType(dataset_type)

        self.displacement_factor = 1.0
        self.energy_scale = energy_scale
        self.energy_min = float('inf')
        self.energy_max = float('-inf')

        self._process_and_normalize_data(cc_dataset)

        self.data = torch.FloatTensor(self.data)
        self.energies = torch.FloatTensor(self.energies)
        self.tags = torch.FloatTensor(self.tags)

    # TODO: decide later on, depending on the shape of the aluminum mesh, if I'll need to normalize the widths and
    #  terminals of parametrizations or points, as the normalization process would be different depending
    #  on the data type as well
    # TODO: When splitting up the data into training, validation, and testing sets, make sure it is not random,
    #  but sequential so that relevant parametrizations are not shared

    ## @brief Process and normalize the raw dataset
    #  @param cc_dataset Raw dataset from C++ code
    #  @details Calculates normalization factors and processes all entries so they are flattened for the neural network
    def _process_and_normalize_data(self, cc_dataset):
        displacement_max = 0.0

        for entry in cc_dataset:
            displacements = entry.displacements.flatten()
            displacement_max = max(displacement_max, np.max(np.abs(displacements)))
            self.energy_min = min(self.energy_min, entry.energyDifference)
            self.energy_max = max(self.energy_max, entry.energyDifference)

        self.displacement_factor = displacement_max if displacement_max > 0 else 1.0

        for entry in cc_dataset:
            if self.dataset_type == DataType.PARAMETRIZATION:
                input_features = self._process_parametrization_entry(entry)
            else:
                input_features = self._process_point_entry(entry)

            self.data.append(input_features)
            normalized_energy = self._normalize_energy(entry.energyDifference)
            self.energies.append(normalized_energy)
            self.tags.append(entry.tags)

    ## @brief Normalize energy values
    #  @param energy Raw energy value
    #  @return Normalized energy value
    def _normalize_energy(self, energy):
        return (energy - self.energy_min) / (self.energy_max - self.energy_min) * self.energy_scale

    ## @brief Denormalize energy values
    #  @param normalized_energy Normalized energy value
    #  @return Original energy value
    def _denormalize_energy(self, normalized_energy):
        return normalized_energy / self.energy_scale * (self.energy_max - self.energy_min) + self.energy_min

    ## @brief Process a parametrization entry
    #  @param entry Parametrization data entry
    #  @return Processed feature vector
    def _process_parametrization_entry(self, entry):
        widths = entry.param.widths.flatten()
        terminals = entry.param.terminals.flatten()
        vectors = entry.param.vectors.flatten()
        displacements = entry.displacements.flatten() / self.displacement_factor
        return np.concatenate([widths, terminals, vectors, displacements])

    ## @brief Process a point entry
    #  @param entry Point data entry
    #  @return Processed feature vector
    def _process_point_entry(self, entry):
        points = entry.points.points.flatten()
        displacements = entry.displacements.flatten() / self.displacement_factor
        return np.concatenate([points, displacements])

    ## @brief Get dataset length
    #  @return Number of entries in dataset
    def __len__(self):
        return len(self.data)

    ## @brief Get item by index
    #  @param idx Index of item
    #  @return Tuple of (data, energy, tags)
    def __getitem__(self, idx):
        return self.data[idx], self.energies[idx], self.tags[idx]

    ## @brief Get normalization factors
    #  @return Dictionary of normalization factors
    def get_normalization_factors(self):
        return {
            'displacement_factor': self.displacement_factor,
            'energy_min': self.energy_min,
            'energy_max': self.energy_max,
            'energy_scale': self.energy_scale
        }

    ## @brief Print dataset summary
    #  @details Prints size, feature dimensions, and normalization information
    def print_dataset_summary(self):
        print("Dataset Summary:")
        print(f"Total samples: {len(self.data)}")
        print(f"Input feature size: {self.data[0].size()}")
        print(f"Dataset type: {self.dataset_type}")
        print(f"Displacement factor: {self.displacement_factor}")
        print(f"Energy scale: {self.energy_scale}")
        print(f"Energy min: {self.energy_min}")
        print(f"Energy max: {self.energy_max}")
        print(f"Normalized energy range: {self.energies.min().item():.6f} to {self.energies.max().item():.6f}")
        print(f"Original energy range: {self._denormalize_energy(self.energies.min().item()):.6f} to "
              f"{self._denormalize_energy(self.energies.max().item()):.6f}")

        max_tags = torch.max(self.tags, dim=0).values
        print(f"There are {max_tags[0].item() + 1} base parametrizations")
        print(f"There are {max_tags[1].item() + 1} base perturbations")
        print(f"There are {max_tags[2].item() + 1} displacements")
        print(f"There are {max_tags[3].item() + 1} rotations")

# The following class is designed to train the neural network regardless of architecture, given a dataset and some
# hyper-parameters, note it also initializes the normalization factors for the neural network
## @brief Class for training neural networks
#  @details Handles the complete training pipeline including data preparation,
#           training loop, validation, and performance analysis
class NeuralNetworkTrainer:
    ## @brief Initialize the trainer
    #  @param NN_model Neural network model to train
    #  @param cc_dataset Raw dataset from C++
    #  @param dataset_type Type of dataset (PARAMETRIZATION or POINT)
    #  @param batch_size Size of training batches (default: 32)
    #  @param learning_rate Learning rate for optimization (default: 0.001)
    #  @param split_ratios Tuple of (train, val, test) ratios (default: (0.7, 0.15, 0.15))
    #  @param optimizer_name Name of optimizer to use (default: 'adam')
    #  @param scheduler_name Name of learning rate scheduler (default: 'reduce_lr_on_plateau')
    #  @param criterion_name Name of loss function (default: 'mse')
    def __init__(self, NN_model, cc_dataset, dataset_type, batch_size=32, learning_rate=0.001, split_ratios=(0.7, 0.15, 0.15),
                 optimizer_name='adam', scheduler_name='reduce_lr_on_plateau', criterion_name='mse'):
        self.NN_model = NN_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.split_ratios = split_ratios
        self.dataset_type = dataset_type
        self.optimizer_name = optimizer_name.lower()
        self.scheduler_name = scheduler_name.lower()
        self.criterion_name = criterion_name.lower()

        self.dataset = ParamPointDataset(cc_dataset, dataset_type)
        self.prepare_data()

        self.NN_model.set_normalization_factors(self.dataset.displacement_factor, self.dataset.energy_min,
                                                self.dataset.energy_max, self.dataset.energy_scale)

        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        self.train_losses = []
        self.val_losses = []

    ## @brief Get the loss criterion
    #  @return Loss function
    def _get_criterion(self):
        if self.criterion_name == 'mse':
            return nn.MSELoss(reduction='mean')
        elif self.criterion_name == 'mae':
            return nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"Invalid criterion sent to NeuralNetworkTrainer: {self.criterion_name}")

    ## @brief Get the optimizer
    #  @return Optimizer instance
    def _get_optimizer(self):
        if self.optimizer_name == 'adam':
            return optim.Adam(self.NN_model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'rmsprop':
            return optim.RMSprop(self.NN_model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return optim.SGD(self.NN_model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Invalid optimizer sent to NeuralNetworkTrainer: {self.optimizer_name}")

    ## @brief Get the learning rate scheduler
    #  @return Scheduler instance or None
    def _get_scheduler(self):
        if self.scheduler_name == 'reduce_lr_on_plateau':
            return ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        elif self.scheduler_name == 'step_lr':
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Invalid scheduler sent to NeuralNetworkTrainer: {self.scheduler_name}")

    ## @brief Prepare data by splitting into train, validation, and test sets
    def prepare_data(self):
        total_size = len(self.dataset)
        train_size = int(self.split_ratios[0] * total_size)
        val_size = int(self.split_ratios[1] * total_size)
        test_size = total_size - train_size - val_size

        # Create sequential indices for each split, since nearby entries of the dataset can be similar
        indices = range(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

    ## @brief Train the model
    #  @param epochs Number of epochs to train
    def train(self, epochs):
        for epoch in range(epochs):
            # Set the mode of the neural network to train to track gradients
            self.NN_model.train()
            train_loss = self._train_epoch()
            val_loss = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Update the learning rate if we have set a scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

    ## @brief Train for one epoch
    #  @return Average training loss for the epoch
    def _train_epoch(self):
        total_loss = 0
        for inputs, energies, _ in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.NN_model(inputs).squeeze()
            loss = self.criterion(outputs, energies)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    ## @brief Validate the model
    #  @return Average validation loss
    def validate(self):
        # Set the mode to eval to ensure 0 gradients
        self.NN_model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, energies, _ in self.val_loader:
                outputs = self.NN_model(inputs).squeeze()
                loss = self.criterion(outputs, energies)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    ## @brief Test the model
    #  @return Tuple of (average loss, all losses, all tags)
    def test(self):
        # Ensure no gradient tracking when evaluating the test set
        self.NN_model.eval()
        total_loss = 0
        all_losses = []
        all_tags = []
        with torch.no_grad():
            for inputs, energies, tags in self.test_loader:
                outputs = self.NN_model(inputs)
                batch_losses = ((outputs.squeeze() - energies) ** 2)
                total_loss += batch_losses.sum().item()
                all_losses.extend(batch_losses.tolist())
                all_tags.extend(tags.tolist())

        avg_loss = total_loss / len(self.test_loader.dataset)
        return avg_loss, all_losses, all_tags

    # The function below uses the tagging system I added in data_operations to see how the neural network performs
    # with respect to rotation invariance, flip invariance, base performance, and so on
    ## @brief Analyze model performance across different aspects
    #  @return Dictionary containing performance metrics
    def analyze_performance(self):
        avg_loss, all_losses, all_tags = self.test()

        def get_tag_performance(index):
            performance_dict = defaultdict(list)
            for loss, tags in zip(all_losses, all_tags):
                key = tuple(tags[:index+1])
                performance_dict[key].append(loss)

            performances = [np.mean(losses) for losses in performance_dict.values()]
            return np.mean(performances) if performances else None

        base_performance = get_tag_performance(TagIndex.BASE)
        perturbation_performance = get_tag_performance(TagIndex.BASE_PERTURBATION)
        displacement_performance = get_tag_performance(TagIndex.DISPLACEMENT)
        rotation_performance = get_tag_performance(TagIndex.ROTATION)

        print("Performance Analysis:")
        print(f"Overall Average Loss: {avg_loss:.6f}")
        print(f"Base Performance: {base_performance:.6f}")
        print(f"Perturbation Performance: {perturbation_performance:.6f}")
        print(f"Displacement Performance: {displacement_performance:.6f}")
        print(f"Rotation Performance: {rotation_performance:.6f}")

        if self.dataset_type == DataType.PARAMETRIZATION:
            flip_differences = []
            for i in range(len(all_tags)):
                for j in range(i + 1, len(all_tags)):
                    if all_tags[i][:TagIndex.ROTATION] == all_tags[j][:TagIndex.ROTATION] and all_tags[i] != all_tags[j]:
                        flip_differences.append(abs(all_losses[i] - all_losses[j]))

            if flip_differences:
                avg_flip_difference = np.mean(flip_differences)
                print(f"Average Flip Performance Difference: {avg_flip_difference:.6f}")
            else:
                avg_flip_difference = None
                print("No flip pairs found in the dataset")
        else:
            avg_flip_difference = None

        return {
            "overall_avg_loss": avg_loss,
            "base": base_performance,
            "perturbation": perturbation_performance,
            "displacement": displacement_performance,
            "rotation": rotation_performance,
            "avg_flip_difference": avg_flip_difference
        }

    ## @brief Plot training and validation losses
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    ## @brief Plot learning rate changes
    def plot_learning_rate(self):
        plt.figure(figsize=(10, 5))
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Over Time')
        plt.grid(True)
        plt.show()

    ## @brief Print dataset information
    def print_dataset_info(self):
        print("Dataset Information:")
        print(f"Total dataset size: {len(self.dataset)}")
        print(f"Training set size: {len(self.train_loader.dataset)}")
        print(f"Validation set size: {len(self.val_loader.dataset)}")
        print(f"Test set size: {len(self.test_loader.dataset)}")

## @brief Get the number of displacement values based on number of branches
#  @param num_branches Number of branches in the structure
#  @return Number of displacement values
def get_displacement_count(num_branches):
    return 8 if num_branches == 1 else 4 * num_branches

# TODO: Make sure these functions are correct, in that I will likely be using them quite a bit, or at least their
#  skeletons, when optimizing the displacement vectors for the overall mesh
# TODO: Once I've seen if calculate_displacements.py actually works, consider getting rid of these functions

## @brief Normalize input data
#  @param input_data Raw input data
#  @param displacement_factor Factor for normalizing displacements
#  @param num_branches Number of branches in the structure
#  @param data_type Type of data (PARAMETRIZATION or POINT)
#  @return Normalized input data
def normalize_input(input_data, displacement_factor, num_branches, data_type):
    num_displacement = get_displacement_count(num_branches)

    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)

    if data_type == DataType.PARAMETRIZATION:
        non_displacement = input_data[:, :-num_displacement]
        displacements = input_data[:, -num_displacement:] / displacement_factor
        normalized = np.concatenate([non_displacement, displacements], axis=1)
    elif data_type == DataType.POINT:
        points = input_data[:, :-num_displacement]
        displacements = input_data[:, -num_displacement:] / displacement_factor
        normalized = np.concatenate([points, displacements], axis=1)
    else:
        raise ValueError("Invalid data type sent to normalize_input")

    return normalized.squeeze()

## @brief Denormalize input data
#  @param normalized_input Normalized input data
#  @param displacement_factor Factor for denormalizing displacements
#  @param num_branches Number of branches in the structure
#  @param data_type Type of data (PARAMETRIZATION or POINT)
#  @return Denormalized input data
def denormalize_input(normalized_input, displacement_factor, num_branches, data_type):
    num_displacement = get_displacement_count(num_branches)

    # Ensure normalized_input is 2D
    if normalized_input.ndim == 1:
        normalized_input = normalized_input.reshape(1, -1)

    if data_type == DataType.PARAMETRIZATION:
        non_displacement = normalized_input[:, :-num_displacement]
        displacements = normalized_input[:, -num_displacement:] * displacement_factor
        denormalized = np.concatenate([non_displacement, displacements], axis=1)
    elif data_type == DataType.POINT:
        points = normalized_input[:, :-num_displacement]
        displacements = normalized_input[:, -num_displacement:] * displacement_factor
        denormalized = np.concatenate([points, displacements], axis=1)
    else:
        raise ValueError("Invalid data type sent to denormalize_input")

    return denormalized.squeeze()

## @brief Normalize energy value
#  @param energy Raw energy value
#  @param energy_min Minimum energy value
#  @param energy_max Maximum energy value
#  @param energy_scale Scale factor for normalization
#  @return Normalized energy value
def normalize_energy(energy, energy_min, energy_max, energy_scale):
    return (energy - energy_min) / (energy_max - energy_min) * energy_scale

## @brief Denormalize energy value
#  @param normalized_energy Normalized energy value
#  @param energy_min Minimum energy value
#  @param energy_max Maximum energy value
#  @param energy_scale Scale factor used in normalization
#  @return Original energy value
def denormalize_energy(normalized_energy, energy_min, energy_max, energy_scale):
    return normalized_energy / energy_scale * (energy_max - energy_min) + energy_min

## @brief Use model for inference
#  @param model Trained neural network model
#  @param input_data Input data for inference
#  @return Model prediction (denormalized)
def use_model_for_inference(model, input_data):
    normalized_input = normalize_input(input_data, model.displacement_factor, model.num_branches, model.data_type)
    normalized_input_tensor = torch.FloatTensor(normalized_input)

    if normalized_input_tensor.ndim == 1:
        normalized_input_tensor = normalized_input_tensor.unsqueeze(0)

    with torch.no_grad():
        normalized_output = model(normalized_input_tensor)
        denormalized_output = denormalize_energy(normalized_output.numpy().squeeze(),
                                                 model.energy_min, model.energy_max, model.energy_scale)
    return denormalized_output

# The below functions simply use the functions defined by the pybindings in src/bindings.cc for easy access
## @brief Print dataset contents
#  @param dataset Dataset to print
def print_dataset(dataset):
    metal_foams.printDataSet(dataset)

## @brief Load dataset from file
#  @param filename Name of file to load (no ".bin" extenstion)
#  @return Loaded dataset
def load_dataset(filename):
    return metal_foams.loadDataSet(filename)

## @brief Save dataset to file
#  @param dataset Dataset to save (no ".bin" extenstion)
#  @param filename Name of file to save to
def save_dataset(dataset, filename):
    metal_foams.saveDataSet(dataset, filename)

## @brief Convert parametrization dataset to point dataset
#  @param parametrization_dataset Dataset to convert
#  @return Converted point dataset
def parametrization_to_point(parametrization_dataset):
    return metal_foams.parametrizationToPoint(parametrization_dataset)

## @brief Generate a new dataset
#  @param generation_params Parameters for dataset generation
#  @param verbose Whether to print verbose output (default: False)
#  @return Generated dataset
def generate_dataset(generation_params, verbose=False):
    return metal_foams.generateParametrizationDataSet(generation_params, verbose)
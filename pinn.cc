//
// Created by Liam Curtis on 2024-05-10.
//
#include <torch/torch.h>
#include <iostream>
#include "mesh.h"

using namespace torch::indexing;

//Structure for the overall neural network
struct LinearElasticityPINN: torch::nn::Module {

    //Implement the neural network with 5 layers, each with a size hidden_size
    //Input to the network should be size 2 (coordinates x and y)
    LinearElasticityPINN(int input_size, int output_size, int hidden_size) :
        input(torch::nn::Linear(input_size, hidden_size)),
        hidden_0(torch::nn::Linear(hidden_size, hidden_size)),
        hidden_1(torch::nn::Linear(hidden_size, hidden_size)),
        hidden_2(torch::nn::Linear(hidden_size, hidden_size)),
        hidden_3(torch::nn::Linear(hidden_size, hidden_size)),
        hidden_4(torch::nn::Linear(hidden_size, hidden_size)),
        output(torch::nn::Linear(hidden_size, output_size))
        {
        register_module("input", input);
        register_module("hidden_0", hidden_0);
        register_module("hidden_1", hidden_1);
        register_module("hidden_2", hidden_2);
        register_module("hidden_3", hidden_3);
        register_module("hidden_4", hidden_4);
        register_module("output", output);
    }

    //We choose tanh as activation function as it performed the best in the paper
    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(input(x));
        x = torch::tanh(hidden_0(x));
        x = torch::tanh(hidden_1(x));
        x = torch::tanh(hidden_2(x));
        x = torch::tanh(hidden_3(x));
        x = torch::tanh(hidden_4(x));
        x = output(x);

        return x;
    }

    torch::nn::Linear input, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4, output;
};

int main() {

    test_mesh("/meshes/test0.msh");

}
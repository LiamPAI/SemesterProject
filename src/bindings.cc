//
// Created by Liam Curtis on 06.10.2024.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <mesh_parametrization.h>
#include <data_operations.h>

PYBIND11_MODULE(metal_foams, m) {
    pybind11::class_<GenerationParams>(m, "GenerationParams")
        .def(pybind11::init<>())
        .def_readwrite("datasetSize", &GenerationParams::datasetSize)
        .def_readwrite("numBranches", &GenerationParams::numBranches)
        .def_readwrite("lengthInterval", &GenerationParams::lengthInterval)
        .def_readwrite("widthInterval", &GenerationParams::widthInterval)
        .def_readwrite("flipParams", &GenerationParams::flipParams)
        .def_readwrite("dataRotationParams", &GenerationParams::dataRotationParams)
        .def_readwrite("modulusOfElasticity", &GenerationParams::modulusOfElasticity)
        .def_readwrite("poissonRatio", &GenerationParams::poissonRatio)
        .def_readwrite("yieldStrength", &GenerationParams::yieldStrength)
        .def_readwrite("numPerturbations", &GenerationParams::numPerturbations)
        .def_readwrite("perturbProbability", &GenerationParams::perturbProbability)
        .def_readwrite("width_perturb", &GenerationParams::width_perturb)
        .def_readwrite("vector_perturb", &GenerationParams::vector_perturb)
        .def_readwrite("terminal_perturb", &GenerationParams::terminal_perturb)
        .def_readwrite("numDisplacements", &GenerationParams::numDisplacements)
        .def_readwrite("percentYieldStrength", &GenerationParams::percentYieldStrength)
        .def_readwrite("displaceProbability", &GenerationParams::displaceProbability)
        .def_readwrite("meshSize", &GenerationParams::meshSize)
        .def_readwrite("order", &GenerationParams::order)
        .def_readwrite("seed", &GenerationParams::seed);

    pybind11::class_<MeshParametrizationData>(m, "MeshParametrizationData")
        .def(pybind11::init<int, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>())
        .def_readwrite("numBranches", &MeshParametrizationData::numBranches)
        .def_readwrite("widths", &MeshParametrizationData::widths)
        .def_readwrite("terminals", &MeshParametrizationData::terminals)
        .def_readwrite("vectors", &MeshParametrizationData::vectors);

    pybind11::class_<ParametrizationPoints>(m, "ParametrizationPoints")
        .def(pybind11::init<int, Eigen::MatrixXd>())
        .def_readwrite("numBranches", &ParametrizationPoints::numBranches)
        .def_readwrite("points", &ParametrizationPoints::points);

    pybind11::class_<ParametrizationEntry>(m, "ParametrizationEntry")
        .def(pybind11::init<int, MeshParametrizationData, Eigen::MatrixXd, double>())
        .def_readwrite("numBranches", &ParametrizationEntry::numBranches)
        .def_readwrite("param", &ParametrizationEntry::param)
        .def_readwrite("displacements", &ParametrizationEntry::displacements)
        .def_readwrite("energyDifference", &ParametrizationEntry::energyDifference)
        .def_readwrite("tags", &ParametrizationEntry::tags);

    pybind11::class_<PointEntry>(m, "PointEntry")
        .def(pybind11::init<int, ParametrizationPoints, Eigen::MatrixXd, double, std::vector<int>>())
        .def_readwrite("numBranches", &PointEntry::numBranches)
        .def_readwrite("points", &PointEntry::points)
        .def_readwrite("displacements", &PointEntry::displacements)
        .def_readwrite("energyDifference", &PointEntry::energyDifference)
        .def_readwrite("tags", &PointEntry::tags);

    m.def("generateParametrizationDataSet", &DataOperations::generateParametrizationDataSet,
        "Generate a ParametrizationDataSet", pybind11::arg("gen_params"));
    m.def("parametrizationToPoint", &DataOperations::parametrizationToPoint,
        "Convert ParametrizationDataSet to PointDataSet", pybind11::arg("param_dataset"));

    pybind11::enum_<TagIndex>(m, "TagIndex")
        .value("BASE", TagIndex::BASE)
        .value("BASE_PERTURBATION", TagIndex::BASE_PERTURBATION)
        .value("DISPLACEMENT", TagIndex::DISPLACEMENT)
        .value("ROTATION", TagIndex::ROTATION);
}
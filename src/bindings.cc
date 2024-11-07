//
// Created by Liam Curtis on 06.10.2024.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/chrono.h>
#include <mesh_parametrization.h>
#include <data_operations.h>
#include <graph_mesh.h>
#include "../test/test_data_operations.cc"

// TODO: Include all the bindings related to graph_mesh that we will need to "run" the FEM calculation on the NN

/** @file
 *  @brief Python bindings for all functions related to the metal foams
 *
 *  This file provides the Python bindings interface access the core C++ functionality. It exposes classes and functions
 *  for metal foam structure generation, mesh operations, and finite element calculations.
 */
PYBIND11_MODULE(metal_foams, m) {
    pybind11::class_<GenerationParams>(m, "GenerationParams")
    .def(pybind11::init<int, int, std::pair<double, double>, std::pair<double, double>,
                  std::pair<double, int>, std::pair<double, double>, double, double,
                  double, int, double, std::pair<double, double>,
                  std::pair<double, double>, std::pair<double, double>,
                  int, std::pair<double, double>, double, double, int, unsigned>(),
         pybind11::arg("size"), pybind11::arg("numB"), pybind11::arg("lenInterval"), pybind11::arg("widInterval"),
         pybind11::arg("flipP"), pybind11::arg("dataRotP"), pybind11::arg("modulus"), pybind11::arg("poisson"),
         pybind11::arg("yieldStren"), pybind11::arg("numPer"), pybind11::arg("perProb"), pybind11::arg("widPer"),
         pybind11::arg("vecPer"), pybind11::arg("termPer"), pybind11::arg("numDisp"), pybind11::arg("perStrength"),
         pybind11::arg("dispProb"), pybind11::arg("meshS"), pybind11::arg("o"),
         pybind11::arg("s") = std::chrono::system_clock::now().time_since_epoch().count())
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
    .def_readwrite("widthPerturb", &GenerationParams::widthPerturb)
    .def_readwrite("vectorPerturb", &GenerationParams::vectorPerturb)
    .def_readwrite("terminalPerturb", &GenerationParams::terminalPerturb)
    .def_readwrite("numDisplacements", &GenerationParams::numDisplacements)
    .def_readwrite("percentYieldStrength", &GenerationParams::percentYieldStrength)
    .def_readwrite("displaceProbability", &GenerationParams::displaceProbability)
    .def_readwrite("meshSize", &GenerationParams::meshSize)
    .def_readwrite("order", &GenerationParams::order)
    .def_readwrite("seed", &GenerationParams::seed)
    .def("to_dict", &GenerationParams::to_dict)
    .def_static("from_dict", &GenerationParams::from_dict);

    pybind11::class_<MeshParametrizationData>(m, "MeshParametrizationData")
        .def(pybind11::init<int, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>())
        .def_readwrite("numBranches", &MeshParametrizationData::numBranches)
        .def_readwrite("widths", &MeshParametrizationData::widths)
        .def_readwrite("terminals", &MeshParametrizationData::terminals)
        .def_readwrite("vectors", &MeshParametrizationData::vectors);

    pybind11::class_<calculationParams>(m, "CalculationParams")
        .def(pybind11::init<double, double, double, double, int>())
        .def_readwrite("yield_strength", &calculationParams::yieldStrength)
        .def_readwrite("youngs_modulus", &calculationParams::youngsModulus)
        .def_readwrite("poisson_ratio", &calculationParams::poissonRatio)
        .def_readwrite("mesh_size", &calculationParams::meshSize)
        .def_readwrite("order", &calculationParams::order);

    m.def("polynomialPoints", &MeshParametrization::polynomialPoints,
        "Find the points of a parametrization", pybind11::arg("param"));

    m.def("displacementEnergy", &MeshParametrization::displacementEnergy,
        "Return the energy given a vector of displacements", pybind11::arg("param"),
        pybind11::arg("displacement"), pybind11::arg("calc_params"));

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

    pybind11::enum_<DataSetType>(m, "DataSetType")
        .value("Parametrization", DataSetType::Parametrization)
        .value("Point", DataSetType::Point);

    pybind11::class_<ParametrizationDataSet>(m, "ParametrizationDataSet")
        .def(pybind11::init<>())
        .def("__len__", [](const ParametrizationDataSet &v) { return v.size(); })
        .def("__iter__", [](ParametrizationDataSet &v) {
            return pybind11::make_iterator(v.begin(), v.end());
        }, pybind11::keep_alive<0, 1>())
        .def("__getitem__", [](const ParametrizationDataSet &v, size_t i) {
            if (i >= v.size()) throw pybind11::index_error();
            return v[i];
        });

    pybind11::class_<PointDataSet>(m, "PointDataSet")
        .def(pybind11::init<>())
        .def("__len__", [](const PointDataSet &v) { return v.size(); })
        .def("__iter__", [](PointDataSet &v) {
            return pybind11::make_iterator(v.begin(), v.end());
        }, pybind11::keep_alive<0, 1>())
        .def("__getitem__", [](const PointDataSet &v, size_t i) {
            if (i >= v.size()) throw pybind11::index_error();
            return v[i];
        });

    pybind11::class_<DataSet>(m, "DataSet")
        .def(pybind11::init<>())
        .def(pybind11::init<ParametrizationDataSet>())
        .def(pybind11::init<PointDataSet>());

    pybind11::enum_<TagIndex>(m, "TagIndex")
        .value("BASE", TagIndex::BASE)
        .value("BASE_PERTURBATION", TagIndex::BASE_PERTURBATION)
        .value("DISPLACEMENT", TagIndex::DISPLACEMENT)
        .value("ROTATION", TagIndex::ROTATION);

    m.def("generateParametrizationDataSet", &DataOperations::generateParametrizationDataSet,
        "Generate a ParametrizationDataSet", pybind11::arg("gen_params"), pybind11::arg("verbose") = false);
    m.def("parametrizationToPoint",
          pybind11::overload_cast<ParametrizationDataSet&>(&DataOperations::parametrizationToPoint),
          "Convert ParametrizationDataSet to PointDataSet",
          pybind11::arg("param_dataset"));
    m.def("parametrizationToPoint",
          pybind11::overload_cast<const std::vector<MeshParametrizationData>&>(&DataOperations::parametrizationToPoint),
          "Convert vector of MeshParametrizationData to vector of ParametrizationPoints",
          pybind11::arg("param_vector"));

    m.def("saveDataSet", &DataOperations::saveDataSet, "Save a dataset to a file");
    m.def("loadDataSet", &DataOperations::loadDataSet, "Load a dataset from a file");

    m.def("printDataSet", &printDataSet, "Print a dataset");
    m.def("printParametrizationDataSet", &printParametrizationDataSet, "Print a parametrization dataset");
    m.def("printPointDataSet", &printPointDataSet, "Print a point dataset");

    pybind11::class_<Node2D>(m, "Node2D")
        .def(pybind11::init<>())
        .def_readwrite("id", &Node2D::id)
        .def_readwrite("surfaceTags", &Node2D::surfaceTags)
        .def_readwrite("boundaryTags", &Node2D::boundaryTags)
        .def_readwrite("connectedEdges", &Node2D::connectedEdges);

    pybind11::class_<Edge2D>(m, "Edge2D")
        .def(pybind11::init<>())
        .def_readwrite("id", &Edge2D::id)
        .def_readwrite("surfaceTags", &Edge2D::surfaceTags)
        .def_readwrite("boundaryTags", &Edge2D::boundaryTags)
        .def_readwrite("connectedNodes", &Edge2D::connectedNodes);

    pybind11::class_<GraphMesh> graphMesh(m, "GraphMesh");

    pybind11::enum_<MeshPart::Type>(m, "MeshPartType")
        .value("NODE", MeshPart::Type::NODE)
        .value("EDGE", MeshPart::Type::EDGE)
        .export_values();

    pybind11::class_<MeshPart>(m, "MeshPart")
        .def(pybind11::init<>())
        .def_readwrite("type", &MeshPart::type)
        .def_readwrite("id", &MeshPart::id)
        .def_readwrite("subId", &MeshPart::subId)
        .def_readwrite("surfaceTags", &MeshPart::surfaceTags)
        .def_readwrite("boundaryTags", &MeshPart::boundaryTags)
        .def_readwrite("connectedEdges", &MeshPart::connectedEdges)
        .def_readwrite("connectedEdgeTags", &MeshPart::connectedEdgeTags)
        .def_readwrite("curveTags", &MeshPart::curveTags);

    pybind11::class_<PartGraph>(m, "PartGraph")
        .def(pybind11::init<>())
        .def_readwrite("parts", &PartGraph::parts)
        .def_readwrite("adjacencyList", &PartGraph::adjacencyList);

    pybind11::class_<NNTrainingParams>(m, "NNTrainingParams")
        .def(pybind11::init<std::pair<double, double>, std::pair<double, double>,
                      std::pair<double, double>, std::pair<double, double>>())
        .def_readwrite("minMaxLength", &NNTrainingParams::minMaxLength)
        .def_readwrite("minMaxWidth", &NNTrainingParams::minMaxWidth)
        .def_readwrite("minMaxWidthDiff", &NNTrainingParams::minMaxWidthDiff)
        .def_readwrite("minMaxAngleDiff", &NNTrainingParams::minMaxAngleDiff);

    pybind11::class_<CompatibilityCondition>(m, "CompatibilityCondition")
        .def(pybind11::init<std::pair<int, int>, std::pair<int, int>, std::pair<int, int>>())
        .def_readwrite("indices", &CompatibilityCondition::indices)
        .def_readwrite("firstLocation", &CompatibilityCondition::firstLocation)
        .def_readwrite("secondLocation", &CompatibilityCondition::secondLocation);

    pybind11::class_<FixedDisplacementCondition>(m, "FixedDisplacementCondition")
        .def(pybind11::init<std::pair<int, int>, Eigen::Vector4d>())
        .def_readwrite("indices", &FixedDisplacementCondition::indices)
        .def_readwrite("displacements", &FixedDisplacementCondition::displacements);

    graphMesh.def(pybind11::init<>())
        .def("loadMeshFromFile", &GraphMesh::loadMeshFromFile)
        .def("closeMesh", &GraphMesh::closeMesh)
        .def("buildGraphFromMesh", &GraphMesh::buildGraphFromMesh)
        .def("splitMesh", &GraphMesh::splitMesh)
        .def("getNNTrainingParams", &GraphMesh::getNNTrainingParams)
        .def("getCompatibilityConditions", &GraphMesh::getCompatibilityConditions)
        .def("getGeometryPolynomialPoints", &GraphMesh::getGeometryPolynomialPoints)
        .def("getMeshPolynomialPoints", &GraphMesh::getMeshPolynomialPoints)
        .def("getGeometryParametrizations", &GraphMesh::getGeometryParametrizations)
        .def("getMeshParametrizations", &GraphMesh::getMeshParametrizations)
        .def("getInitialDisplacements", &GraphMesh::getInitialDisplacements)
        .def_static("centerPointParametrizations", &GraphMesh::centerPointParametrizations)
        .def_static("centerMeshParametrizations", &GraphMesh::centerMeshParametrizations)
        .def("printMeshGeometry", &GraphMesh::printMeshGeometry)
        .def("buildSplitAndPrintMesh", &GraphMesh::buildSplitAndPrintMesh)
        .def("printGraphState", &GraphMesh::printGraphState)
        .def("printPartGraphState", &GraphMesh::printPartGraphState)
        .def("printCompatibilityConditions", &GraphMesh::printCompatibilityConditions)
        .def("printTrainingParams", &GraphMesh::printTrainingParams)
        .def("printMeshData", &GraphMesh::printMeshData)
        .def("compareMeshParametrizationData", &GraphMesh::compareMeshParametrizationData)
        .def("printMatrixComparison", &GraphMesh::printMatrixComparison)
        .def("buildSplitAndPrintMesh", &GraphMesh::buildSplitAndPrintMesh)
        .def("getFixedDisplacementConditions", &GraphMesh::getFixedDisplacementConditions)
        .def("meshFEMCalculation", &GraphMesh::meshFEMCalculation)
        .def("meshDisplacementVectors", &GraphMesh::meshDisplacementVectors)
        .def("meshEnergy", &GraphMesh::meshEnergy)
        .def("printMeshParamsAndDisplacements", &GraphMesh::printMeshParamsAndDisplacements)
        .def("printMeshParamsAndDisplacementsAndTrueEnergies", &GraphMesh::printMeshParamsAndDisplacementsAndTrueEnergies)
        .def("printMeshParamsAndDisplacementsAndEnergies", &GraphMesh::printMeshParamsAndDisplacementsAndEnergies)
        .def("printMeshPointsAndDisplacements", &GraphMesh::printMeshPointsAndDisplacements)
        .def("printMeshMatricesAndDisplacements", &GraphMesh::printMeshMatricesAndDisplacements)
        .def("printFixedDisplacementConditions", &GraphMesh::printFixedDisplacementConditions)
        .def("printMeshParamsAndEnergies", &GraphMesh::printMeshParamsAndEnergies);
}
cmake_minimum_required(VERSION 3.10)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.25.3.tar.gz"
    SHA1 "0dfbc2cb5c4cf7e83533733bdfd2125ff96680cb"
    LOCAL
)

### Setup CMake Project ###
project(MetalFoams)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find GMSH
find_path(GMSH_INCLUDE_DIR gmsh.h PATHS /usr/local/include)
find_library(GMSH_LIBRARY NAMES gmsh PATHS /usr/local/lib)

# download dependencies
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)

hunter_add_package(ZLIB)

# Collect all source files
set(SOURCES
        src/testFEM.cc
        src/graph_mesh.cc
        src/linear_matrix_computation.cc
        src/parametric_matrix_computation.cc
        src/data_operations.cc
        src/line_mapping.cc
        src/mesh_parametrization.cc
)

# Define executable
add_executable(metal_foam_mesh ${SOURCES})

# Include Directories
target_include_directories(metal_foam_mesh PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GMSH_INCLUDE_DIR})

target_link_libraries(metal_foam_mesh PRIVATE
        ${GMSH_LIBRARY}
        LF::lf.base
        LF::lf.assemble
        LF::lf.fe
        LF::lf.geometry
        LF::lf.io
        LF::lf.mesh
        LF::lf.quad
        LF::lf.refinement
        LF::lf.uscalfe
        )

#file(COPY ${CMAKE_SOURCE_DIR}/python DESTINATION ${CMAKE_BINARY_DIR})
file(GLOB PYTHON_SCRIPTS "python/*.py")
file(COPY ${PYTHON_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/python)

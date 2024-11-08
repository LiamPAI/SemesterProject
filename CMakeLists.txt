cmake_minimum_required(VERSION 3.10)

# Set the correct SDK path
execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Hunter configuration
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.25.3.tar.gz"
    SHA1 "0dfbc2cb5c4cf7e83533733bdfd2125ff96680cb"
    LOCAL
)

### Setup CMake Project ###
project(MetalFoams)

find_package(Python3 3.12 COMPONENTS Interpreter Development REQUIRED)

# Find GMSH
set(GMSH_ROOT "/opt/homebrew/Cellar/gmsh/4.13.1_1")

find_path(GMSH_INCLUDE_DIR gmsh.h PATHS ${GMSH_ROOT}/include)
find_library(GMSH_LIBRARY NAMES gmsh PATHS ${GMSH_ROOT}/lib)

# download dependencies
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)

hunter_add_package(ZLIB)

# Add pybind11
add_subdirectory(pybind11)
set(PYBIND11_INCLUDE_DIRS ${pybind11_INCLUDE_DIRS})

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)

# Collect all source files
set(SOURCES
        src/graph_mesh.cc
        src/linear_matrix_computation.cc
        src/parametric_matrix_computation.cc
        src/data_operations.cc
        src/line_mapping.cc
        src/mesh_parametrization.cc
)

# Define executables
add_executable(metal_foam_mesh ${SOURCES}
        src/testFEM.cc)

# Include Directories for each executable
target_include_directories(metal_foam_mesh PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GMSH_INCLUDE_DIR}
        ${PYBIND11_INCLUDE_DIRS}
)

target_link_libraries(metal_foam_mesh PRIVATE
        LF::lf.base
        LF::lf.assemble
        LF::lf.fe
        LF::lf.geometry
        LF::lf.io
        ${GMSH_LIBRARY}
        LF::lf.mesh
        LF::lf.quad
        LF::lf.refinement
        LF::lf.uscalfe
        )

add_executable(test_mesh_parametrization
        test/test_mesh_parametrization.cc
        ${SOURCES})

target_include_directories(test_mesh_parametrization PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GMSH_INCLUDE_DIR}
        ${PYBIND11_INCLUDE_DIRS}
)

target_link_libraries(test_mesh_parametrization PRIVATE
        LF::lf.base
        LF::lf.assemble
        LF::lf.fe
        LF::lf.geometry
        LF::lf.io
        ${GMSH_LIBRARY}
        LF::lf.mesh
        LF::lf.quad
        LF::lf.refinement
        LF::lf.uscalfe
)

add_executable(test_data_operations
        test/test_data_operations.cc
        ${SOURCES})

target_include_directories(test_data_operations PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GMSH_INCLUDE_DIR}
        ${PYBIND11_INCLUDE_DIRS}
)

target_link_libraries(test_data_operations PRIVATE
        LF::lf.base
        LF::lf.assemble
        LF::lf.fe
        LF::lf.geometry
        LF::lf.io
        ${GMSH_LIBRARY}
        LF::lf.mesh
        LF::lf.quad
        LF::lf.refinement
        LF::lf.uscalfe
)

# Python module
pybind11_add_module(metal_foams_py src/bindings.cc ${SOURCES})
target_include_directories(metal_foams_py PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GMSH_INCLUDE_DIR}
        ${pybind11_INCLUDE_DIRS})
target_link_libraries(metal_foams_py PRIVATE
        pybind11::module
        LF::lf.base
        LF::lf.assemble
        LF::lf.fe
        LF::lf.geometry
        LF::lf.io
        ${GMSH_LIBRARY}
        LF::lf.mesh
        LF::lf.quad
        LF::lf.refinement
        LF::lf.uscalfe
)

# Set the output name of the module
set_target_properties(metal_foams_py PROPERTIES OUTPUT_NAME "metal_foams")

# Install the module
install(TARGETS metal_foams_py
        LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/python
        RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/python)

file(GLOB PYTHON_SCRIPTS "python/*.py")
file(COPY ${PYTHON_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/python)
add_custom_command(TARGET metal_foams_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:metal_foams_py> ${CMAKE_BINARY_DIR}/python)
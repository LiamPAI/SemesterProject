cmake_minimum_required(VERSION 3.10)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.25.3.tar.gz"
    SHA1 "0dfbc2cb5c4cf7e83533733bdfd2125ff96680cb"
    LOCAL
)

### Setup CMake Project ###
project(MetalFoams)

# Find GMSH
find_path(GMSH_INCLUDE_DIR gmsh.h
        PATHS /usr/local/include
        )
find_library(GMSH_LIBRARY
        NAMES gmsh
        PATHS /usr/local/lib
        )

# download dependencies
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)

hunter_add_package(ZLIB)

#set(CPPFLAGS="-I/opt/homebrew/opt/zlib/include")
#set(CMAKE_PREFIX_PATH "/Users/liamcurtis/Documents/ETH Classes/Winter 2024/Semester Project/zlib-1.3.1")
find_package(Torch REQUIRED)
 
# define executables
add_executable(metal_foam_mesh
        testFEM.cc
        linear_matrix_computation.cc
        linear_elasticity_assembler.h
        parametric_matrix_computation.h
        parametric_matrix_computation.cc
        mesh_parametrization.h
        mesh_parametrization.cc
        graph_mesh.h
        graph_mesh.cc line_mapping.h line_mapping.cc data_operations.h data_operations.cc)

add_executable(neural_net
        pinn.cc
        mesh.h
        linear_matrix_computation.cc
        linear_elasticity_assembler.h
        parametric_matrix_computation.h
        parametric_matrix_computation.cc)

target_include_directories(metal_foam_mesh PRIVATE ${GMSH_INCLUDE_DIR})

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

target_link_libraries(neural_net PRIVATE ${TORCH_LIBRARIES}
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
target_include_directories(neural_net PRIVATE ${EIGEN3_INCLUDE_DIRS})

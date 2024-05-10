cmake_minimum_required(VERSION 3.10)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.25.3.tar.gz"
    SHA1 "0dfbc2cb5c4cf7e83533733bdfd2125ff96680cb"
    LOCAL
)

### Setup CMake Project ###
project(MetalFoams)
 
# download dependencies
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)
 
# define executable
add_executable(metal_foam_mesh
        mesh.cc
        linear_matrix_computation.cc
        linear_elasticity_assembler.h
        parametric_matrix_computation.h
        parametric_matrix_computation.cc )

target_link_libraries(metal_foam_mesh
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
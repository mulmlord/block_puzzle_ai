cmake_minimum_required(VERSION 3.15...4.4)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BPA_NATIVE "build with -march=native" ON)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_engine NB_STATIC cpp/bindings.cpp)
# The spawn level must round exactly like float32 in the game: no -ffast-math, no FMA contraction.
target_compile_options(_engine PRIVATE -O3 -ffp-contract=off)
if(BPA_NATIVE)
  target_compile_options(_engine PRIVATE -march=native)
endif()
install(TARGETS _engine LIBRARY DESTINATION block_puzzle_ai)

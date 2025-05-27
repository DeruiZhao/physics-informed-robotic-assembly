cmake_minimum_required(VERSION 3.14)
project(ikfast_ur5e_robotiq)

find_package(pybind11 REQUIRED)

find_package(LAPACK REQUIRED)

pybind11_add_module(ikfast_ur5e_robotiq
	ikfast_pybind_ur5e_robotiq.cpp
	ikfast_ur5e_robotiq.cpp
)

target_link_libraries(ikfast_ur5e_robotiq PRIVATE ${LAPACK_LIBRARIES})
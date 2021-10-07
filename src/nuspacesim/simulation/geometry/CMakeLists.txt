cmake_minimum_required(VERSION 3.12)

project(nssgeometry)

find_package(pybind11 REQUIRED)

pybind11_add_module(nssgeometry src/nssgeometry.cpp)

# find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

# include_directories(${HDF5_C_INCLUDE_DIRS} ${HDF5_CXX_INCLUDE_DIRS})

# target_link_libraries(nssgeometry PUBLIC ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${HDF5_C_HL_LIBRARIES} ${HDF5_CXX_HL_LIBRARIES})

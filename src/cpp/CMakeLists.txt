add_library(qre_engine INTERFACE)
target_include_directories(qre_engine INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(qre_engine INTERFACE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_CUDA)
    target_link_libraries(qre_engine INTERFACE CUDA::cudart)
endif()

# pybind11 module (only if pybind11 is available)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(qre_python bindings/pybind_module.cpp)
    target_link_libraries(qre_python PRIVATE qre_engine)
endif()

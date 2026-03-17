# Core engine library
add_library(quant_engine
    src/risk_metrics.cpp
    src/sobol.cpp
)
target_include_directories(quant_engine PUBLIC include/)
target_link_libraries(quant_engine PUBLIC OpenMP::OpenMP_CXX)

if(ENABLE_CUDA)
    target_compile_definitions(quant_engine PUBLIC ENABLE_CUDA)
    target_link_libraries(quant_engine PUBLIC CUDA::cudart)
endif()

# Python bindings (only if pybind11 found)
if(pybind11_FOUND)
    pybind11_add_module(quant_engine_py bindings/pybind_module.cpp)
    target_link_libraries(quant_engine_py PRIVATE quant_engine)
endif()

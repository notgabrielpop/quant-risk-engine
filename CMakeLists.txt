cmake_minimum_required(VERSION 3.20)
project(quant_risk_engine
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Quantitative risk engine for multi-asset portfolio simulation"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options ---
option(ENABLE_CUDA "Enable CUDA GPU backend" OFF)
option(ENABLE_TESTS "Build unit tests" ON)

# --- OpenMP ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found — CPU engine will run single-threaded")
endif()

# --- CUDA (optional) ---
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
    add_compile_definitions(QRE_CUDA_ENABLED)
endif()

# --- Engine library ---
add_subdirectory(src/cpp)

# --- Tests ---
if(ENABLE_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
    add_subdirectory(tests/cpp)
endif()

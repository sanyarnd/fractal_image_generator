cmake_minimum_required(VERSION 3.13.1)
project(fractal CUDA CXX)

# debugging
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# packages
find_package(Threads)
find_package(TIFF REQUIRED)
# Linking to CUDA libraries from non-.cu files
# not yet fixed in 3.13.2
find_package(CUDA REQUIRED)

# main target
add_executable(fractal
        include/image.hpp
        src/image.cpp
        include/params.hpp
        src/params.cpp
        src/main.cpp
        include/utils.hpp
        src/utils.cpp
        src/rand_gen_std.cpp
        include/rand_gen_std.hpp
        include/rand_gen_thurst.hpp
        src/rand_gen_thurst.cu
        include/renderer.hpp
        src/renderer.cpp
        include/renderer_cuda.hpp
        src/renderer_cuda.cu
        include/fractal_data.hpp
        src/fractal_data.cpp
        include/transforms.hpp
        src/transforms.cu)

set_target_properties(fractal PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"

        CMAKE_CUDA_STANDARD 17
        CMAKE_CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        # CUDA_PTX_COMPILATION ON

        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS OFF)
target_include_directories(fractal
        PRIVATE 3rdparty
        PRIVATE include
        PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_options(fractal
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wimplicit-fallthrough -Wduplicated-cond -Wduplicated-branches
        -Wlogical-op -Wrestrict -Wnull-dereference -Wuseless-cast -Wdouble-promotion -Wformat=2 -pedantic-errors>
        # any C++ compiler flag needs to be replaced to -Xcompiler=<flag> in CUDA case
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_50,code=sm_50 --expt-relaxed-constexpr>)
target_link_libraries(fractal
        PRIVATE
        TIFF::TIFF
        Threads::Threads
        ${CUDA_LIBRARIES}
        ${CUDA_CURAND_LIBRARIES}
        ${CUDA_cudadevrt_LIBRARY})
# doesn't work (yet)
# see https://gitlab.kitware.com/cmake/cmake/issues/18265
# target_link_options(fractal
#         PRIVATE
##         $<$<COMPILE_LANGUAGE:CXX>:>
#         $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_50>)
list(APPEND CMAKE_CUDA_FLAGS "-gencode=arch=compute_50,code=sm_50")


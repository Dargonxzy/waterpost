cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

# 水域后处理（blockgs/water_postprocessing 精细层流程）的 C++ / CUDA 版：输入 BIN LOD + 语义掩码 + 分块相机（或位姿 JSON），
# 输出补水后的 LOD。基础能力来自 ai3d 共用库：错误类型（ai3d-core）、COLMAP 读取（ai3d-io）、CUDA 编译设置（ai3d-gpu）。
project(ai3d_waterpost VERSION 0.1.0 LANGUAGES CXX CUDA)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
include(GNUInstallDirs)

option(AI3D_WATERPOST_USE_SIBLINGS "使用相邻目录的 ai3d-core / ai3d-platform / ai3d-io / ai3d-gpu 源码联调" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(AI3D_WATERPOST_USE_SIBLINGS)
    include(cmake/siblings.cmake)
endif()
if(NOT TARGET ai3d_core::core)
    find_package(ai3d_core 0.3 CONFIG REQUIRED)
endif()
if(NOT TARGET ai3d_io::io)
    find_package(ai3d_io 0.6 CONFIG REQUIRED)
endif()
find_package(CUDAToolkit REQUIRED)
if(NOT TARGET ai3d_gpu::cuda AND AI3D_WATERPOST_USE_SIBLINGS)
    set(AI3D_GPU_WITH_CUDA ON)
    set(AI3D_GPU_WITH_VULKAN OFF)
    set(AI3D_GPU_INSTALL OFF)
    add_subdirectory("${AI3D_WATERPOST_SIBLING_ROOT}/ai3d-gpu" "siblings/gpu")
endif()
if(NOT COMMAND ai3d_add_cuda_library)
    find_package(ai3d_gpu CONFIG REQUIRED)
endif()
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(zstd CONFIG REQUIRED)

function(ai3d_waterpost_compile_settings target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /W3 /permissive- /bigobj>)
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES)
    else()
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")
    endif()
endfunction()

ai3d_add_cuda_library(ai3d_waterpost_cuda STATIC src/waterpost/gpu.cu)
target_include_directories(ai3d_waterpost_cuda PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/waterpost")
target_compile_definitions(ai3d_waterpost_cuda PRIVATE NOMINMAX _USE_MATH_DEFINES)
target_link_libraries(ai3d_waterpost_cuda PRIVATE CUDA::cudart)

add_library(ai3d_waterpost_lib STATIC
    src/waterpost/apply_common.cpp
    src/waterpost/apply_vote.cpp
    src/waterpost/apply_write.cpp
    src/waterpost/camera.cpp
    src/waterpost/detect.cpp
    src/waterpost/evaluate.cpp
    src/waterpost/fs_util.cpp
    src/waterpost/geo.cpp
    src/waterpost/json_util.cpp
    src/waterpost/lod.cpp
    src/waterpost/masks.cpp
    src/waterpost/npz.cpp
    src/waterpost/parallel.cpp
    src/waterpost/project.cpp
    src/waterpost/reconcile.cpp
    src/waterpost/scene.cpp
)
ai3d_waterpost_compile_settings(ai3d_waterpost_lib)
target_include_directories(ai3d_waterpost_lib
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src/waterpost"
)
target_link_libraries(ai3d_waterpost_lib
    PUBLIC
        ai3d_core::core
        nlohmann_json::nlohmann_json
        opencv_core
        TBB::tbb
    PRIVATE
        ai3d_io::io
        ai3d_waterpost_cuda
        CUDA::cudart
        opencv_imgproc
        opencv_imgcodecs
        Eigen3::Eigen
        ZLIB::ZLIB
        $<IF:$<TARGET_EXISTS:zstd::libzstd>,zstd::libzstd,$<IF:$<TARGET_EXISTS:zstd::libzstd_static>,zstd::libzstd_static,zstd::libzstd_shared>>
)

add_executable(ai3d_waterpost apps/ai3d_waterpost.cpp)
ai3d_waterpost_compile_settings(ai3d_waterpost)
target_link_libraries(ai3d_waterpost PRIVATE ai3d_waterpost_lib)
if(WIN32)
    target_link_libraries(ai3d_waterpost PRIVATE shell32)
endif()

# 单元测试（纯函数）：ctest，或直接运行 ai3d_waterpost_tests.exe。
enable_testing()
add_executable(ai3d_waterpost_tests tests/test_main.cpp)
ai3d_waterpost_compile_settings(ai3d_waterpost_tests)
target_link_libraries(ai3d_waterpost_tests PRIVATE ai3d_waterpost_lib)
add_test(NAME ai3d_waterpost_tests COMMAND ai3d_waterpost_tests)

option(AI3D_WATERPOST_GPU_SMOKE_TEST "Build and register the CUDA deployment smoke test" OFF)
if(AI3D_WATERPOST_GPU_SMOKE_TEST)
    add_executable(ai3d_waterpost_gpu_smoke tests/gpu_smoke.cpp)
    ai3d_waterpost_compile_settings(ai3d_waterpost_gpu_smoke)
    target_link_libraries(ai3d_waterpost_gpu_smoke PRIVATE ai3d_waterpost_lib)
    add_test(NAME ai3d_waterpost_gpu_smoke COMMAND ai3d_waterpost_gpu_smoke)
    set_tests_properties(ai3d_waterpost_gpu_smoke PROPERTIES LABELS gpu)
endif()

install(TARGETS ai3d_waterpost RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

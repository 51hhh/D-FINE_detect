cmake_minimum_required(VERSION 3.16)
project(zed_ds_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Git commit hash 集成 ──
execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT GIT_HASH)
  set(GIT_HASH "unknown")
endif()
add_definitions(-DZED_DS_GIT_HASH="${GIT_HASH}")
add_definitions(-DZED_DS_APP_VERSION="0.2.0")

# ── 依赖 ──
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)
find_package(yaml-cpp REQUIRED)

set(DEEPSTREAM_INCLUDE_DIR /opt/nvidia/deepstream/deepstream/sources/includes)
set(DEEPSTREAM_LIB_DIR /opt/nvidia/deepstream/deepstream/lib)
set(CUDA_INCLUDE_DIR /usr/local/cuda/include)

# ── 主程序 ──
add_executable(zed_ds_app
  src/main.cpp
  src/config.cpp
  src/pipeline_builder.cpp
  src/depth_estimator.cpp
  src/output_writer.cpp
  src/perf_monitor.cpp
  src/ball_tracker.cpp
  src/trajectory_estimator.cpp
  src/landing_predictor.cpp
)

# ── D-FINE 自定义推理库 ──
add_library(nvdsinfer_custom_dfine SHARED
  plugins/nvdsinfer_custom_dfine.cpp
)

target_include_directories(zed_ds_app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GST_INCLUDE_DIRS}
  ${DEEPSTREAM_INCLUDE_DIR}
)
target_include_directories(nvdsinfer_custom_dfine PRIVATE ${DEEPSTREAM_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})

target_link_directories(zed_ds_app PRIVATE ${GST_LIBRARY_DIRS})
target_link_directories(zed_ds_app PRIVATE ${DEEPSTREAM_LIB_DIR})

target_link_libraries(zed_ds_app PRIVATE
  ${GST_LIBRARIES}
  nvdsgst_meta
  nvds_meta
  yaml-cpp
)

target_compile_options(zed_ds_app PRIVATE ${GST_CFLAGS_OTHER} -Wall -Wextra -Wno-unused-parameter)
target_compile_options(nvdsinfer_custom_dfine PRIVATE -Wall -Wextra -Wno-unused-parameter)

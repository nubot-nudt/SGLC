cmake_minimum_required(VERSION 3.0)
project(SGLC)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CONFIGURATION_TYPES Debug RelWithDebInfo Release)
set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} -Wall -O3 -march=native")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -pg -march=native")
set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG    "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pg -march=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(./)
add_subdirectory(semgraph)

add_executable(eval_lcd_pair src/eval_lcd_pair.cpp)
target_link_libraries(eval_lcd_pair  semgraph )
add_executable(eval_lcd_seq src/eval_lcd_seq.cpp)
target_link_libraries(eval_lcd_seq  semgraph )

add_executable(eval_loop_poses src/eval_loop_poses.cpp)
target_link_libraries(eval_loop_poses  semgraph )

add_executable(eval_overlap src/eval_overlap.cpp)
target_compile_options(eval_overlap PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(eval_overlap semgraph  ${OpenMP_CXX_FLAGS})

# ford campus
add_executable(eval_overlap_ford_campus src/eval_overlap_ford_campus.cpp)
target_compile_options(eval_overlap_ford_campus PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(eval_overlap_ford_campus semgraph  ${OpenMP_CXX_FLAGS})

# apollo
add_executable(eval_overlap_apollo src/eval_overlap_apollo.cpp)
target_compile_options(eval_overlap_apollo PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(eval_overlap_apollo semgraph  ${OpenMP_CXX_FLAGS})


#kitti 360
add_executable(eval_lcd_seq_kitti360 src/eval_lcd_seq_kitti360.cpp)
target_link_libraries(eval_lcd_seq_kitti360  semgraph )
add_executable(eval_loop_poses_kitti360 src/eval_loop_poses_kitti360.cpp)
target_link_libraries(eval_loop_poses_kitti360  semgraph )


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
if (NOT yaml-cpp_FOUND)
find_package (yaml-cpp REQUIRED)
endif()
if (NOT PCL_FOUND)
    find_package(PCL 1.10 REQUIRED)
endif ()
if (NOT OPENCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
endif ()
find_package(Sophus REQUIRED)
# find_package ( TBB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIRS})
file(GLOB SRC_LIST *.cpp)
add_library(semgraph SHARED ${SRC_LIST})
target_include_directories(semgraph PUBLIC 
${PCL_INCLUDE_DIRS} 
${YAML_CPP_INCLUDE_DIR} 
./
)
if (NOT YAML_CPP_LIBRARIES)
    set(YAML_CPP_LIBRARIES yaml-cpp)
endif ()

option(BUILD_SHARED_LIBS OFF)
option(TBBMALLOC_BUILD OFF)
option(TBB_EXAMPLES OFF)
option(TBB_STRICT OFF)
option(TBB_TEST OFF)

add_subdirectory(3dParty/tsl_robin/robin-map)
add_subdirectory(3dParty/oneTBB)
target_link_libraries(semgraph PUBLIC
        ${PCL_LIBRARIES}
	    ${YAML_CPP_LIBRARIES}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBRARIES}
        ${FMT_LIBRARIES}
        ${CERES_LIBRARIES}
        tsl::robin_map
        TBB::tbb
        Sophus::Sophus
        )

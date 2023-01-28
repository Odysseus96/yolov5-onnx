cmake_minimum_required(VERSION 3.0)
project(main)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

link_directories(${OpenCV_LIB_DIRS})

add_executable(main main.cpp src/detector.cpp src/common.cpp)

target_link_libraries(main ${OpenCV_LIBS})

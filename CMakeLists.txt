cmake_minimum_required (VERSION 2.8)
project(bench)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)

find_package(OpenCL REQUIRED)
message(STATUS "OpenCL found: ${OpenCL_FOUND}")
message(STATUS "OpenCL include dir: ${OpenCL_INCLUDE_DIR}")
message(STATUS "OpenCL library: ${OpenCL_LIBRARY}")
message(STATUS "OpenCL version: ${OpenCL_VERSION_STRING}")

set(bench_VERSION_MAJOR 0)
set(bench_VERSION_MINOR 1)

# USE_CL2_HEADERS only required on linux because most distributions do not use ocl-icd provided by khronos
# the package used on e.g. ubuntu 16.04 provides cl2.hpp instead of cl.hpp
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    if (${OpenCL_VERSION_STRING} STREQUAL "2.0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CL2_HEADERS=1")
    endif()
endif()

# set c++ standard if compiled with another compiler than MSVC
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Weffc++ -std=c++11")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Weffc++ -std=c++11")
endif()

# disable warnings in visual studio about deprecated functions in cl.hpp
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")
endif()

add_subdirectory("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/benchmarks")

set(BENCH_TARGET_NAME bench)
add_executable(${BENCH_TARGET_NAME} ${SOURCE} ${HEADERS})
target_include_directories(${BENCH_TARGET_NAME} SYSTEM PRIVATE ${OpenCL_INCLUDE_DIR})
target_link_libraries(${BENCH_TARGET_NAME} PRIVATE ${OpenCL_LIBRARY})

install (TARGETS ${BENCH_TARGET_NAME} DESTINATION bin)

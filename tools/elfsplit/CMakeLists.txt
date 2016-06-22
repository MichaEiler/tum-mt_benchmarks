cmake_minimum_required (VERSION 2.8)
project(elfsplit)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Weffc++ -std=c++11")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Weffc++ -std=c++11")
endif()

set(ELFSPLIT_TARGET_NAME elfsplit)
add_executable(${ELFSPLIT_TARGET_NAME} elfsplit.cpp)

install (TARGETS ${ELFSPLIT_TARGET_NAME} DESTINATION bin)

cmake_minimum_required (VERSION 2.8)
project(elfunpack)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Weffc++ -std=c++11")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Weffc++ -std=c++11")
endif()

set(ELFUNPACK_TARGET_NAME elfunpack)
add_executable(${ELFUNPACK_TARGET_NAME} elfunpack.cpp)

target_link_libraries(${ELFUNPACK_TARGET_NAME} PRIVATE bfd)

install (TARGETS ${ELFUNPACK_TARGET_NAME} DESTINATION bin)

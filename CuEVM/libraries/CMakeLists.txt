cmake_minimum_required(VERSION 3.22)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory(CuBigInt/CuBigInt)
add_subdirectory(CGBN/CGBN)
add_subdirectory(CuCrypto/CuCrypto)
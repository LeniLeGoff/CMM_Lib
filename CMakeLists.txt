cmake_minimum_required(VERSION 2.8)
project(iagmm_lib)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include_directories(/usr/local/include/
                    /usr/include/eigen2
                    /usr/local/include/SFML
                    include)


add_library(iagmm src/gmm.cpp src/component.cpp)
target_link_libraries(iagmm tbb)


add_executable(test src/test.cpp)
target_link_libraries(test iagmm tbb sfml-window sfml-system sfml-graphics)



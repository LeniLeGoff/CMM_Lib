cmake_minimum_required(VERSION 2.8)
project(iagmm)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include_directories(/usr/local/include/
                    /usr/include/eigen3
                    /usr/local/include/SFML
                    include/)


FILE(GLOB_RECURSE HEADFILES "include/iagmm/*.hpp")
FILE(GLOB_RECURSE SRCFILES "src/*cpp" "include/iagmm/mcs_fct.hpp")

add_custom_target(headers SOURCES ${HEADFILES})

install(DIRECTORY include  DESTINATION include/${PROJECT_NAME})



add_library(iagmm ${SRCFILES})
target_link_libraries(iagmm tbb boost_serialization boost_chrono boost_system)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)


#add_executable(load_archive tools/load_archive.cpp)
#target_link_libraries(load_archive iagmm )

#add_executable(test_1 test/test.cpp)
#target_link_libraries(test_1 iagmm tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

#add_executable(test_serial test/test_serial.cpp)
#target_link_libraries(test_serial iagmm sfml-window sfml-system sfml-graphics boost_serialization)




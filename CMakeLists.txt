cmake_minimum_required(VERSION 2.8)
project(iagmm)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include_directories(/usr/local/include/
                    /usr/include/eigen2
                    /usr/local/include/SFML
                    include)

FILE(GLOB_RECURSE HEADFILES "include/*.hpp")
add_custom_target(headers SOURCES ${HEADFILES})

install(DIRECTORY include/iagmm/  DESTINATION /usr/local/include/${PROJECT_NAME})



add_library(iagmm src/gmm.cpp src/component.cpp src/nnmap.cpp)
target_link_libraries(iagmm tbb boost_serialization)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION /usr/local/lib
    LIBRARY DESTINATION /usr/local/lib)


#add_executable(test_1 src/test.cpp)
#target_link_libraries(test_1 iagmm tbb sfml-window sfml-system sfml-graphics boost_serialization)

#add_executable(test_serial src/test_serial.cpp)
#target_link_libraries(test_serial iagmm sfml-window sfml-system sfml-graphics boost_serialization)




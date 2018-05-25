cmake_minimum_required(VERSION 2.8)
project(iagmm)



if(VERBOSE)
    set(CMAKE_CXX_FLAGS "-std=c++11 -O3  -DVERBOSE ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-std=c++11 -O3 ${CMAKE_CXX_FLAGS}")
endif()

include_directories(/usr/local/include/
                    /usr/include/eigen3
                    /usr/local/include/SFML
                    include/)


FILE(GLOB_RECURSE HEADFILES "include/iagmm/*.hpp")
FILE(GLOB_RECURSE SRCFILES "src/*cpp" "include/iagmm/mcs_fct.hpp")

add_custom_target(headers SOURCES ${HEADFILES})


install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)



add_library(iagmm SHARED ${SRCFILES})
target_link_libraries(iagmm tbb boost_serialization boost_chrono boost_filesystem yaml-cpp)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)


#tools

#add_executable(load_archive tools/load_archive.cpp)
#target_link_libraries(load_archive iagmm )

add_executable(gen_archive_stat tools/gen_archive_stat.cpp)
target_link_libraries(gen_archive_stat iagmm yaml-cpp boost_serialization )

add_executable(gmm_loglikelihood tools/gmm_loglikelihood.cpp)
target_link_libraries(gmm_loglikelihood iagmm yaml-cpp boost_serialization boost_system boost_filesystem)

add_executable(batch_train_gmm tools/batch_train_gmm.cpp)
target_link_libraries(batch_train_gmm iagmm yaml-cpp boost_serialization boost_system)

add_executable(online_train_gmm tools/online_train_gmm.cpp)
target_link_libraries(online_train_gmm iagmm yaml-cpp boost_serialization boost_system)


####

#test

if(TEST)

add_executable(test_1 test/test.cpp)
target_link_libraries(test_1 iagmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

add_executable(test_batch test/test_batch.cpp)
target_link_libraries(test_batch iagmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

add_executable(test_false_samples test/test_false_samples.cpp)
target_link_libraries(test_false_samples iagmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)


add_executable(test_mvn test/test_mvn.cpp)
target_link_libraries(test_mvn iagmm tbb)

add_executable(test_em test/test_em.cpp)
target_link_libraries(test_em iagmm tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

add_executable(test_serial test/test_serial.cpp)
target_link_libraries(test_serial iagmm sfml-window sfml-system sfml-graphics boost_serialization)

endif()

####


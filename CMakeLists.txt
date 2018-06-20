cmake_minimum_required(VERSION 2.8)
project(iagmm)


set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

if(DEBUG)
    set(CMAKE_CXX_FLAGS "-DCMAKE_BUILD_TYPE=Debug -O0 ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-DCMAKE_BUILD_TYPE=Release -O3 ${CMAKE_CXX_FLAGS}")
endif()

if(VERBOSE)
    set(CMAKE_CXX_FLAGS "-DVERBOSE ${CMAKE_CXX_FLAGS}")
endif()

find_package(OpenCV)

include_directories(/usr/local/include/
                    /usr/include/eigen3
                    /usr/local/include/SFML
                    include/)


FILE(GLOB_RECURSE HEADFILES "include/iagmm/*.hpp")
FILE(GLOB_RECURSE SRCFILES "src/*cpp" "include/iagmm/mcs_fct.hpp")

add_custom_target(headers SOURCES ${HEADFILES})


install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)



add_library(iagmm SHARED ${SRCFILES})
target_link_libraries(iagmm tbb boost_serialization boost_chrono boost_filesystem)

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

#add_executable(batch_train_gmm tools/batch_train_gmm.cpp)
#target_link_libraries(batch_train_gmm iagmm yaml-cpp boost_serialization boost_system)

add_executable(online_train_gmm tools/online_train_gmm.cpp)
target_link_libraries(online_train_gmm iagmm yaml-cpp boost_serialization boost_system)

add_executable(yml_to_libsvm tools/yml_to_libsvm.cpp)
target_link_libraries(yml_to_libsvm iagmm yaml-cpp boost_serialization boost_system)

add_executable(yml_to_data_labels tools/yml_to_data_labels.cpp)
target_link_libraries(yml_to_data_labels iagmm yaml-cpp boost_serialization boost_system)

add_executable(mnist_batch example/mnist_batch.cpp)
target_link_libraries(mnist_batch iagmm tbb yaml-cpp)


#add_executable(mnist_online example/mnist_online.cpp)
#target_link_libraries(mnist_online iagmm tbb)

####

#test
if(TEST)
#add_executable(test_1 test/test.cpp)
#target_link_libraries(test_1 iagmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

#add_executable(test_batch test/test_batch.cpp)
#target_link_libraries(test_batch iagmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

#add_executable(test_false_samples test/test_false_samples.cpp)
#target_link_libraries(test_false_samples iagmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

add_executable(multiclass_test test/multiclass_test.cpp)
target_link_libraries(multiclass_test iagmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)


add_executable(incrgmm_test test/incrgmm_test.cpp)
target_link_libraries(incrgmm_test iagmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

add_executable(mnist_test test/mnist_test.cpp)
target_link_libraries(mnist_test tbb  ${OpenCV_LIBRARIES})

#add_executable(mcs_test test/mcs_test.cpp)
#target_link_libraries(mcs_test iagmm  yaml-cpp.so)

#add_executable(test_mvn test/test_mvn.cpp)
#target_link_libraries(test_mvn iagmm tbb)

#add_executable(test_em test/test_em.cpp)
#target_link_libraries(test_em iagmm tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

#add_executable(test_serial test/test_serial.cpp)
#target_link_libraries(test_serial iagmm sfml-window sfml-system sfml-graphics boost_serialization)
endif()
####


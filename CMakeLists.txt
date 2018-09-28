cmake_minimum_required(VERSION 2.8)
project(iagmm)

# The syntax set...CACHE used below chooses a value while letting caller override it.

# Ref https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ standard to be used")
# It would be cleaner to declare per-target target_compile_features
# and https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html


# Eigen runs much much much faster when compiled in -O3.
# The line below cleanly ensure release builds have O3 optimization level.
add_compile_options("$<$<CONFIG:RELEASE>:-O3>")

if(VERBOSE)
    set(CMAKE_CXX_FLAGS "-DVERBOSE ${CMAKE_CXX_FLAGS}")
endif()


include_directories(/usr/local/include/
                    /usr/include/eigen3
                    /usr/local/include/SFML
                    include/
/home/le_goff/Libraries/eigen-3.3.4)


FILE(GLOB_RECURSE HEADFILES "include/iagmm/*.hpp")
FILE(GLOB_RECURSE SRCFILES "src/*cpp" "include/iagmm/mcs_fct.hpp")

add_custom_target(headers SOURCES ${HEADFILES})


install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)

if(NO_PARALLEL)
    add_definitions(-DNO_PARALLEL)
    add_library(iagmm SHARED ${SRCFILES})
    target_link_libraries(iagmm boost_serialization boost_chrono boost_filesystem)
else()
    set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
    add_library(iagmm SHARED ${SRCFILES})
    target_link_libraries(iagmm tbb boost_serialization boost_chrono boost_filesystem)
endif()



# The EXPORT option in install() and the install(EXPORT...) make iagmm
# library available to other projects, ref
# https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/Exporting-and-Importing-Targets
install(TARGETS ${PROJECT_NAME} EXPORT iagmm-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

install(EXPORT iagmm-targets DESTINATION lib/iagmm)

# Caller CMake project can use:
# add_library(iagmm SHARED IMPORTED)
# set_property(TARGET iagmm PROPERTY IMPORTED_LOCATION "${IAGMM_INSTALL_TREE}/lib/libiagmm.so" )
# and be called like this:
# cmake -DIAGMM_INSTALL_TREE=/some/path/ ...
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

#examples
add_executable(mnist_batch example/mnist_batch.cpp)
target_link_libraries(mnist_batch iagmm tbb yaml-cpp)

add_executable(eggholder_two_class_problem example/eggholder_two_class_problem.cpp)
target_link_libraries(eggholder_two_class_problem iagmm tbb yaml-cpp)

####

# if(TEST) below is deprecated because CMake 3.10.2 introduced a TEST operator... cmake  --help-policy CMP0064
if(POLICY CMP0064)
  cmake_policy(SET CMP0064 OLD)
endif()

#test
if(TEST)
find_package(OpenCV)

add_executable(test_1 test/test.cpp)
target_link_libraries(test_1 iagmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

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


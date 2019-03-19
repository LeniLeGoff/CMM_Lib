cmake_minimum_required(VERSION 2.8)
project(cmm)

# The syntax set...CACHE used below chooses a value while letting caller override it.

# Ref https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html
# set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ standard to be used")
# It would be cleaner to declare per-target target_compile_features
# and https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html


# Eigen runs much much much faster when compiled in -O3.
# The line below cleanly ensure release builds have O3 optimization level.
add_compile_options("$<$<CONFIG:RELEASE>:-O3>")


if(CMAKE_VERSION VERSION_LESS "3.1")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options("-std=c++11")
  endif()
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

option(VERBOSE_RUNTIME "Set true to create a build with more run-time output." FALSE)

# Since in CMake-generated makefiles, make VERBOSE=1 means verbose
# *build*, renamed it to VERBOSE_RUNTIME.  In C++ code it remains VERBOSE.
if(VERBOSE_RUNTIME)
  add_definitions("-DVERBOSE")
endif()

# FIXME could find package eigen and declare that it's necessary only for tests.
# FIXME could find package eigen.
# FIXME could find boost components: boost_serialization boost_chrono boost_filesystem
# FIXME could find package sfml and declare that it's necessary only for tests.
include_directories(/usr/local/include/
                    /usr/include/eigen3
                    /usr/local/include/SFML
                    include/
/home/le_goff/Libraries/eigen-3.3.4)


FILE(GLOB_RECURSE HEADFILES "include/cmm/*.hpp")
FILE(GLOB_RECURSE SRCFILES "src/*cpp" "include/cmm/mcs_fct.hpp")

add_custom_target(headers SOURCES ${HEADFILES})


install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(DEFAULT_NO_PARALLEL TRUE)
else(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(DEFAULT_NO_PARALLEL FALSE)
endif(CMAKE_BUILD_TYPE STREQUAL Debug)

option(NO_PARALLEL "If enabled, force sequential execution.  Very useful for debugging.  Enabled by default on debug build." ${DEFAULT_NO_PARALLEL})

#set(NO_PARALLEL TRUE CACHE BOOL "If enabled, force sequential execution.  Very useful for debugging.  Enabled by default on debug build." )



add_library(cmm SHARED ${SRCFILES})

if (NO_PARALLEL)
  add_definitions("-DNO_PARALLEL")
else (NO_PARALLEL)
  set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
  target_link_libraries(cmm tbb)
endif(NO_PARALLEL) 

target_link_libraries(cmm boost_serialization boost_chrono boost_filesystem)


# The EXPORT option in install() and the install(EXPORT...) make cmm
# library available to other projects, ref
# https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/Exporting-and-Importing-Targets
install(TARGETS ${PROJECT_NAME} EXPORT cmm-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

install(EXPORT cmm-targets DESTINATION lib/cmm)

# Caller CMake project can use:
# add_library(cmm SHARED IMPORTED)
# set_property(TARGET cmm PROPERTY IMPORTED_LOCATION "${cmm_INSTALL_TREE}/lib/libcmm.so" )
# and be called like this:
# cmake -Dcmm_INSTALL_TREE=/some/path/ ...
#tools

#add_executable(load_archive tools/load_archive.cpp)iagmm
#target_link_libraries(load_archive cmm )

add_executable(gen_archive_stat tools/gen_archive_stat.cpp)
target_link_libraries(gen_archive_stat cmm yaml-cpp boost_serialization )

add_executable(gmm_loglikelihood tools/gmm_loglikelihood.cpp)
target_link_libraries(gmm_loglikelihood cmm yaml-cpp boost_serialization boost_system boost_filesystem)

#add_executable(batch_train_gmm tools/batch_train_gmm.cpp)
#target_link_libraries(batch_train_gmm cmm yaml-cpp boost_serialization boost_system)

add_executable(online_train_gmm tools/online_train_gmm.cpp)
target_link_libraries(online_train_gmm cmm yaml-cpp boost_serialization boost_system)

add_executable(yml_to_libsvm tools/yml_to_libsvm.cpp)
target_link_libraries(yml_to_libsvm cmm yaml-cpp boost_serialization boost_system)

add_executable(yml_to_data_labels tools/yml_to_data_labels.cpp)
target_link_libraries(yml_to_data_labels cmm yaml-cpp boost_serialization boost_system)

#examples
add_executable(mnist_batch example/mnist_batch.cpp)
target_link_libraries(mnist_batch cmm tbb yaml-cpp)

add_executable(eggholder_two_class_problem example/eggholder_two_class_problem.cpp)
target_link_libraries(eggholder_two_class_problem cmm tbb yaml-cpp)

####

option(BUILD_TESTS "Set true to compile some test executables." FALSE)

if(BUILD_TESTS)
find_package(OpenCV)

add_executable(test_1 test/test.cpp)
target_link_libraries(test_1 cmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

#add_executable(test_batch test/test_batch.cpp)
#target_link_libraries(test_batch cmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

#add_executable(test_false_samples test/test_false_samples.cpp)
#target_link_libraries(test_false_samples cmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

add_executable(multiclass_test test/multiclass_test.cpp)
target_link_libraries(multiclass_test cmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)


add_executable(incrgmm_test test/incrgmm_test.cpp)
target_link_libraries(incrgmm_test cmm yaml-cpp tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

add_executable(mnist_test test/mnist_test.cpp)
target_link_libraries(mnist_test tbb  ${OpenCV_LIBRARIES})

#add_executable(mcs_test test/mcs_test.cpp)
#target_link_libraries(mcs_test cmm  yaml-cpp.so)

#add_executable(test_mvn test/test_mvn.cpp)
#target_link_libraries(test_mvn cmm tbb)

#add_executable(test_em test/test_em.cpp)
#target_link_libraries(test_em cmm tbb sfml-window sfml-system sfml-graphics boost_serialization boost_system boost_chrono)

#add_executable(test_serial test/test_serial.cpp)
#target_link_libraries(test_serial cmm sfml-window sfml-system sfml-graphics boost_serialization)
endif()
####

message(STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
message(STATUS "VERBOSE_RUNTIME is ${VERBOSE_RUNTIME}")
message(STATUS "NO_PARALLEL is ${NO_PARALLEL}")
message(STATUS "BUILD_TESTS is ${BUILD_TESTS}")


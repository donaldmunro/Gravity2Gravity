cmake_minimum_required(VERSION 3.5)
project(GravityPose CXX)

set(CMAKE_CXX_STANDARD 17) # 14 and possibly 11 (g++ has an experimental/filesystem) should work
set(CXX_STANDARD_REQUIRED ON)

message(STATUS "CXX compiler ID: '${CMAKE_CXX_COMPILER_ID}'")
message(STATUS "CXX compiler Version: '${CMAKE_CXX_COMPILER_VERSION}'")
message(STATUS "Cxx compiler path:" ${CMAKE_CXX_COMPILER})

set(INCLUDES "${PROJECT_SOURCE_DIR}/src")
set(LIBS m z)
set(FLAGS "")

#include (CheckIncludeFiles)
#include(CheckIncludeFileCXX)
#check_include_file_cxx(filesystem HAVE_STD_FILESYSTEM)
#check_include_file_cxx(experimental/filesystem HAVE_STD_EXPERIMENTAL_FILESYSTEM)
#MESSAGE(STATUS "HAVE_STD_FILESYSTEM " ${HAVE_STD_FILESYSTEM})
#MESSAGE(STATUS "HAVE_STD_EXPERIMENTAL_FILESYSTEM " ${HAVE_STD_EXPERIMENTAL_FILESYSTEM})
#if (HAVE_STD_FILESYSTEM)
#   MESSAGE(STATUS "Using include/filesystem")
#   list(APPEND FLAGS "-DSTD_FILESYSTEM")
#   list(APPEND LIBS stdc++fs)
#elseif(HAVE_STD_EXPERIMENTAL_FILESYSTEM)
#   list(APPEND FLAGS "-DFILESYSTEM_EXPERIMENTAL")
#   list(APPEND LIBS stdc++fs)
#   MESSAGE(STATUS "Using include/experimental/filesystem")
#else()
#   FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem)
#   list(APPEND FLAGS "-DFILESYSTEM_BOOST")
#   list(APPEND INCLUDES ${Boost_INCLUDE_DIRS})
#   list(APPEND LIBS ${Boost_LIBRARIES})
#   MESSAGE(STATUS "Using filesystem from Boost")
#endif()

#Warning: Not tested in Windows
#Normally results in experimental/filesystem for C++ 14,11 and filesystem for C++ 17 at least with g++ and clang++
MESSAGE(STATUS "CMAKE_BINARY_DIR " ${CMAKE_BINARY_DIR})
file(WRITE ${CMAKE_BINARY_DIR}/filesystem_test.cc "#include <filesystem>\nint main(int argc, char** argv)\n{ std::filesystem::path p; return 0;}\n")
#try_compile is a complete abortion; it keeps trying to link with a non-existent libSTANDARD. Anyway a link is not required (assuming libstdc++fs Windows ?).
#try_compile(FILESYSTEM_COMPILE ${CMAKE_BINARY_DIR}/filesystem_test ${CMAKE_BINARY_DIR}/filesystem_test.cc LINK_LIBRARIES stdc++fs STANDARD 14)
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -c --std=c++${CMAKE_CXX_STANDARD} ${CMAKE_BINARY_DIR}/filesystem_test.cc
                RESULT_VARIABLE FILESYSTEM_STATUS OUTPUT_QUIET ERROR_QUIET)
MESSAGE(STATUS "C++: " ${CMAKE_CXX_STANDARD})
if (FILESYSTEM_STATUS)
   file(WRITE ${CMAKE_BINARY_DIR}/exfilesystem_test.cc "#include <experimental/filesystem>\nint main(int argc, char** argv)\n{ std::experimental::filesystem::path p; return 0;}\n")
   #try_compile(EXFILESYSTEM_COMPILE ${CMAKE_BINARY_DIR}/exfilesystem_test ${CMAKE_BINARY_DIR}/exfilesystem_test.cc LINK_LIBRARIES stdc++fs STANDARD 14)
   execute_process(COMMAND ${CMAKE_CXX_COMPILER} -c --std=c++${CMAKE_CXX_STANDARD} ${CMAKE_BINARY_DIR}/exfilesystem_test.cc
                   RESULT_VARIABLE EXFILESYSTEM_STATUS OUTPUT_QUIET ERROR_QUIET)
   if (EXFILESYSTEM_STATUS)
   # This might not work (without further ifdefs) as boost::filesystem does not appear to be completely compatible
      FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem)
      list(APPEND FLAGS "-DFILESYSTEM_BOOST")
      list(APPEND INCLUDES ${Boost_INCLUDE_DIRS})
      list(APPEND LIBS ${Boost_LIBRARIES})
      MESSAGE(STATUS "Using #include <boost/filesystem>")
   else()
      list(APPEND FLAGS "-DFILESYSTEM_EXPERIMENTAL")
      list(APPEND LIBS stdc++fs)
      MESSAGE(STATUS "Using #include <experimental/filesystem>")
   endif()
else()
   MESSAGE(STATUS "Using #include <filesystem>")
   list(APPEND FLAGS "-DSTD_FILESYSTEM")
   list(APPEND LIBS stdc++fs)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set (USE_CUSTOM_OPENCV FALSE)
set(USE_INSTALLED_EIGEN TRUE)
set (CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")
set (CUSTOM_OPENCV_DIR "/opt/opencv")
#set(OpenCV_STATIC ON)
set(use_theia_ransac true)

set(DEPENDENCIES "${PROJECT_SOURCE_DIR}/libs")
set(LOCAL_EIGEN_REPO "https://github.com/eigenteam/eigen-git-mirror.git")
set(LOCAL_EIGEN "${DEPENDENCIES}/eigen-git-mirror")
set(LOCAL_EIGEN_GIT "${LOCAL_EIGEN}/.git")

if(USE_CUSTOM_OPENCV)
    set(OpenCV_FIND_QUIETLY false)
    set (OpenCV_DIR "${CUSTOM_OPENCV_DIR}")
    find_package( OpenCV REQUIRED
                  HINTS "${CMAKE_PREFIX_PATH}${CUSTOM_OPENCV_DIR}")
    if ("${OpenCV_INCLUDE_DIR}" STREQUAL "")
        set(OpenCV_INCLUDE_DIR "${CUSTOM_OPENCV_DIR}/include/")
    endif()
    list(APPEND INCLUDES ${OpenCV_INCLUDE_DIR})
else()
    find_package( OpenCV REQUIRED)
    set(OpenCV_INCLUDE_DIR "/usr/include/")
endif()

list(APPEND LIBS ${OpenCV_LIBS})

if (use_theia_ransac)
    list(APPEND FLAGS "-DUSE_THEIA_RANSAC")
    MESSAGE(STATUS "Using Theia RANSAC")
else()
    MESSAGE(STATUS "Using TemplRANSAC")
endif()

MESSAGE(STATUS "OpenCV Dir:" "${OpenCV_INSTALL_PATH}")
MESSAGE(STATUS "OpenCV Version:" "${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}")
MESSAGE(STATUS "OpenCV include directory: " "${OpenCV_INCLUDE_DIR}")
MESSAGE(STATUS "OpenCV libs: " "${OpenCV_LIBS}")

if(USE_INSTALLED_EIGEN)
   find_package(Eigen3 REQUIRED)
else()
   file(MAKE_DIRECTORY ${LOCAL_EIGEN})
   if (EXISTS ${LOCAL_EIGEN} AND EXISTS ${LOCAL_EIGEN_GIT})
      execute_process(COMMAND git pull WORKING_DIRECTORY ${LOCAL_EIGEN})
   else()
      execute_process(COMMAND git clone ${LOCAL_EIGEN_REPO} WORKING_DIRECTORY ${DEPENDENCIES})
   endif()
   set(EIGEN3_INCLUDE_DIR "${LOCAL_EIGEN}")
endif()
MESSAGE(STATUS "Eigen 3 Include: " ${EIGEN3_INCLUDE_DIR})
list(APPEND INCLUDES ${EIGEN3_INCLUDE_DIR})

#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#   MESSAGE(STATUS "GCC Detected")
#   SET(CMAKE_CXX_FLAGS "-Wall -std=c++14")
#   SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-ggdb -O2")
#   SET(CMAKE_CXX_FLAGS_RELEASE "-march=native -O2 -pipe -fstack-protector --param=ssp-buffer-size=4 -D_FORTIFY_SOURCE=2")
#   #   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g3 -ggdb")
#   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g3 -gdwarf-4")
#   list(APPEND FLAGS "-DISGPP")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#   MESSAGE(STATUS "Clang Detected")
#   SET(CMAKE_CXX_FLAGS "-std=c++14")
#   SET(CMAKE_CXX_FLAGS_RELEASE "-march=native -Ofast3")
#   #   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g3 -ggdb")
#   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g3")
#endif()

find_package(YamlCpp05 REQUIRED)
message(STATUS "YAML include:" "${YAMLCPP_INCLUDE_DIR}")
message(STATUS "YAML lib:" "${YAMLCPP_LIBRARY}")
list(APPEND INCLUDES ${YAMLCPP_INCLUDE_DIR})
list(APPEND LIBS ${YAMLCPP_LIBRARY})

message(STATUS "Flags: " ${FLAGS})
message(STATUS "Includes: " ${INCLUDES})
message(STATUS "Libs: " ${LIBS})

add_library( theia-ransac theia-RANSAC/random.h theia-RANSAC/random.cc theia-RANSAC/estimator.h
             theia-RANSAC/mle_quality_measurement.h
             theia-RANSAC/random_sampler.cc theia-RANSAC/ransac.h theia-RANSAC/sampler.h theia-RANSAC/inlier_support.h
             theia-RANSAC/quality_measurement.h theia-RANSAC/random_sampler.h theia-RANSAC/sample_consensus_estimator.h
             theia-RANSAC/create_and_initialize_ransac_variant.h theia-RANSAC/prosac.h theia-RANSAC/prosac_sampler.cc
             theia-RANSAC/prosac_sampler.h theia-RANSAC/exhaustive_ransac.h theia-RANSAC/exhaustive_sampler.cc
             theia-RANSAC/exhaustive_sampler.h)
target_include_directories( theia-ransac PUBLIC ${CMAKE_SOURCE_DIR}/theia-RANSAC
                            ${EIGEN3_INCLUDE_DIR})

add_executable(planargrav src/optionparser.h src/common.cc src/common.h
               src/SelectMatchedPoints.cc src/SelectMatchedPoints.h src/math.hh
               src/Ransac.hh src/five_point_relative_pose.cc src/pose2d.cc src/pose2d.h src/pose3d.cc src/pose3d.h
               src/poseocv.cc src/poseocv.h src/pose-other.h src/pose-other.cc
               src/PoseRANSAC.hh src/PoseRANSAC.cc
               src/two_point_pose_partial_rotation.cc src/two_point_pose_partial_rotation.h src/Polynomial.cc
               src/Optimization.cc src/Optimization.h src/synth2d.cc src/synth2d.h src/planargrav.cc)
target_compile_options( planargrav PRIVATE ${FLAGS} )
TARGET_INCLUDE_DIRECTORIES(planargrav PUBLIC ${INCLUDES})
TARGET_LINK_LIBRARIES(planargrav theia-ransac ${LIBS})

add_executable(pnpgrav src/optionparser.h src/math.hh src/Ransac.hh src/common.cc src/common.h
               src/SelectMatchedPoints.cc src/SelectMatchedPoints.h src/pose2d.cc src/pose2d.h
               src/pose3d.cc src/pose3d.h src/poseocv.cc src/poseocv.h src/PoseRANSAC.hh src/PoseRANSAC.cc
               src/two_point_pose_partial_rotation.cc src/two_point_pose_partial_rotation.h src/Polynomial.cc
               src/Optimization.cc src/Optimization.h src/RandomHelper.h src/pnpgrav.cc src/synth3d.h src/synth3d.cc)
target_compile_options( pnpgrav PRIVATE ${FLAGS} )
TARGET_INCLUDE_DIRECTORIES(pnpgrav PUBLIC ${INCLUDES})
TARGET_LINK_LIBRARIES(pnpgrav theia-ransac ${LIBS})
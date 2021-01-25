# ----------------------------------------------------------------------------
#  Root CMake file for the rfc pso Project
# ----------------------------------------------------------------------------
#
# To perform an out of source build do the following on the command line:
# 1. navigate to the root project directory (in this case DfD2)
# 2. mkdir build
# 3. cd build
# 4. depending on your system (windows/linux) type the following
# Example CMake build line for windows & VS2015
# windows: cmake -G "Visual Studio 14 2015 Win64" -DUSE_AVX_INSTRUCTIONS=ON ..
# linux: cmake -DUSE_AVX_INSTRUCTIONS=ON ..
#
# 5. To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j4
#

cmake_minimum_required(VERSION 2.8.12)

PROJECT(RFC_Project)
set(exe_name "rfc_pso")

set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
message("Platform: " ${PLATFORM})

option(USE_AVX_INSTRUCTIONS  "Compile your program with AVX instructions"  ON)

if(WIN32)
    if(${PLATFORM} STREQUAL "Laptop")
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        include(D:/dlib-19.21/dlib/cmake)
    elseif(${PLATFORM} STREQUAL "LaptopN")
        set(COMMON_ROOT "C:/Projects/Common")
        set(DLIB_CONTRIB "C:/Projects/dlib-contrib")
        include(C:/dlib-19.21/dlib/cmake)        
    else()
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        include(D:/dlib-19.21/dlib/cmake)
    endif()
else()
    if(${PLATFORM} STREQUAL "MainGear")
        set(_ECLIPSE_VERSION "3.8")
        set(COMMON_ROOT "~/Projects/Common")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
        include(~/dlib-19.21/dlib/cmake)
    elseif(${PLATFORM} STREQUAL "HPC")
        set(COMMON_ROOT "~/Projects/Common")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
        include(~/dlib-19.21/dlib/cmake)
    else()
        set(COMMON_ROOT "~/Projects/Common")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
        include(~/dlib-19.21/dlib/cmake)
    endif()
endif()

# grab all of the *.c and *.cpp files in the src directory
file(GLOB SOURCES "src/*.c*"	
    "${COMMON_ROOT}/src/*.cpp"
    "${DLIB_CONTRIB}/src/*.cpp"
    ) 

set(project_includes "include" 
    "${COMMON_ROOT}/include" 
    "${DLIB_CONTRIB}/include" 
    )


message(" ")
message("--------------------------------------------------------")
message("Source Files: ")
foreach(S ${SOURCES})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")

message(" ")

message("--------------------------------------------------------")
message("Project Includes: ")
foreach(S ${project_includes})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")


include_directories(${project_includes})
add_executable(${exe_name} ${SOURCES})
target_link_libraries(${exe_name} dlib::dlib)

if(WIN32)
    message("MSVC Version: " ${MSVC_VERSION})
    # uncomment for MSVC and cmake greater than 3.6.0
    set(default_build_type "Release")
    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        target_compile_options(${exe_name} PRIVATE "/MP")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${exe_name})
    endif()

endif()

message(" ")
message(STATUS "Build Complete!")
message(" ")



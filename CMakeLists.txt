cmake_minimum_required(VERSION 3.20)
project(nanoback LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(NANOBACK_PGO_MODE "OFF" CACHE STRING "Profile-guided optimization mode: OFF, GENERATE, USE")
set_property(CACHE NANOBACK_PGO_MODE PROPERTY STRINGS OFF GENERATE USE)
option(NANOBACK_ENABLE_NATIVE_OPTIMIZATIONS "Enable architecture-specific optimization flags for local builds" OFF)

find_package(pybind11 CONFIG REQUIRED)
find_package(TBB QUIET)

pybind11_add_module(_nanoback
    cpp/backtester.cpp
    cpp/policy.cpp
    cpp/bindings.cpp
)

target_include_directories(_nanoback PRIVATE include)

if(MSVC)
    target_compile_options(_nanoback PRIVATE /O2 /permissive-)
    if(NANOBACK_ENABLE_NATIVE_OPTIMIZATIONS)
        target_compile_options(_nanoback PRIVATE /GL)
        target_link_options(_nanoback PRIVATE /LTCG)
    endif()
else()
    target_compile_options(_nanoback PRIVATE -O3)
    if(NANOBACK_ENABLE_NATIVE_OPTIMIZATIONS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(_nanoback PRIVATE -march=native -flto)
            target_link_options(_nanoback PRIVATE -flto)
        endif()
    endif()
    if(NANOBACK_PGO_MODE STREQUAL "GENERATE")
        target_compile_options(_nanoback PRIVATE -fprofile-generate)
        target_link_options(_nanoback PRIVATE -fprofile-generate)
    elseif(NANOBACK_PGO_MODE STREQUAL "USE")
        target_compile_options(_nanoback PRIVATE -fprofile-use -fprofile-correction)
        target_link_options(_nanoback PRIVATE -fprofile-use -fprofile-correction)
    endif()
    if(TBB_FOUND)
        target_compile_definitions(_nanoback PRIVATE NANOBACK_USE_PAR_STL=1)
        target_link_libraries(_nanoback PRIVATE TBB::tbb)
    endif()
endif()

install(TARGETS _nanoback LIBRARY DESTINATION nanoback RUNTIME DESTINATION nanoback)

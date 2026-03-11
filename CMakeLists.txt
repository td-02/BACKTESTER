cmake_minimum_required(VERSION 3.20)
project(nanoback LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_nanoback
    cpp/backtester.cpp
    cpp/policy.cpp
    cpp/bindings.cpp
)

target_include_directories(_nanoback PRIVATE include)

if(MSVC)
    target_compile_options(_nanoback PRIVATE /O2 /GL /permissive-)
else()
    target_compile_options(_nanoback PRIVATE -O3 -march=native)
endif()

install(TARGETS _nanoback LIBRARY DESTINATION nanoback RUNTIME DESTINATION nanoback)

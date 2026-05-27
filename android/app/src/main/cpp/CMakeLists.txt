cmake_minimum_required(VERSION 3.22.1)

project(vaultiq_gguf_runner LANGUAGES CXX)

add_library(
    vaultiq_gguf_runner
    SHARED
    gguf_inspector.cpp
    native_bridge.cpp
)

target_compile_features(vaultiq_gguf_runner PRIVATE cxx_std_17)
target_compile_options(vaultiq_gguf_runner PRIVATE -Wall -Wextra)
target_compile_definitions(vaultiq_gguf_runner PRIVATE _FILE_OFFSET_BITS=64)

find_library(log-lib log)
target_link_libraries(vaultiq_gguf_runner PRIVATE ${log-lib})

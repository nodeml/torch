cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# Name of the project (will be the name of the plugin)
project(nodeml_torch VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DNAPI_VERSION=4)

include_directories(${CMAKE_JS_INC})

set(PYTORCH_VERSION 2.1.0)

include(${CMAKE_SOURCE_DIR}/utils.cmake)

DownloadTorch(${PYTORCH_VERSION} "${CMAKE_SOURCE_DIR}/deps/dynamic")

set(TORCH_DEPS_DIR "${CMAKE_SOURCE_DIR}/deps/dynamic/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DEPS_DIR})
list(APPEND CMAKE_PREFIX_PATH "${TORCH_DEPS_DIR}/share/cmake/Torch")

# Build a shared library named after the project from the files in `src/`
file(GLOB_RECURSE SOURCE_FILES "*.cc" "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

# Gives our library file a .node extension without any "lib" prefix
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Essential include files to build a node addon,
# You should add this line in every CMake.js based project
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include N-API wrappers
IncludeNapi(${PROJECT_NAME})

# Libtorch
find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})


# Essential library files to link to a node addon
# You should add this line in every CMake.js based project
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)


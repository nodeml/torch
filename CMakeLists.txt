cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# Name of the project (will be the name of the plugin)
project(nodeml_torch)

set(CMAKE_CXX_STANDARD 17)


set(PYTORCH_VERSION 2.1.0)

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/deps/dynamic/libtorch)
  file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip ${CMAKE_CURRENT_SOURCE_DIR}/torch.zip SHOW_PROGRESS)
  message(STATUS "Extracting")
  file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_SOURCE_DIR}/torch.zip DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/deps/dynamic)
  message(STATUS "Done Extracting")
  file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/torch.zip)
endif()

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/deps/dynamic/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/deps/dynamic/libtorch/share/cmake/Torch")


find_package(Torch REQUIRED)

# Build a shared library named after the project from the files in `src/`
file(GLOB SOURCE_FILES "src/*.cc" "src/*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

# Gives our library file a .node extension without any "lib" prefix
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Essential include files to build a node addon,
# You should add this line in every CMake.js based project
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include N-API wrappers
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/node_modules/node-addon-api" "src" )

# Libtorch
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
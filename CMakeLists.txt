cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# Name of the project (will be the name of the plugin)
project(nodeml_torch VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DNAPI_VERSION=4)

include_directories(${CMAKE_JS_INC})

set(PYTORCH_VERSION 2.1.0)
set(PYTORCH_VISION_VERSION 0.16.0)

include(${CMAKE_SOURCE_DIR}/utils.cmake)

set(TORCH_DEPS_DIR "${CMAKE_SOURCE_DIR}/deps/dynamic/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DEPS_DIR})
list(APPEND CMAKE_PREFIX_PATH "${TORCH_DEPS_DIR}/share/cmake/Torch")

# Build a shared library named after the project from the files in `src/`
file(GLOB_RECURSE SOURCE_FILES "src/*.cc" "src/*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

# Gives our library file a .node extension without any "lib" prefix
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Essential include files to build a node addon,
# You should add this line in every CMake.js based project
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include N-API wrappers
IncludeNapi(${PROJECT_NAME})

# Essential library files to link to a node addon
# You should add this line in every CMake.js based project
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})

# Node lib
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

# Torch
set(TORCH_DEPS_DIR ${CMAKE_SOURCE_DIR}/deps/dynamic/libtorch)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DEPS_DIR})
list(APPEND CMAKE_PREFIX_PATH "${TORCH_DEPS_DIR}/share/cmake/Torch")
DownloadTorch(${PYTORCH_VERSION} "${CMAKE_SOURCE_DIR}/deps/dynamic")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

# Torchvision
set(TORCH_VISION_DEPS_DIR ${CMAKE_SOURCE_DIR}/deps/dynamic/torchvision)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_VISION_DEPS_DIR})
list(APPEND CMAKE_PREFIX_PATH "${TORCH_VISION_DEPS_DIR}/share/cmake/TorchVision")
if(NOT EXISTS ${TORCH_VISION_DEPS_DIR})
  execute_process(
    COMMAND git clone --depth 1 --branch v${PYTORCH_VISION_VERSION} https://github.com/pytorch/vision
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON=OFF -DWITH_PNG=ON -DWITH_JPEG=ON -DCMAKE_PREFIX_PATH=${TORCH_DEPS_DIR} -S "${CMAKE_CURRENT_BINARY_DIR}/vision/" -B "${CMAKE_CURRENT_BINARY_DIR}/vision/build/"
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_CURRENT_BINARY_DIR}/vision/build" --config Release
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}/vision/build" --prefix ${TORCH_VISION_DEPS_DIR}
  )

  file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/vision)
endif()
find_package(TorchVision REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE "${TORCH_VISION_DEPS_DIR}/include")
target_link_libraries(${PROJECT_NAME} TorchVision::TorchVision)



if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll" "${TORCH_VISION_DEPS_DIR}/bin/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
  
endif (MSVC)


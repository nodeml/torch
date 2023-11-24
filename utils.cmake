function(DownloadTorch version cuda destination)
  if(NOT EXISTS ${destination}/libtorch)
    set(DOWNLOAD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/torch.zip)
    if(cuda)
      if(WIN32)
        file(DOWNLOAD https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-${version}%2Bcu118.zip ${DOWNLOAD_FILE} SHOW_PROGRESS)
      elseif(APPLE)
        message(FATAL_ERROR "NO GPU SUPPORT FOR APPLE")
      else()
        file(DOWNLOAD https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${version}%2Bcu118.zip ${DOWNLOAD_FILE} SHOW_PROGRESS)
      endif()
    else()
      #file(DOWNLOAD https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-${version}%2Bcu118.zip ${CMAKE_CURRENT_SOURCE_DIR}/torch.zip SHOW_PROGRESS)
      if(WIN32)
        file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${version}%2Bcpu.zip ${DOWNLOAD_FILE} SHOW_PROGRESS)
      elseif(APPLE)
        file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-macos-${version}.zip ${DOWNLOAD_FILE} SHOW_PROGRESS)
      else()
        file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${version}%2Bcpu.zip ${DOWNLOAD_FILE} SHOW_PROGRESS)
      endif()
    endif()

    message(STATUS "Extracting")
    file(ARCHIVE_EXTRACT INPUT ${DOWNLOAD_FILE} DESTINATION ${destination})
    message(STATUS "Done Extracting")
    file(REMOVE ${DOWNLOAD_FILE})
  endif()
endfunction()

macro(IncludeNapi project_name)
    # Include N-API wrappers
    execute_process(COMMAND node -p "require('node-addon-api').include"
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE NODE_ADDON_API_DIR
            )
    string(REPLACE "\n" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
    string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
    target_include_directories(${project_name} PRIVATE ${NODE_ADDON_API_DIR})
    add_definitions(-DNAPI_VERSION=3)
endmacro()

function(BuildTorchVision VERSION DESTINATION)
  if(NOT EXISTS ${DESTINATION}/torchvision)
    set(CLONED_DIR ${CMAKE_CURRENT_BINARY_DIR}/vision)

    execute_process(
      COMMAND git clone --depth 1 --branch v${VERSION} https://github.com/pytorch/vision ${CLONED_DIR}
    )


    execute_process(
      COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON=OFF -DWITH_CUDA=${WITH_CUDA} -DWITH_PNG=ON -DWITH_JPEG=ON -DCMAKE_PREFIX_PATH=${TORCH_DEPS_DIR} -S ${CLONED_DIR} -B ${CLONED_DIR}/build/
    )

    execute_process(
      COMMAND ${CMAKE_COMMAND} --build ${CLONED_DIR}/build --config Release
    )

    execute_process(
      COMMAND ${CMAKE_COMMAND} --install ${CLONED_DIR}/build --prefix ${DESTINATION}/torchvision
    )
  endif()
endfunction()
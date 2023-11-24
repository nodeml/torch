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

# Get the version from the package.json
function(GetVersion)
  file(READ ${CMAKE_SOURCE_DIR}/package.json PACKAGE_JSON)
  string(JSON PACKAGE_VERSION GET ${PACKAGE_JSON} version)
  set(PACKAGE_VERSION ${PACKAGE_VERSION} PARENT_SCOPE)
endfunction()

# generate node.lib
function(GenerateNodeLib)
  if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    # Generate node.lib
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
  endif()
endfunction()
function(DownloadTorch version destination)
  if(NOT EXISTS ${destination}/libtorch)
    file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${version}%2Bcpu.zip ${CMAKE_CURRENT_SOURCE_DIR}/torch.zip SHOW_PROGRESS)
    message(STATUS "Extracting")
    file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_SOURCE_DIR}/torch.zip DESTINATION ${destination})
    message(STATUS "Done Extracting")
    file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/torch.zip)
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

function(GetNodeMlLib result lib_name)
    set(TEST_1 ${CMAKE_SOURCE_DIR}/node_modules/@nodeml/${lib_name})
    if(EXISTS ${TEST_1})
        set(${result} ${TEST_1} PARENT_SCOPE)
    else()
        set(${result} ${CMAKE_SOURCE_DIR}/../${lib_name} PARENT_SCOPE)
    endif()
endfunction()
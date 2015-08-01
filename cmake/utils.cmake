# short command for setting default target properties
function(default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

# short command for adding all applications
function(add_applications)
  file(GLOB __apps apps/*.cc)

  foreach(__a ${__apps})
    get_filename_component(__name ${__a} NAME_WE)
    add_executable(${__name} ${__a})
    target_link_libraries(${__name} graphmodels)
    default_properties(${__name})
  endforeach()
endfunction()


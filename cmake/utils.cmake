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
  file(GLOB __test test/*.cc)
  list(APPEND __apps ${__test})

  foreach(__a ${__apps})
    get_filename_component(__name ${__a} NAME_WE)
    add_executable(${__name} ${__a})
    target_link_libraries(${__name} graphmodels)
    target_link_libraries(${__name} math)
    target_link_libraries(${__name} /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_imgcodecs.so /usr/local/lib/libopencv_imgproc.so)
    default_properties(${__name})
  endforeach()
endfunction()


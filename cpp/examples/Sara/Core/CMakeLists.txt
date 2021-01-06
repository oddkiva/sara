find_package(DO_Sara COMPONENTS Core REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example}
    PROPERTY
    FOLDER "Examples/Sara/Core")
endmacro ()

if (UNIX)
  sara_add_example(save_to_hdf5_example)
  target_link_libraries(save_to_hdf5_example ${HDF5_LIBRARIES})
endif ()

if (SARA_USE_ZEROMQ)
  sara_add_example(core_ipc_zmq_example)
  target_include_directories(core_ipc_zmq_example PRIVATE ${Boost_INCLUDE_DIR})
  if (UNIX AND NOT APPLE)
    target_link_libraries(core_ipc_zmq_example rt)
  endif ()
  target_link_libraries(core_ipc_zmq_example zmq)
endif ()

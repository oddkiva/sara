if (MSVC)
  do_step_message("Found MSVC compiler:")
  add_definitions(/D _SCL_SECURE_NO_WARNINGS)
  message(STATUS "  - NON-SECURE warnings are disabled.")
  if (MSVC_VERSION EQUAL 1700)
    message(STATUS "  - Using version 2012: setting '_VARIADIC_MAX=10' to compile 'Google Test'")
    add_definitions(/D _VARIADIC_MAX=10)
  endif ()
endif (MSVC)
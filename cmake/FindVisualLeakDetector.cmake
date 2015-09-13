find_path(
  VLD_INSTALL_DIR include/vld.h HINTS
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Visual Leak Detector;InstallLocation]"
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Visual Leak Detector;InstallLocation]"
)

if(NOT VLD_INSTALL_DIR)
  message(
    FATAL_ERROR "VLD installation directory not found. Turn off the "
                "DO_ENABLE_VLD option to build the project."
  )
endif()

set(VLD_INCLUDE_DIR ${VLD_INSTALL_DIR}/include)

if ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
  set(VLD_LINK_DIRECTORIES ${VLD_INSTALL_DIR}/lib/Win64)
else ()
  set(VLD_LINK_DIRECTORIES ${VLD_INSTALL_DIR}/lib/Win32)
endif ()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# Eigen 3
do_message("Installing Eigen")
install(DIRECTORY ${DO_ThirdParty_DIR}/eigen
        DESTINATION include
        COMPONENT Eigen3)
set(CPACK_COMPONENT_Eigen3_REQUIRED 1)

# DO++ source files
install(FILES ${DO_DIR}/COPYING.README
              ${DO_DIR}/COPYING.MPL2
              ${DO_DIR}/CPACK_PACKAGING.README
              ${DO_DIR}/CMakeLists.txt
        DESTINATION share/DO/Sara
        COMPONENT Sources)
install(DIRECTORY ${DO_DIR}/cmake
        DESTINATION share/DO/Sara
        COMPONENT Sources)
install(DIRECTORY ${DO_DIR}/src
        DESTINATION include
        COMPONENT Sources)
install(DIRECTORY ${DO_DIR}/third-party
        DESTINATION include
        COMPONENT Sources)
install(FILES ${DO_SOURCE_DIR}/Defines.hpp
        DESTINATION include/DO/Sara
        COMPONENT Sources)
set(CPACK_COMPONENT_Sources_REQUIRED 1)

# DO++ component libraries
foreach (component ${DO_COMPONENTS})
    do_message("Installing DO.${component}")
    include(${DO_${component}_USE_FILE})
    # Install master header file
    install(FILES ${DO_SOURCE_DIR}/${component}.hpp
            DESTINATION include/DO
            COMPONENT ${component})
    # Install header files
    install(DIRECTORY ${DO_SOURCE_DIR}/${component}
            DESTINATION include/DO
            COMPONENT ${component}
            FILES_MATCHING PATTERN "*.hpp")
endforeach (component)


# List all available components for installation.
set(CPACK_COMPONENTS_ALL Eigen3 Sources Core Graphics Libraries)


set(CPACK_PACKAGE_NAME "libdo-sara")
set(CPACK_PACKAGE_VENDOR "DO-CV")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "DO-CV - An easy-to-use C++ set of libraries for computer vision")
set(CPACK_RESOURCE_FILE_LICENSE "${DO_DIR}/COPYING.README")
set(CPACK_PACKAGE_CONTACT "David OK")

set(CPACK_PACKAGE_VERSION_MAJOR ${DO_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${DO_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${DO_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${DO_SARA_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "DO/Sara")


# Specific variables for Debian packages.
set(CPACK_DEBIAN_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
#set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "cmake, libjpeg-dev, libpng-dev, libtiff5-dev, qtbase5-dev")

if (WIN32)
  set(CPACK_GENERATOR NSIS)
elseif (UNIX)
  set(CPACK_GENERATOR "DEB")
endif()

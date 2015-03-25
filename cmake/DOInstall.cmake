set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# Eigen 3
do_message("Installing Eigen")
install(DIRECTORY ${Eigen3_DIR}/Eigen
        DESTINATION include
        COMPONENT Eigen3)
set(CPACK_COMPONENT_Eigen3_REQUIRED 1)

# DO++ source files
install(FILES ${DO_DIR}/COPYING.README
              ${DO_DIR}/COPYING.MPL2 
              ${DO_DIR}/CPACK_PACKAGING.README
              ${DO_DIR}/CMakeLists.txt
              ${DO_DIR}/Doxyfile.in
        DESTINATION .
        COMPONENT Sources)
install(DIRECTORY ${DO_DIR}/cmake
        DESTINATION .
        COMPONENT Sources)
install(DIRECTORY ${DO_DIR}/src
        DESTINATION .
        COMPONENT Sources)
install(DIRECTORY ${DO_DIR}/third-party
        DESTINATION .
        COMPONENT Sources)
install(DIRECTORY ${DO_DIR}/test
        DESTINATION .
        COMPONENT Sources)
install(FILES ${DO_SOURCE_DIR}/Defines.hpp
        DESTINATION include/DO
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
    # Install dynamic libraries
    # if (NOT ${DO_${component}_LIBRARIES} STREQUAL "") # This is how I check if the library is not header-only...
        # install(TARGETS DO_${component}
                # ARCHIVE DESTINATION lib
                # COMPONENT ${component})
        # install(TARGETS DO_${component}_SHARED
                # LIBRARY DESTINATION bin
                # ARCHIVE DESTINATION bin
                # RUNTIME DESTINATION bin
                # COMPONENT ${component})
    # endif ()
endforeach (component)


if (MSVC)
  if (MSVC9)
    set(DO_LIB_DIR ${CMAKE_SOURCE_DIR}/packaged-libs/msvc9)
  endif ()
  if (MSVC10)
    set(DO_LIB_DIR ${CMAKE_SOURCE_DIR}/packaged-libs/msvc10)
  endif ()
  if (MSVC11)
    set(DO_LIB_DIR ${CMAKE_SOURCE_DIR}/packaged-libs/msvc11)
  endif ()
  install(DIRECTORY ${DO_LIB_DIR}
          DESTINATION .
          COMPONENT Libraries)
endif ()


# List all available components for installation.
set(CPACK_COMPONENTS_ALL Eigen3 Sources Core Graphics Libraries)
# Create windows package with NSIS.
set(CPACK_GENERATOR NSIS)
set(CPACK_PACKAGE_NAME "DO++")
set(CPACK_PACKAGE_VENDOR "David Ok")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "DO++ - Basic set of libraries for computer vision")
# License
set(CPACK_RESOURCE_FILE_LICENSE "${DO_DIR}/COPYING.README")
    
# Put version.
set(CPACK_PACKAGE_VERSION_MAJOR "${DO_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${DO_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${DO_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION "${DO_VERSION}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME}-${DO_VERSION}")

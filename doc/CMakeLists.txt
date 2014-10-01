find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(Doxyfile.in 
                 ${CMAKE_CURRENT_BINARY_DIR}/Doc/Doxyfile @ONLY)
  add_custom_target(MAKE_DO_DOC
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/source
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)
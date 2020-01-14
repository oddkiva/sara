find_package(Sphinx)

if (SPHINX_EXECUTABLE)
  set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source)
  set(SPHINX_BUILD ${CMAKE_BINARY_DIR}/doc/sphinx)

  add_custom_target(doc_sphinx
    COMMAND ${SPHINX_EXECUTABLE} -b html ${SPHINX_SOURCE} ${SPHINX_BUILD}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/source
    COMMENT "Generating documentation with Sphinx")

  add_custom_target(sphinx_autobuild
    COMMAND sphinx-autobuild ${SPHINX_SOURCE} ${SPHINX_BUILD}
            --open-browser -s 1
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Launching sphinx-autobuild server")
endif ()

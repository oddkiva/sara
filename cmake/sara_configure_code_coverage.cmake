# Configure code coverage.
if (CMAKE_COMPILER_IS_GNUCXX)
  find_program(LCOV_PATH lcov)
  find_program(GCOV_PATH gcov)
  find_program(GENHTML_PATH genhtml)

  if (LCOV_PATH AND GCOV_PATH)
    add_custom_target(coverage
      COMMAND ${LCOV_PATH} --gcov-tool=${GCOV_PATH} --compat-libtool
                           --directory=${CMAKE_BINARY_DIR}
                           --base-directory=${DO_Sara_SOURCE_DIR}
                           --capture
                           --output-file=coverage.info
      COMMAND ${LCOV_PATH} --remove coverage.info '/usr/*'
                           --output-file coverage.info
      COMMAND ${LCOV_PATH}
              --remove coverage.info '${DO_Sara_DIR}/cpp/third-party/*'
              --output-file coverage.info
      COMMAND ${LCOV_PATH}
              --remove coverage.info '${DO_Sara_DIR}/cpp/test/*'
              --output-file coverage.info
      COMMAND ${LCOV_PATH}
              --remove coverage.info '${CMAKE_BINARY_DIR}/*'
              --output-file coverage.info

      # Remove python bindings.
      COMMAND ${LCOV_PATH}
              --remove coverage.info '${DO_Sara_DIR}/python/*'
              --output-file coverage.info
      COMMAND ${LCOV_PATH}
              --remove coverage.info '*/numpy/*'
              --output-file coverage.info

      # Remove tests in the drafts folder.
      COMMAND ${LCOV_PATH}
              --remove coverage.info '${DO_Sara_DIR}/cpp/drafts/*/test*'
              --output-file coverage.info
    )
  endif ()

  if (GENHTML_PATH)
    add_custom_target(coverage_html_report
      COMMAND ${GENHTML_PATH} coverage.info
              --output-directory=${CMAKE_BINARY_DIR}/coverage)
    add_dependencies(coverage_html_report coverage)
  endif ()
endif ()

# CHANGELOG
# =========
#
# 2013-09-17, Joakim Söderberg
# - Added support for Clang.
# - Some additional usage instructions.
#
# 2012-01-31, Lars Bilke
# - Enable Code Coverage
#
#
# USAGE
# =====
#
# 1. Copy this file into your cmake modules path.
#
# 2. Add the following line to your CMakeLists.txt:
#    include(CodeCoverage)
#
# 3. Set compiler flags to turn off optimization and enable coverage: 
#    set(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
#    set(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
#  
# 4. Use the function SETUP_TARGET_FOR_COVERAGE to create a custom make target
#    which runs your test executable and produces a lcov code coverage report:
#    Example:
#	 setup_target_for_coverage(
#        my_coverage_target  # Name for custom target.
#
#        test_driver         # Name of the test driver executable that runs 
#                            # the tests.
#                            # NOTE! This should always have a ZERO as exit 
#                            # code, otherwise the coverage generation will 
#                            # not complete.
#
#        coverage            # Name of output directory.
#				)
#
# 5. Build a Debug build:
#	 cmake -DCMAKE_BUILD_TYPE=Debug ..
#	 make
#	 make my_coverage_target
#
#


# Check prerequisites
find_program(GCOV_PATH gcov)
find_program(LCOV_PATH lcov)
find_program(GENHTML_PATH genhtml)
find_program(GCOVR_PATH gcovr PATHS ${CMAKE_SOURCE_DIR}/tests)

if(NOT GCOV_PATH)
	message(FATAL_ERROR "gcov not found! Aborting...")
endif()

if(NOT CMAKE_COMPILER_IS_GNUCXX)
	# Clang version 3.0.0 and greater now supports gcov as well.
	message(WARNING 
        "Compiler is not GNU gcc! Clang Version 3.0.0 and greater supports "
        "gcov as well, but older versions don't."
    )
	
	if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		message(FATAL_ERROR "Compiler is not GNU gcc! Aborting...")
	endif()
endif()

set(CMAKE_C_FLAGS_COVERAGE "-g -O0 --coverage -fprofile-arcs -ftest-coverage"
    CACHE STRING "Flags used by the C compiler during coverage builds."
    FORCE)
set(CMAKE_CXX_FLAGS_COVERAGE
    "-g -O0 --coverage -fprofile-arcs -ftest-coverage"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE)
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE ""
    CACHE STRING "Flags used for linking binaries during coverage builds."
    FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE ""
    CACHE STRING
    "Flags used by the shared libraries linker during coverage builds."
    FORCE)

mark_as_advanced(
    CMAKE_CXX_FLAGS_COVERAGE
    CMAKE_C_FLAGS_COVERAGE
    CMAKE_EXE_LINKER_FLAGS_COVERAGE
    CMAKE_SHARED_LINKER_FLAGS_COVERAGE )

if (NOT (CMAKE_BUILD_TYPE STREQUAL "Debug" OR
         CMAKE_BUILD_TYPE STREQUAL "Coverage"))
  message(WARNING 
      "Code coverage results with an optimized (non-Debug) build may be "
      "misleading" )
endif()


# Param _targetname     The name of new the custom make target
# Param _testrunner     The name of the target which runs the tests.
#						MUST return ZERO always, even on errors. 
#						If not, no coverage report will be created!
# Param _outputname     lcov output is generated as _outputname.info
#                       HTML report is generated in _outputname/index.html
# Optional fourth parameter is passed as arguments to _testrunner
#   Pass them in list form, e.g.: "-j;2" for -j 2
function(SETUP_TARGET_FOR_COVERAGE _targetname _testrunner _outputname)
	if(NOT LCOV_PATH)
		message(FATAL_ERROR "lcov not found! Aborting...")
	endif() # NOT LCOV_PATH

	if(NOT GENHTML_PATH)
		message(FATAL_ERROR "genhtml not found! Aborting...")
	endif() # NOT GENHTML_PATH

	# Setup target
	add_custom_target(${_targetname}
		# Cleanup lcov
		${LCOV_PATH} --directory . --zerocounters
		
		# Run tests
		COMMAND ${_testrunner} ${ARGV3}
		
		# Capturing lcov counters and generating report
		COMMAND ${LCOV_PATH} --directory . --capture --output-file 
                ${_outputname}.info
		COMMAND ${LCOV_PATH} --remove ${_outputname}.info 
                'tests/*' '/usr/*' --output-file ${_outputname}.info.cleaned
		COMMAND ${GENHTML_PATH} -o ${_outputname} ${_outputname}.info.cleaned
		COMMAND ${CMAKE_COMMAND} -E remove ${_outputname}.info 
                ${_outputname}.info.cleaned
		
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		COMMENT "Resetting code coverage counters to zero.\n"
                "Processing code coverage counters and generating report."
	)
	
	# Show info where to find the report
	add_custom_command(TARGET ${_targetname} POST_BUILD
		COMMAND ;
		COMMENT "Open ./${_outputname}/index.html in your browser to view "
                "the coverage report."
	)
endfunction() # SETUP_TARGET_FOR_COVERAGE


# Param _targetname     The name of new the custom make target
# Param _testrunner     The name of the target which runs the tests
# Param _outputname     cobertura output is generated as _outputname.xml
# Optional fourth parameter is passed as arguments to _testrunner
#   Pass them in list form, e.g.: "-j;2" for -j 2
function(SETUP_TARGET_FOR_COVERAGE_COBERTURA _targetname
                                             _testrunner
                                             _outputname)
	if(NOT PYTHON_EXECUTABLE)
		message(FATAL_ERROR "Python not found! Aborting...")
	endif() # NOT PYTHON_EXECUTABLE

	if(NOT GCOVR_PATH)
		message(FATAL_ERROR "gcovr not found! Aborting...")
	endif() # NOT GCOVR_PATH

	add_custom_target(${_targetname}
		# Run tests
		${_testrunner} ${ARGV3}

		# Running gcovr
		COMMAND ${GCOVR_PATH} -x -r ${CMAKE_SOURCE_DIR}
                -e '${CMAKE_SOURCE_DIR}/tests/'  -o ${_outputname}.xml
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		COMMENT "Running gcovr to produce Cobertura code coverage report."
	)

	# Show info where to find the report
	add_custom_command(TARGET ${_targetname} POST_BUILD
		COMMAND ;
		COMMENT "Cobertura code coverage report saved in ${_outputname}.xml."
	)
endfunction() # SETUP_TARGET_FOR_COVERAGE_COBERTURA

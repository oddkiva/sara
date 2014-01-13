cmake_minimum_required(VERSION 2.6)

project(DO_Core_test)

message(STATUS "  - DO_Core_test")

find_package(DO COMPONENTS Core REQUIRED)
include(${DO_USE_FILE})

set(TestGroupName "Core")
set(TestPch "CoreTesting.hpp")
do_unit_test(Core_Defines test_Defines.cpp "${DO_LIBRARIES}" 
             ${TestGroupName} ${TestPch})
do_unit_test(Core_EigenExtension test_EigenExtension.cpp "${DO_LIBRARIES}" 
             ${TestGroupName} ${TestPch})
do_unit_test(Core_Color test_Color.cpp "${DO_LIBRARIES}" 
             ${TestGroupName} ${TestPch})
do_unit_test(Core_Locator test_Locator.cpp "${DO_LIBRARIES}" 
             ${TestGroupName} ${TestPch})
do_unit_test(Core_MultiArray test_MultiArray.cpp "${DO_LIBRARIES}" 
             ${TestGroupName} ${TestPch})
do_unit_test(Core_Tree test_Tree.cpp "${DO_LIBRARIES}" 
             ${TestGroupName} ${TestPch})
do_unit_test(Core_Timer test_Timer.cpp "${DO_LIBRARIES}" 
             ${TestGroupName} ${TestPch})
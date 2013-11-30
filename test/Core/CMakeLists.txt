cmake_minimum_required(VERSION 2.6)

project(DO_Core_test)

message(STATUS "  - DO_Core_test")

find_package(DO COMPONENTS Core REQUIRED)
include(${DO_USE_FILE})

do_unit_test(Core_Defines test_Defines.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_Core_Defines_test PROPERTY FOLDER "DO Tests/Core")

do_unit_test(Core_EigenExtension test_EigenExtension.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_Core_EigenExtension_test PROPERTY FOLDER "DO Tests/Core")

do_unit_test(Core_Color test_Color.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_Core_Color_test PROPERTY FOLDER "DO Tests/Core")

do_unit_test(Core_Locator test_Locator.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_Core_Locator_test PROPERTY FOLDER "DO Tests/Core")

do_unit_test(Core_MultiArray test_MultiArray.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_Core_MultiArray_test PROPERTY FOLDER "DO Tests/Core")

do_unit_test(Core_Tree test_Tree.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_Core_Tree_test PROPERTY FOLDER "DO Tests/Core")

do_unit_test(Core_Timer test_Timer.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_Core_Timer_test PROPERTY FOLDER "DO Tests/Core")

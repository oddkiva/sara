cmake_minimum_required(VERSION 2.6)

project(DO_Core_tests)

message(STATUS "  - DO_Core_tests")

find_package(DO COMPONENTS Core REQUIRED)
include(${DO_USE_FILE})

set(TestGroupName "Core")

do_test(Core_Defines test_Defines.cpp "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_EigenExtension test_EigenExtension.cpp "${DO_LIBRARIES}"
        ${TestGroupName})

do_test(Core_Pixel "colorspace.hpp;pixel.hpp;test_pixel.cpp"
        "${DO_LIBRARIES}" ${TestGroupName})
        
do_test(Core_Pixel_ChannelConversion
        "channel_conversion.hpp;test_channel_conversion.cpp"
        "${DO_LIBRARIES}" ${TestGroupName})
        
do_test(Core_Pixel_ColorConversion
        "color_conversion.hpp;test_color_conversion.cpp"
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_ArrayIterators_Basic_Functions
        test_array_iterator_basic_functions.cpp "${DO_LIBRARIES}"
        ${TestGroupName})

do_test(Core_ArrayIterators_Classes test_array_iterator_classes.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_MultiArray test_multiarray.cpp "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_Tree test_Tree.cpp "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_Timer test_Timer.cpp "${DO_LIBRARIES}" ${TestGroupName})
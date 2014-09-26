cmake_minimum_required(VERSION 2.6)

project(DO_Core_tests)

message(STATUS "  - DO_Core_tests")

find_package(DO COMPONENTS Core REQUIRED)
include(${DO_USE_FILE})

set(TestGroupName "Core")

do_test(Core_Defines
        test_core_defines_and_macros.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_EigenExtension
        test_eigen_extension.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_Pixel_Class
        test_pixel.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_Pixel_PackedPixel "colorspace.hpp;packed_pixel.hpp;test_packed_pixel.cpp"
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_Pixel_ChannelConversion
        test_channel_conversion.cpp
        "${DO_LIBRARIES}" ${TestGroupName})
        
do_test(Core_Pixel_ColorConversion
        test_color_conversion.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_Pixel_Typedefs
        test_pixel_typedefs.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_ArrayIterators_Basic_Functions
        test_array_iterator_basic_functions.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_ArrayIterators_Classes
        test_array_iterator_classes.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_MultiArray
        test_multiarray.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_Image_Class
        test_image_class.cpp
        "${DO_LIBRARIES}" ${TestGroupName})
        
do_test(Core_Image_Operations
        test_image_operations.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_Tree
        test_tree.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

do_test(Core_Timer
        test_timer.cpp
        "${DO_LIBRARIES}" ${TestGroupName})

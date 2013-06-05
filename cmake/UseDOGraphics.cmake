# 1. Prepare macros.

macro (do_set_graphics_source_dir)
    set(DO_Graphics_SOURCE_DIR ${DO_SOURCE_DIR}/Graphics)
endmacro (do_set_graphics_source_dir)

macro (do_list_graphics_source_files)
    # Master header file
    set(DO_Graphics_MASTER_HEADER ${DO_SOURCE_DIR}/Graphics.hpp)
    source_group("0. Master Header File" FILES ${DO_Graphics_MASTER_HEADER})
    # API header files
    file(GLOB DO_Graphics_API_HEADER_FILES
         ${DO_Graphics_SOURCE_DIR}/*.hpp)
    source_group("1. API Header Files" 
                 FILES ${DO_Graphics_API_HEADER_FILES})
    # API source files
    file(GLOB DO_Graphics_API_SOURCE_FILES
         ${DO_Graphics_SOURCE_DIR}/*.cpp)
    source_group("1. API Source Files" 
                 FILES ${DO_Graphics_API_SOURCE_FILES})
    # Derived QObjects header files
    file(GLOB DO_Graphics_DerivedQObjects_HEADER_FILES
         ${DO_Graphics_SOURCE_DIR}/DerivedQObjects/*.hpp)
    source_group("2. Derived QObjects Header Files"
                 FILES ${DO_Graphics_DerivedQObjects_HEADER_FILES})
    # Derived QObjects source files
    file(GLOB DO_Graphics_DerivedQObjects_SOURCE_FILES
         ${DO_Graphics_SOURCE_DIR}/DerivedQObjects/*.cpp)
    source_group("2. Derived QObjects Source Files"
                 FILES ${DO_Graphics_DerivedQObjects_SOURCE_FILES})
    # All files here
    set(DO_Graphics_FILES
        ${DO_Graphics_MASTER_HEADER}
        ${DO_Graphics_API_HEADER_FILES}
        ${DO_Graphics_API_SOURCE_FILES}
        ${DO_Graphics_DerivedQObjects_HEADER_FILES}
        ${DO_Graphics_DerivedQObjects_SOURCE_FILES})
endmacro (do_list_graphics_source_files)

macro (do_load_packages_for_graphics_library)
    if (DEFINED ENV{QT5_DIR})
        list(APPEND CMAKE_PREFIX_PATH $ENV{QT5_DIR})
    else ()
        message("WARNING: it is recommended that you set QT5_DIR to help cmake find Qt 5: ${QT5_DIR} is then appended to CMAKE_PREFIX_PATH variable to locate Qt 5.")
    endif ()
    find_package(Qt5Widgets REQUIRED)
    find_package(Qt5OpenGL REQUIRED)
    find_package(OpenGL REQUIRED)
    include_directories(${Qt5Widgets_INCLUDE_DIRS}
                        ${Qt5OpenGL_INCLUDE_DIRS})
    include(${DO_Core_USE_FILE})
endmacro (do_load_packages_for_graphics_library)

macro (do_create_variables_for_graphics_library)
    # Use this variable if you want to link statically DO_Graphics
    set(DO_Graphics_LIBRARIES DO_Graphics)
    # External libraries which DO_Graphics depends on.
    list(APPEND DO_Graphics_LINK_LIBRARIES ${OPENGL_LIBRARIES}) # Actually there is also Qt but I use qt5_use_modules which takes care of the linking with Qt5.
endmacro (do_create_variables_for_graphics_library)





# 2. Setup the project by calling the macros
do_load_packages_for_graphics_library()

if (DO_USE_FROM_SOURCE)
    get_property(DO_Graphics_ADDED GLOBAL PROPERTY _DO_Graphics_INCLUDED)
    if (NOT DO_Graphics_ADDED)
        do_set_graphics_source_dir()
        do_list_graphics_source_files()
        do_create_variables_for_graphics_library()
    endif ()
    if (NOT DO_Graphics_ADDED)
        set_property(GLOBAL PROPERTY _DO_Graphics_INCLUDED 1)

        # Moc files
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_INCLUDE_CURRENT_DIR ON)
        source_group("Moc Files" FILES ${DOGraphics_MOC_SOURCES})

        # Static library
        add_library(DO_Graphics STATIC
                    ${DO_Graphics_FILES} ${DOGraphics_MOC_SOURCES})
        qt5_use_modules(DO_Graphics Widgets OpenGL)
        target_link_libraries(DO_Graphics ${OPENGL_LIBRARIES})
        do_set_specific_target_properties(DO_Graphics DO_STATIC) # See DOMacros.cmake for details
        
        # Shared library
        if (DO_BUILD_SHARED_LIBS)
            add_library(DO_Graphics_SHARED SHARED
                        ${DO_Graphics_FILES} ${DOGraphics_MOC_SOURCES})
            qt5_use_modules(DO_Graphics_SHARED Widgets OpenGL)    
            target_link_libraries(DO_Graphics_SHARED ${OPENGL_LIBRARIES})
            do_set_specific_target_properties(DO_Graphics_SHARED DO_EXPORTS)
        endif ()
        
        
        set(CMAKE_AUTOMOC OFF)
        set(CMAKE_INCLUDE_CURRENT_DIR OFF)
    endif()
endif ()
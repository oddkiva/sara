do_substep_message("UseDO running for project '${PROJECT_NAME}'")

if (POLICY CMP0011)
    cmake_policy(SET CMP0011 OLD)
endif (POLICY CMP0011)

set(DO_LIBRARIES "")
foreach (COMPONENT ${DO_USE_COMPONENTS})
    include(UseDO${COMPONENT})
    if ("${DO_LIBRARIES}" STREQUAL "" AND 
        NOT "${DO_${COMPONENT}_LIBRARIES}" STREQUAL "")
        set (DO_LIBRARIES "${DO_${COMPONENT}_LIBRARIES}")
    elseif (NOT "${DO_${COMPONENT}_LIBRARIES}" STREQUAL "")
        set(DO_LIBRARIES "${DO_LIBRARIES};${DO_${COMPONENT}_LIBRARIES}")
    endif ()
endforeach (COMPONENT)



# Not stable yet.
macro (do_use_modules TARGET COMPONENT_LIST)
    # Include the following useful macros.
    set_target_properties(${TARGET} PROPERTIES
                          COMPILE_FLAGS -DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR})

    if (DO_USE_FROM_SOURCE)
        set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS DO_STATIC)
        target_link_libraries(${TARGET} ${COMPONENT_LIST})
    else ()
        # Warn the user that precompiled libraries does not work for every build types...
        message (WARNING "Precompiled libraries only work for DEBUG or RELEASE builds... If you want to use other build modes, you need to recompile DO++ from the sources. To do so, just insert at the top of the CMakeLists.txt: '(set DO_USE_FROM_SOURCE 1)'")

        # Static or dynamic linking?
        if (DO_USE_STATIC_LIBS)
            set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS DO_STATIC)
        endif ()

        # Set library paths
        if (WIN32)
            set(FLAG /LIBPATH:"${DO_DIR}/lib")
        else ()
            set(FLAG -L"${DO_DIR}/lib")
        endif ()
        set_target_properties(${TARGET} PROPERTIES LINK_FLAGS ${FLAG})        

        # Link with libraries listed in ${COMPONENT_LIST}
        foreach (f ${COMPONENT_LIST})
            if ("${f}" STREQUAL "DO_Graphics")
                qt5_use_modules(${TARGET} Widgets OpenGL) # Link with Qt5
                target_link_libraries(${TARGET} ${OPENGL_LIBRARIES})
            endif ()
            
            if (NOT "${${f}_LIBRARIES}" STREQUAL "" AND NOT DO_USE_STATIC_LIBS)
                if (WIN32)
                    target_link_libraries(${TARGET} 
                        debug ${f}_SHARED-1.0.0-debug
                        optimized ${f}_SHARED-1.0.0-release)
                else ()
                    target_link_libraries(${TARGET} ${f}_SHARED-1.0.0)
                endif ()
            elseif (NOT "${${f}_LIBRARIES}" STREQUAL "")
                 if (WIN32)
                    target_link_libraries(${TARGET} debug ${f}-1.0.0-debug
                                                    optimized ${f}-1.0.0-release)
                else ()
                    target_link_libraries(${TARGET} ${f}-1.0.0)
                endif ()
            endif ()
        endforeach ()
    endif ()
endmacro (do_use_modules)
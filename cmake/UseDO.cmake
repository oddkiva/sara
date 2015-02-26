do_substep_message("UseDO running for project '${PROJECT_NAME}'")

if (POLICY CMP0011)
  cmake_policy(SET CMP0011 OLD)
endif (POLICY CMP0011)


# Retrieve the set of dependencies when linking projects with DO-CV.
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

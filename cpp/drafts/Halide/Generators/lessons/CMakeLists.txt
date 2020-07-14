find_package(DO_Sara COMPONENTS Core Graphics REQUIRED)

file(GLOB LESSONS_SOURCE_FILES
  lesson_01*.cpp
  lesson_02*.cpp
  lesson_03*.cpp
  lesson_04*.cpp
  lesson_05*.cpp
  lesson_06*.cpp
  lesson_07*.cpp
  lesson_08*.cpp
  # lesson_09*.cpp
  # lesson_10*.cpp
  lesson_11*.cpp
  lesson_12*.cpp
  lesson_13*.cpp
  lesson_14*.cpp
  # lesson_15*.cpp
  # lesson_16*.cpp
  lesson_17*.cpp
  lesson_18*.cpp
  lesson_19*.cpp
  lesson_20*.cpp
  # lesson_21*.cpp
)


function (add_halide_executable source_file)
  get_filename_component(name ${source_file} NAME_WE)
  add_executable(${name} ${source_file})
  target_include_directories(${name}
    PRIVATE
    ${HALIDE_DISTRIB_DIR}/include
    ${HALIDE_DISTRIB_DIR}/tools)

  target_link_libraries(${name}
    PRIVATE
    ${DO_Sara_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${PNG_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    Halide)

  target_compile_options(${name}
    PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-but-set-variable>)

  set_property(TARGET ${name} PROPERTY FOLDER "Tutorials/Halide")
endfunction ()

foreach (lesson ${LESSONS_SOURCE_FILES})
  add_halide_executable(${lesson})
endforeach ()

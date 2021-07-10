find_package(DO_Sara COMPONENTS Core Graphics ImageIO VideoIO REQUIRED)

function (add_halide_executable source_file)
  get_filename_component(name ${source_file} NAME_WE)
  add_executable(${name} ${source_file})
  target_link_libraries(${name}
    PRIVATE
    ${CMAKE_DL_LIBS}
    ${DO_Sara_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    Halide::Halide)

  target_compile_options(${name}
    PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-but-set-variable>)
  target_compile_features(${name}
    PRIVATE
    cxx_std_17)

  set_property(TARGET ${name} PROPERTY FOLDER "Tutorials/Halide")
endfunction ()

file(GLOB examples_SOURCE_FILES *.cpp)
foreach (example ${examples_SOURCE_FILES})
  add_halide_executable(${example})
endforeach ()

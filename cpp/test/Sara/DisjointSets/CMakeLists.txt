file(GLOB test_disjoint_sets_SOURCE_FILES FILES *.cpp)

foreach (file ${test_disjoint_sets_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::DisjointSets
    FOLDER DisjointSets)
endforeach ()

target_link_libraries(test_disjointsets_algorithm_v2
  PRIVATE
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)

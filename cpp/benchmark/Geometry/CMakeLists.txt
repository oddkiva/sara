macro (add_benchmark NAME)
  add_executable(${NAME} ${NAME}.cpp)
endmacro (add_benchmark)

add_benchmark(benchmark_ellipse_intersections)

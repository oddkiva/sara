sara_add_test(
  NAME test_halide_blur
  SOURCES test_halide_blur.cpp
  DEPENDENCIES ${DO_Sara_LIBRARIES} Halide
  FOLDER "DRAFTS/Halide")

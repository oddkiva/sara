file(GLOB stlplus_SOURCE_FILES FILE portability/*.cpp portability/*.hpp)

add_library(stlplus ${stlplus_SOURCE_FILES})

file(GLOB MultiArray_SRC_FILES FILES *.hpp *.cu)

cuda_add_library(DO_Shakti_MultiArray ${MultiArray_SRC_FILES})
set_property(TARGET DO_Shakti_MultiArray PROPERTY FOLDER "CUDA Libraries")

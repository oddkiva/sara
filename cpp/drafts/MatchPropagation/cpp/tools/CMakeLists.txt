add_executable(DO_MikolajczykDataset_test DO_MikolajczykDataset_test.cpp)
do_use_modules(DO_MikolajczykDataset_test "${DO_Sara_LIBRARIES};DO_MikolajczykDataset")
set_property(TARGET DO_MikolajczykDataset_test PROPERTY FOLDER "DO 4th-Order Match Propagation")

add_executable(DO_Region_test DO_Region_test.cpp)
do_use_modules(DO_Region_test "${DO_Sara_LIBRARIES};DO_MikolajczykDataset;DO_Region")
set_property(TARGET DO_Region_test PROPERTY FOLDER "DO 4th-Order Match Propagation")

add_executable(DO_GrowRegion_test DO_GrowRegion_test.cpp)
do_use_modules(DO_GrowRegion_test
  "${DO_Sara_LIBRARIES};DO_GrowRegion;DO_MikolajczykDataset;DO_Region;DO_LoweSIFTWrapper")
set_property(TARGET DO_GrowRegion_test PROPERTY FOLDER "DO 4th-Order Match Propagation")


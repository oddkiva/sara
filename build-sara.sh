mkdir ../sara-build-shared
cd ../sara-build-shared
cmake ../sara \
  -DCMAKE_BUILD_TYPE=Release \
  -DSARA_BUILD_SHARED_LIBS=ON \
  -DSARA_BUILD_TESTS=ON \
  -DSARA_BUILD_SAMPLES=ON

make -j`nproc` && make test && make package

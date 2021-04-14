add_executable(nuscenes_cpp_devkit_usage_example nuscenes_cpp_devkit_usage_example.cpp)
target_compile_features(nuscenes_cpp_devkit_usage_example
  PRIVATE
  cxx_std_17)
target_include_directories(nuscenes_cpp_devkit_usage_example
  PRIVATE
  ${cpp_INCLUDE_DIR}/third-party
  ${cpp_INCLUDE_DIR}/third-party/eigen)
target_link_libraries(nuscenes_cpp_devkit_usage_example Boost::filesystem)

#include <drafts/NuScenes/NuScenes.hpp>


int main()
{
  using namespace std::string_literals;

  const auto nuscenes_version = "v1.0-mini"s;
  const auto nuscenes_root_path = "/home/david/Downloads/nuscenes"s;

  const auto nuscenes = NuScenes{nuscenes_version, nuscenes_root_path, true};

  const auto& sample_token = nuscenes.samples.begin()->first;
  std::cout << "sample_token = " << sample_token << std::endl;

  const auto sample_data = nuscenes.filter_by_sample_token(  //
      nuscenes.sample_data,                                  //
      sample_token                                           //
  );
  std::cout << "sample_data = " << sample_data.size() << std::endl;

  const auto sample_annotation = nuscenes.filter_by_sample_token(  //
      nuscenes.sample_annotations,                                 //
      sample_token                                                 //
  );
  std::cout << "sample_annotation = " << sample_annotation.size() << std::endl;

  return 0;
}

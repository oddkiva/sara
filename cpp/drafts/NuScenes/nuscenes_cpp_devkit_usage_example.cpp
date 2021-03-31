#include <drafts/NuScenes/NuScenes.hpp>


int main()
{
  using namespace std::string_literals;

  const auto nuscenes_version = "v1.0-mini"s;
  const auto nuscenes_root_path = "/home/david/Downloads/nuscenes"s;

  const auto nuscenes = NuScenes{nuscenes_version, nuscenes_root_path, true};

  return 0;
}

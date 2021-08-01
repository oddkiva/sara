#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>

#include <drafts/NuScenes/NuImages.hpp>


namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
  using namespace std::string_literals;

  const auto nuimages_version = "v1.0-mini"s;
  const auto nuimages_root_path = "/Users/david/Downloads/nuimages-v1.0-mini"s;
  const auto nuimages = NuImages{nuimages_version, nuimages_root_path, true};



  return 0;
}

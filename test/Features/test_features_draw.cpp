#include <gtest/gtest.h>

#include <DO/Sara/Features/Draw.hpp>

#include <DO/Sara/Graphics.hpp>


using namespace std;
using namespace DO::Sara;


TEST(TestOERegionDrawing, test_draw_method)
{
  create_window(300, 300);
  auto f = OERegion{ Point2f{ 300 / 2.f, 300 / 2.f }, 1.f };
  f.draw(Red8);
}

TEST(TestOERegionDrawing, test_draw_oe_regions)
{
  create_window(300, 300);
  auto features = vector<OERegion>{
    OERegion{ Point2f{ 300 / 2.f, 300 / 2.f }, 1.f },
    OERegion{ Point2f{ 200.f, 300 / 2.f }, 1.f }
  };
  draw_oe_regions(features, Red8);
}

int worker_thread(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

int main(int argc, char **argv)
{
  GraphicsApplication gui_app(argc, argv);
  gui_app.register_user_main(worker_thread);
  return gui_app.exec();
}

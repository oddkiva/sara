#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/InfiniteImage.hpp>

using namespace std;
using namespace DO::Sara;


GRAPHICS_MAIN()
{
  auto image = Image<Rgb8>{};
  imread(image, "/home/david/GitHub/DO-CV/sara/data/stinkbug.png");

  auto pad = PeriodicPadding{};
  auto inf_image = make_infinite(image, pad);

  const auto repeat = 2;
  const auto border = Vector2i::Ones() * 50;

  const Vector2i begin{-border};
  const Vector2i end = image.sizes() + border;
  cout << begin.transpose() << endl;
  cout << end.transpose() << endl;

  auto ext_image = Image<Rgb8>{end - begin};
  cout << (end-begin).transpose() << endl;

  Timer t;
  double start, finish;
  t.restart();

  start = t.elapsed_ms();

  for (int i = 0; i < 10; ++i)
  {
    auto src_c = CoordsIterator<MultiArrayView<Rgb8, 2, ColMajor>>{begin, end};
    auto dst_i = ext_image.begin_array();
    for (; !dst_i.end(); ++src_c, ++dst_i)
      *dst_i = inf_image(*src_c);
  }

  finish = t.elapsed_ms();
  std::cout << (finish - start) / 10 << " ms" << std::endl;

  // Resize image.
  create_window(ext_image.sizes());
  display(ext_image);
  get_key();

  close_window();

  return EXIT_SUCCESS;
}

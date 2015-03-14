#include <DO/Core/Timer.hpp>
#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{
  Image<Color3ub> I;
  cout << src_path("../../datasets/ksmall.jpg") << endl;
  if ( !load(I, src_path("../../datasets/ksmall.jpg")) )
  {
    cerr << "Error: could not open 'ksmall.jpg' file" << endl;
    return 1;
  }
  int w = I.width(), h = I.height();
  int x = 0, y = 0;

  create_window(2*w, h);

  Timer drawTimer;
  drawTimer.restart();
  double elapsed;
  for (int i = 0; i < 1; ++i)
  {
    clear_window();
    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        draw_point(x, y, I(x,y));
        draw_point(w+x, y, I(x,y));
#ifdef Q_OS_MAC
        microsleep(10);
#endif
      }
    }
  }

  elapsed = drawTimer.elapsed();
  std::cout << "Drawing time: " << elapsed << "s" << std::endl;

  click();

  int step = 2;
  Timer t;
  clear_window();
  while (true)
  {
    microsleep(10);
    display(I, x, y);
    clear_window();

    x += step;
    if (x < 0 || x > w)
      step *= -1;
    //cout << x << endl;

    if (t.elapsed() > 2)
      break;
  }
  close_window(active_window());

  cout << "Finished!" << endl;

  return 0;
}
#include <DO/Core/Timer.hpp>
#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;
 
int main()
{
  Image<Color3ub> I;
  cout << srcPath("../../datasets/ksmall.jpg") << endl;
  if ( !load(I, srcPath("../../datasets/ksmall.jpg")) )
  {
    cerr << "Error: could not open 'ksmall.jpg' file" << endl;
    return 1;
  }
  int w = I.width(), h = I.height();
  int x = 0, y = 0;

  openWindow(2*w, h);

  Timer drawTimer;
  drawTimer.restart();
  double elapsed;
  for (int i = 0; i < 1; ++i)
  {
    clearWindow();
    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        drawPoint(x, y, I(x,y));
        drawPoint(w+x, y, I(x,y));
#ifdef Q_OS_MAC
        microSleep(10);
#endif
      }
    }
  }

  elapsed = drawTimer.elapsed();
  std::cout << "Drawing time: " << elapsed << "s" << std::endl;

  click();

  int step = 2;
  Timer t;
  clearWindow();
  while (true)
  {
    microSleep(10);
    display(I, x, y);
    clearWindow();

    x += step;
    if (x < 0 || x > w)
      step *= -1;
    //cout << x << endl;

    if (t.elapsed() > 2)
      break;
  }
  closeWindow(getActiveWindow());

  cout << "Finished!" << endl;

  return 0;
}
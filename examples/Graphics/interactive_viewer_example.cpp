#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

int main()
{
  Image<Rgb8> I;
  load(I, srcPath("../../datasets/ksmall.jpg"));

  openGraphicsView(I.width(), I.height());

  for (int i = 0; i < 10; ++i)
  {
    ImageItem image = addImage(I);
    if (!image)
      cerr << "Error image display" << endl;
  }
  
  while (getKey() != KEY_ESCAPE);
  closeWindow();

  return 0;
}
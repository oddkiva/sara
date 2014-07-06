// ========================================================================= //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================= //

#define TEST_2D
#define TEST_3D
#define TEST_GRAPHICSVIEW

#include "precompiled.hpp"

using namespace std;
using namespace DO;

#ifdef TEST_2D
void twoDimBasics()
{
  cout << "Basic 2D drawings ... click when done" << endl;

  Window W = openWindow(512, 512, "2D basics");

  // Draw a red line from (20, 10) to (300, 100) with 5-pixel thickness.
  drawLine(20, 10, 300, 100, Red8, 5);
  // Draw a black line from (320, 100) to (500, 100) with 5-pixel thickness.
  drawLine(Point2i(320, 100), Point2i(500, 100), Black8, 5);

  // Draw a blue rectangle with top-left corner (400, 10) and size (100, 50).
  drawRect(400, 10, 100, 50, Blue8, 3);
  // Draw a green color-filled rectangle with top-left corner (400, 400) and
  // size (100, 50).
  fillRect(Point2i(400, 400), 100, 50, Green8);

  // Draw an axis-aligned ellipse bounded by a rectangle whose top-left 
  // corner is (50,350) and size is (50, 90) using a cyan Pen with a 2-pixel
  // thickness.
  drawEllipse(50, 350, 50, 90, Cyan8, 2);
  // Simple exercise: decipher this one.
  fillEllipse(350, 150, 90, 100, Color3ub(128, 128, 128));
  // A circle with a center point located at (200, 200) and a 40-pixel radius.
  drawCircle(Point2i(200, 200), 40, Red8);

  /*
   Draw an oriented ellipse with
   - center = (150, 100)
   - radii r1 = 10, r2 = 20,
   - orientation = 45 degree
   - in cyan color,
   - pencil width = 1.
   */
  drawEllipse(Vector2f(150.f, 100.f), 10.f, 20.f, 45.f, Cyan8, 1);
  drawEllipse(Vector2f(50.f, 50.f), 10.f, 20.f, 0.f, Red8, 1);

  // Draw a few black points.
  for (int i = 0; i < 20; i+= 2)
    drawPoint(i+100, i+200, Black8);
  // Draw a string.
  drawString(50, 250, "a string", Red8);
  // Draw another string but with font size=18 and in italic.
  drawString(40, 270, "another string", Magenta8, 18, 0, true);   
  // ... font size=24, rotation angle=-10, bold
  drawString(30,300,"yet another string",Black8,24,-10,
             false,true);
  // Draw a polygon with the following points.
  int px[] = { 201, 200, 260, 240 };
  int py[] = { 301, 350, 330, 280 };
  fillPoly(px, py, 4, Blue8);
  // Draw another polygon.
  //      { (x1, y1), (x2, y2), (x3, y3) }
  int t[]={ 300, 300, 300, 400, 400, 350 };
  fillPoly(t, 3, Green8);
  // Draw another polygon.
  Point2i P[]={
    Point2i(100, 100),
    Point2i(100, 150),
    Point2i(150, 120)
  };
  drawPoly(P, 3, Red8, 3);  // ... with a red pen with 3-pixel thickness.

  // Draw a blue arrow from (100,450) to (200,450).
  drawArrow(100, 470, 200, 450, Blue8);
  // Draw a red arrow with the a 30x10 pixels with style 1.
  // TODO: improve this API.
  drawArrow(300, 470, 200, 450, Red8, 30, 10, 1);
  drawArrow(200, 450, 250, 400, Black8, 20, 20, 2);
  // Draw another arrow with tip: (angle,length)=(35,8) , style=0, width=2.
  // TODO: improve this **horrible** legacy API.
  drawArrow(200, 450, 150, 400, Green8, 35., 8., 0, 2); 

  click();
  closeWindow(W);
}

void floatingPointDrawing()
{
  // Open a 300x200 window.
  Window W = openWindow(300, 200);
  setAntialiasing(getActiveWindow());
  setTransparency(getActiveWindow());

  drawPoint(Point2f(10.5f, 10.5f), Green8);
  drawPoint(Point2f(20.8f, 52.8132f), Green8);

  drawLine(Point2f(10.5f, 10.5f), Point2f(20.8f, 52.8132f), Blue8, 2);
  drawLine(Point2f(10.5f, 20.5f), Point2f(20.8f, 52.8132f), Magenta8, 5);

  // Draw an oriented ellipse with:
  // center = (150, 100)
  // r1 = 10
  // r2 = 20
  // orientation = 45°
  // in cyan color, and a pencil width = 1.
  drawEllipse(Point2f(150.f, 100.f), 10.f, 20.f, 45.f, Cyan8, 1);
  drawEllipse(Point2f(50.f, 50.f), 10.f, 20.f, 0.f, Red8, 1);

  fillCircle(Point2f(100.f, 100.f), 10.f, Blue8);
  fillEllipse(Point2f(150.f, 150.f), 10.f, 20.f, 72.f, Green8);

  Point2f p1(rand()%300, rand()%200);
  Point2f p2(rand()%300, rand()%200);
  drawPoint((p1*2+p2)/2, Green8);

  click();
  closeWindow(W);
}

void bitmapBasics()
{
  cout << "Basic bitmap drawings... click when done" << endl;
  Window W = openWindow(512, 512, "Bitmaps");
  // Array of bytes
  Color3ub cols[256*256];
  // Some (RED,GREEN,BLUE) function of (i,j)
  for (int j = 0; j < 256; j++)
    for (int i = 0; i < 256; i++)
      cols[i+256*j]= Color3ub(i, 255-i, (j<128)?255:0);
  // Draw this 256x256 (r,g,b) bitmap in (0,0)
  putColorImage(0, 0, cols, 256, 256);

  // An array of colors.
  // Color3ub = 3D color vector where each channel has a value in [0,255].
  Color3ub cols2[256*256];
  for (int j = 0; j < 256; j++) 
    for (int i = 0; i < 256; i++) 
      cols2[i+256*j]=Color3ub(i, (2*j)%256, (i+j)%256);  // RGB colors.
  // Display the bitmap from the following top-left corner (0,256)
  // TODO: rename this function.
  putColorImage(Point2i(0, 256), cols2, 256, 256);

  // A grayscale image.
  unsigned char grey[256*256];
  for (int j = 0; j < 256; ++j) 
    for (int i = 0; i < 256; ++i) 
      grey[i+256*j] = static_cast<unsigned char>(128+127*sin((i+j)/10.));
  // Display the bitmap from the following top-left corner (0,256)
  // TODO: rename this function.
  putGreyImage(256 ,0, grey, 256, 256); // Draw at point (256,0);

  click();
  closeWindow(W);
}

void mouseBasics()
{
  cout << "Basic mouse functions" << endl;

  Window W = openWindow(512, 512, "Mouse");
  drawString(10, 10, "Please click anywhere", Black8);

  click();  

  drawString(10, 40, "click again (left=BLUE, middle=RED, right=done)",
             Black8);

  int button;    
  Point2i p;
  while ((button=getMouse(p)) != MOUSE_RIGHT_BUTTON)
  {
    Rgb8 color;
    if (button == MOUSE_LEFT_BUTTON)
      color = Blue8;
    else if (button == MOUSE_MIDDLE_BUTTON)
      color = Red8;
    else
      color = Black8;
    fillCircle(p, 5, color);
  }

  closeWindow(W);
}

void imageAnimation()
{
  Image<Color3ub> I;
  cout << srcPath("../../datasets/ksmall.jpg") << endl;
  if ( !load(I, srcPath("../../datasets/ksmall.jpg")) )
  {
    cerr << "Error: could not open 'ksmall.jpg' file" << endl;
    return;
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
}

void naiveAnimation()
{
  Window w = openWindow(300, 300);
  setActiveWindow(w);

  Event e;
  do
  {
    getEvent(1, e);
    fillRect(rand()%300, rand()%300, rand()%50, rand()%50,
      Color3ub(rand()%256, rand()%256, rand()%256));
    //microSleep(100);  // TODO: sometimes if you don't put this, the program
                        // freezes in some machine. Investigate.
  } while (e.key != KEY_ESCAPE);

  cout << "Finished!" << endl;

  closeWindow(getActiveWindow());
}

void advancedEvents()
{
  cout << "Advanced event handling" << endl;
  Window W = openWindow(1024, 768);
  setActiveWindow(W);
  Image<Color3ub> I;
  load(I, srcPath("../../datasets/ksmall.jpg"));

  Event ev;
  do {
    getEvent(500,ev); // Wait an event (return if no event for 500ms)
    switch (ev.type){
    case NO_EVENT:
      break;
    case MOUSE_PRESSED_AND_MOVED:
      clearWindow();
      display(I, ev.mousePos-I.sizes()*3/8, 0.75);
      cout << "Mouse moved. Position = " << endl << ev.mousePos << endl;
      break;
    case KEY_PRESSED:
      cout << "Key " << ev.key << " pressed"<< endl;
      break;
    case KEY_RELEASED:
      cout << "Key " << ev.key << " released"<< endl;
      break;
    case MOUSE_PRESSED:
      clearWindow();
      display(I, ev.mousePos-I.sizes()*3/8, 0.75);
      cout << "Button " << ev.buttons << " pressed"<< endl;
      break;
    case MOUSE_RELEASED:
      cout << "Button " << ev.buttons << " released"<< endl;
      break;
    }
  } while (ev.type != KEY_PRESSED || ev.key != KEY_UP);
  closeWindow(W);
}
 
void imageDrawing()
{
  Image<Rgb8> image(300, 300);
  image.array().fill(White8);

  openWindow(300, 300);
  display(image);

  for (int y = 0; y < 300; ++y)
  {
    for (int x = 0; x < 300; ++x)
    {
      Color3ub c(rand()%256, rand()%256, rand()%256);
      drawPoint(image, x, y, c);
    }
  }
  display(image);
  getKey();

  image.array().fill(White8);
  display(image);
  getKey();

  for (int i = 0; i <10; ++i)
  {
    int x, y, w, h;
    x = rand()%300;
    y = rand()%300;
    w = rand()%300;
    h = rand()%300;
    Color3ub c(rand()%256, rand()%256, rand()%256);
    fillRect(image, x, y, w, h, c);
  }
  display(image);
  getKey();
  closeWindow();
}

void loadImageFromDialogBox()
{
  Image<Rgb8> I;
  if (!loadFromDialogBox(I))
    return;
  openWindow(I.width(), I.height(), "Image loaded from dialog box");
  display(I);
  getKey();
  closeWindow();
}
#endif

#ifdef TEST_3D
void checkOpenGLWindow()
{
  Window w = openGLWindow(300, 300);
  setActiveWindow(w);

  SimpleTriangleMesh3f mesh;
  string filename = srcPath("../../datasets/pumpkin_tall_10k.obj");
  if (!MeshReader().readObjFile(mesh, filename))
  {
    cout << "Error reading mesh file:\n" << filename << endl;
    closeWindow();
    return;
  }
  cout << "Read " << filename << " successfully" << endl;

  displayMesh(mesh);

  bool quit = false;
  while (!quit)
  {
    int c = getKey();
    quit = (c==KEY_ESCAPE || c==' ');
  }
  closeWindow();
}
#endif

#ifdef TEST_GRAPHICSVIEW
void graphicsViewExample()
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
}
#endif

int main()
{
  using namespace DO;
#ifdef TEST_2D
  // Window management examples.
  aWindow();
  twoWindows();
  twoWindows2();
  multipleWindows();
  // Drawing examples.
  twoDimBasics();
  floatingPointDrawing();
  // Image display examples
  bitmapBasics(); 
  imageAnimation();
  naiveAnimation();
  imageDrawing();  
  // I/O management examples.
  mouseBasics();
  advancedEvents();
  // 
  loadImageFromDialogBox();
#endif

#ifdef TEST_3D
  checkOpenGLWindow();
#endif

#ifdef TEST_GRAPHICSVIEW
  graphicsViewExample();
#endif

  return 0;
}
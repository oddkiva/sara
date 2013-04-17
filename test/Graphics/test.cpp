// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Graphics.hpp>

using namespace std;

BEGIN_NAMESPACE_DO

const bool stepByStep = true;

#define TEST_2D
#define TEST_3D
#define TEST_GRAPHICSVIEW

#ifdef TEST_2D
void aWindow()
{
	cout << "A window... click when done" << endl;
	// Open 300x200 window
	Window W = openWindow(300,200, "A window");
	// A 150x100 filled RED rectangle in (20,10)
	fillRect(20, 10, 150, 100, Red8);
	// Wait for a click
	if (stepByStep)
		click();
	// Close window...
	closeWindow(W);
}

void twoWindows()
{
	cout << "Two windows ... click when done" << endl;
	// A 300x200 window in (10,10)
	Window W1=openWindow(300,200,"A first window",10,10);
	// A 150x100 RED rectangle in (20,10) (up-left corner)
	drawRect(20,10,150,100,Red8);
	// A 200x300 window in (320,10)
	Window W2=openWindow(200,300,"A second window",330,10);
	// Note that openWindow does not make the new window the current one
	setActiveWindow(W2);
	// A BLUE line from (20,10) to (150,270)
	drawLine(20,10,150,270,Blue8);
	setActiveWindow(W1);
	// Another line, in GREEN, in the first window
	drawLine(20,10,250,170,Green8);
	// Wait for a click in any window
	if (stepByStep) anyClick();
	closeWindow(W1);
	closeWindow(W2);
}

void twoWindows2()
{
  openWindow(300, 300);
  getKey();

  Window w1 = activeWindow();
  Window w2 = openWindow(100, 100);
  setActiveWindow(w2);
  getKey();
  closeWindow(w2);

  setActiveWindow(w1);
  drawCircle(120, 120, 30, Red8);
  getKey();
  closeWindow();
}

void multipleWindows()
{
	vector<Window> windows;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
	{
		windows.push_back(
			openWindow(200, 200,
					   "Window #" + toString(i*3+j),
					   300*j+300, 300*i+50));
		setActiveWindow(windows.back());
		fillRect(0, 0, windows.back()->width(), windows.back()->height(),
				 Color3ub(rand()%255, rand()%255, rand()%255));
		drawString(100, 100, toString(i*3+j), Yellow8, 15);
		cout << "Pressed '" << char(anyGetKey()) << "'" << endl;
	}

	setActiveWindow(windows.back());
	anyClick();

	for (size_t i = 0; i < windows.size(); ++i)
	{
		anyGetKey();
		cout << "Closing window #" << i << endl;
		closeWindow(windows[i]);
	}
}

void twoDimBasics() {
	cout << "Basic 2D drawings ... click when done" << endl;
	Window W=openWindow(512,512,"2D basics");
	// Lines
	drawLine(20,10,300,100,Red8,5);							            // Line (20,10)-(300,100) in RED, thickness=5
	drawLine(Point2i(320,100),Point2i(500,100),Black8,5);	  // Specify 2 pixels instead of 4 coords
	// Rects
	drawRect(400,10,100,50,Blue8,3);						            // A 100x50 rectangle in (400,10) (up-left corner)
	fillRect(Point2i(400,400),100,50,Green8);				        // A filled rectangle
	// Ellipspe
	drawEllipse(50,350,50,90,Cyan8,2);						          // A 50x90 ellipse in (50,350) (up-left corner), thick=2
	drawCircle(Point2i(200,200),40,Red8);					          // A circle centered in (200,200), radius=40 	
	fillEllipse(350,150,90,100,Color3ub(128,128,128));		  // A filled grey ellipse
  drawEllipse(Vector2f(150.f, 100.f), 10.f, 20.f, 45.f, Cyan8, 1); // an oriented ellipse whose
                                                                   // center = (150, 100)
                                                                   // r1 = 10
                                                                   // r2 = 20
                                                                   // orientation = 45°
                                                                   // in cyan color, 
                                                                   // and a pencil width = 1.
  drawEllipse(Vector2f(50.f, 50.f), 10.f, 20.f, 0.f, Red8, 1);
	// Points
	for (int i=0;i<20;i+=2)
		drawPoint(i+100,i+200,Black8);						            // Some BLACK points
	// Strings
	drawString(50,250,"a string",Red8);						          // A RED string in (50,200)
	drawString(40,270,"another string",Magenta8,18,0,true);	// size=18, italic
	// size=24, angle=-10, bold
	drawString(30,300,"yet another string",Black8,24,-10,false,true);
	// Polygons
	int px[]={201,200,260,240},py[]={301,350,330,280};
	fillPoly(px,py,4,Blue8);								                // A filled polygon (px[i],py[i])
	int t[]={300,300,300,400,400,350};
	fillPoly(t,3,Green8);									                  // A filled polygon (t[2*i],t[2*i+1])
	Point2i P[]={Point2i(100,100),Point2i(100,150),Point2i(150,120)};
	drawPoly(P,3,Red8,3);									                  // A polygon P[i] (thickness=3)
	// Arrows
	drawArrow(100,470,200,450,Blue8);						            // An arrow from (100,450) to (200,450)
	drawArrow(300,470,200,450,Red8,30,10,1);				        // tip=30x10 pixels, style=1
	drawArrow(200,450,250,400,Black8,20,20,2);				      // tip=20x20 pixels, style=2
	drawArrow(200,450,150,400,Green8,35.,8.,0,2);			      // tip: (angle,length)=(35,8) , style=0, width=2
	if (stepByStep) click();
	closeWindow(W);
}

void floatingPointDrawing()
{
  // Open 300x200 window
  Window W = openWindow(300,200);
  setAntialiasing(activeWindow());
  setTransparency(activeWindow());

  drawPoint(Point2f(10.5f, 10.5f), Green8);
  drawPoint(Point2f(20.8f, 52.8132f), Green8);

  drawLine(Point2f(10.5f, 10.5f), Point2f(20.8f, 52.8132f), Blue8, 2);
  drawLine(Point2f(10.5f, 20.5f), Point2f(20.8f, 52.8132f), Magenta8, 5);

  // Draw an oriented ellipse whose
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

  if (stepByStep)
    click();
  // Close window...
  closeWindow(W);
}

void bitmapBasics()
{
	cout << "Basic bitmap drawings... click when done" << endl;
	Window W=openWindow(512,512,"Bitmaps");
	Color3ub cols[256*256];										            // byte arrays
	for (int j=0;j<256;j++)
		for (int i=0;i<256;i++)
			cols[i+256*j]= Color3ub(i, 255-i, (j<128)?255:0);	// Some (RED,GREEN,BLUE) functions of (i,j)
	putColorImage(0,0,cols,256,256);							        // Draw this 256x256 (r,g,b) bitmap in (0,0)

	Color3ub cols2[256*256];									            // A Color3ub array
	for (int j=0;j<256;j++) 
		for (int i=0;i<256;i++) 
			cols2[i+256*j]=Color3ub(i,(2*j)%256,(i+j)%256);		// Colors, functions of (i,j)
	putColorImage(Point2i(0,256),cols2,256,256);				  // Draw this 256x256 color bitmap in (0,256)

	uchar grey[256*256];											            // A grey array
	for (int j=0;j<256;j++) 
		for (int i=0;i<256;i++) 
			grey[i+256*j]=uchar(128+127*sin((i+j)/10.));			// Some pattern
	putGreyImage(256,0,grey,256,256);							        // Draw at point (256,0);

	if (stepByStep) click();
	closeWindow(W);
}

void mouseBasics()
{
	cout << "Basic mouse functions" << endl;
	Window W=openWindow(512,512,"Mouse");
	drawString(10,10,"Please click anywhere",Black8);
	if (stepByStep) click();	
	drawString(10,40,"click again (left=BLUE, middle=RED, right=done)",Black8);
	int button;		
	Point2i p;
	while ((button=getMouse(p))!=3)							// Get clicked point p, and used button (1,2,3)=(left,middle,right)
		fillCircle(p,5,(button==1)?Blue8:Red8);
	closeWindow(W);
}

void imageBasics()
{
	cout << "Basic image reading/writing. click when done" << endl; 
	int w, h;												                    // Dimensions
	Color3ub* col;											                // RGB bitmaps
	loadColorImage(srcPath("ksmall.jpg"),col,w,h);			// Read image (and allocate)
	Window W = openWindow(w,2*h,"Images");
	putColorImage(0, 0, col, w, h);							        // Draw it

	cout << sizeof(Color3ub) << endl;
	saveColorImage("outcol.png", col, w, h);		// Write image
	saveColorImage("outcol.tif", col, w, h);		// Write image
	saveColorImage("outcol.jpg", col, w, h);		// Write image
	delete[] col;											                  // ...

	Color3ub *I;											                  // Color3ub bitmap
	loadColorImage("outcol.png",I,w,h);			  // Read again (in one Color3ub array)
	putColorImage(0, h, I, w, h, 2.0);						      // Draw it under previous one, scaling 2.0
	delete[] I;

	uchar *grey;											                  // grey bitmap
	loadGreyImage("outcol.png",grey,w,h);			// Read again (and convert into grey if not already)
	putGreyImage(20, 2*h/3, grey, w, h, 0.5);				    // Draw it somewhere, scale 0.5
	saveGreyImage("outgrey.tif", grey, w, h);	// Write grey image
	delete[] grey;

	if (stepByStep) click();
  saveScreen(W, "capturedScreen.png");

	closeWindow(W);
}

void imageAnimation()
{
  Image<Color3ub> I;
  cout << srcPath("ksmall.jpg") << endl;
  if ( !load(I, srcPath("ksmall.jpg")) )
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
  closeWindow(activeWindow());

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
    //microSleep(100); // TODO: sometimes if you don't put this, the program 
                     // freezes in some machine. Investigate.
  } while (e.key != Qt::Key_Escape);

  cout << "Finished!" << endl;

  closeWindow(activeWindow());
}

void advancedEvents()
{
	cout << "Advanced event handling" << endl;
	Window W = openWindow(1024, 768);
	setActiveWindow(W);
	Image<Color3ub> I;
	load(I, srcPath("ksmall.jpg"));

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
	} while (ev.type != KEY_PRESSED || ev.key != Qt::Key_Up);
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

  SimpleTriangleMesh3f mesh;
  string filename = srcPath("dragon.obj");
  if (!MeshReader().readObjFile(mesh, filename))
  {
    cout << "Error reading mesh file:\n" << filename << endl;
    return;
  }
  cout << "Read " << filename << " successfully" << endl;

  displayMesh(mesh);

  bool quit = false;
  while (!quit)
  {
    int c = getKey();
    quit = (c==Qt::Key_Escape || c==Qt::Key_Space);
  }
  closeWindow();
}
#endif

#ifdef TEST_GRAPHICSVIEW
void graphicsViewExample()
{
  Image<Rgb8> I;
  load(I, srcPath("ksmall.jpg"));

  openGraphicsView(I.width(), I.height());

  for (int i = 0; i < 10; ++i)
    ImageItem image = addImage(I);

  while (getKey() != Qt::Key_Escape);
  closeWindow();
}
#endif

END_NAMESPACE_DO

int main()
{
  using namespace DO;
#ifdef TEST_2D
  // Window manag ement examples.
  aWindow();
  twoWindows();
  twoWindows2();
  multipleWindows();
  // Drawing examples.
  twoDimBasics();
  floatingPointDrawing();
  // Image display examples
  bitmapBasics(); 
  imageBasics();
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
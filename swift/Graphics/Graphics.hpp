#pragma once


class GraphicsContext
{
public:
  GraphicsContext();
  ~GraphicsContext();

  auto initQApp() -> void*;
  auto deinitQApp(void* qApp) -> void;

  auto initContext() -> void*;
  auto deinitContext(void* context) -> void;

  auto initWidgetList() -> void*;
  auto deinitWidgetList(void* widgetList) -> void;

  auto registerUserMainFunc(auto (*userMain)(void)->void) -> void;
  auto exec(void* appObj) -> void;

private:
  int argc = 0;
  char** argv = nullptr;

  void* _qApp = nullptr;
  void* _context = nullptr;
  void* _widgetList = nullptr;
  auto (*_userMain)(void) -> void = nullptr;
};


#if 0
// Window management API.
void* createWindow(int w, int h);
void closeWindow(void* window);
void resizeWindow(int width, int height);
int getKey();


struct Color
{
  unsigned char r;
  unsigned char g;
  unsigned char b;
  unsigned char a;
};


// Draw API.
void drawPoint(int x, int y, const struct Color* c);
void drawLine(int x1, int y1, int x2, int y2, const struct Color* c,
              int penWidth);
void drawRect(int x, int y, int w, int h, int r, int g, int b, int penWidth);
void drawCircle(int xc, int yc, int radius, int r, int g, int b, int penWidth);
void drawEllipse(int x, int y, int w, int h, int r, int g, int b, int penWidth);
void drawOrientedEllipse(float cx, float cy, float r1, float r2, float degree,
                         int r, int g, int b, int penWidth);
void drawArrow(int x1, int y1, int x2, int y2, int r, int g, int b,
               int arrowWidth, int arrowHeight, int style, int width);
void drawText(int x, int y, const char* s, int r, int g, int b, int fontSize,
              double alpha, char italic, char bold, char underlined);
void drawImage(const unsigned char* rgbDataPtr, int w, int h, int xoff,
               int yoff, double fact);


// Fill API.
void fillCircle(int x, int y, int radius, int r, int g, int b);
void fillEllipse(int x, int y, int w, int h, int r, int g, int b);
void fillRect(int x, int y, int w, int h, int r, int g, int b);

void clearWindow();

// Image I/O.
void* JpegImageReader_init(const char* name);
void JpegImageReader_deinit(void* reader);
void JpegImageReader_imageSizes(void* reader, int* w, int* h, int* c);
void JpegImageReader_readImageData(void* reader, unsigned char* dataPtr);

// Video I/O.
void* VideoStream_init(const char* name);
void VideoStream_deinit(void* stream);
unsigned char* VideoStream_getFramePtr(void* stream);
int VideoStream_getFrameWidth(void* stream);
int VideoStream_getFrameHeight(void* stream);
int VideoStream_readFrame(void* stream);
#endif

#pragma once

#include <string>


class QApplication;

namespace DO::Sara {

  class GraphicsContext;
  class WidgetList;

}  // namespace DO::Sara

class GraphicsContext
{
public:
  GraphicsContext();
  ~GraphicsContext();

  auto registerUserMainFunc(auto(*userMain)(void)->void) -> void;

  auto exec() -> void;

private:
  int argc = 0;
  char** argv = nullptr;

  QApplication* _qApp = nullptr;
  DO::Sara::GraphicsContext* _context = nullptr;
  DO::Sara::WidgetList* _widgetList = nullptr;
  auto(*_userMain)(void) -> void = nullptr;
};


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
void drawPoint(int x, int y, const Color& c);
void drawLine(int x1, int y1, int x2, int y2, const Color& c, int penWidth);
void drawRect(int x, int y, int w, int h, int r, int g, int b, int penWidth);
void drawCircle(int xc, int yc, int radius, int r, int g, int b, int penWidth);
void drawEllipse(int x, int y, int w, int h, int r, int g, int b, int penWidth);
void drawOrientedEllipse(float cx, float cy, float r1, float r2, float degree,
                         int r, int g, int b, int penWidth);
void drawArrow(int x1, int y1, int x2, int y2, int r, int g, int b,
               int arrowWidth, int arrowHeight, int style, int width);
void drawText(int x, int y, const std::string& s, int r, int g, int b,
              int fontSize, double alpha, char italic, char bold,
              char underlined);
void drawImage(const unsigned char* rgbDataPtr, int w, int h, int xoff,
               int yoff, double fact);


// Fill API.
void fillCircle(int x, int y, int radius, int r, int g, int b);
void fillEllipse(int x, int y, int w, int h, int r, int g, int b);
void fillRect(int x, int y, int w, int h, int r, int g, int b);

void clearWindow();

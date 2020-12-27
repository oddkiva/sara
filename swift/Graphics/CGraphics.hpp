#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Graphics context API.
void* GraphicsContext_initQApp();
void* GraphicsContext_initWidgetList();
void GraphicsContext_deinitWidgetList(void* widgetList);
void GraphicsContext_registerUserMainFunc(void (*user_main)(void));
void GraphicsContext_exec(void* appObj);


// Window management API.
void* createWindow(int w, int h);
void closeWindow(void* window);
int getKey();


// Draw API.
void drawPoint(int x, int y, int r, int g, int b);
void drawLine(int x1, int y1, int x2, int y2, int r, int g, int b,
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
void drawImage(const void* rgbDataPtr, int w, int h, int xoff, int yoff,
               double fact);


// Fill API.
void fillCircle(int x, int y, int radius, int r, int g, int b);
void fillEllipse(int x, int y, int w, int h, int r, int g, int b);
void fillRect(int x, int y, int w, int h, int r, int g, int b);

void clearWindow();


#ifdef __cplusplus
}
#endif

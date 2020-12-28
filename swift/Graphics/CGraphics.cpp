#include "CGraphics.hpp"

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsContext.hpp>
#include <DO/Sara/ImageIO/Details/ImageIOObjects.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <QApplication>


namespace sara = DO::Sara;

int argc = 0;
char** argv = nullptr;


auto GraphicsContext_initQApp() -> void*
{
  static QApplication app{argc, argv};
  return reinterpret_cast<void*>(&app);
}

auto GraphicsContext_initWidgetList() -> void*
{
  auto& ctx = sara::GraphicsContext::instance();

  auto widgetList = new sara::WidgetList;
  ctx.m_widgetList = widgetList;

  return reinterpret_cast<void*>(widgetList);
}

auto GraphicsContext_deinitWidgetList(void* widgetListObj) -> void
{
  auto widgetList = reinterpret_cast<sara::WidgetList*>(widgetListObj);
  delete widgetList;

  auto& ctx = sara::GraphicsContext::instance();
  ctx.m_widgetList = nullptr;
}


auto GraphicsContext_registerUserMainFunc(void (*user_main)(void)) -> void
{
  auto& ctx = sara::GraphicsContext::instance();
  auto user_main_func = [=](int, char**) -> int {
    (*user_main)();
    return 0;
  };
  ctx.registerUserMain(user_main_func);
}

auto GraphicsContext_exec(void* appObj) -> void
{
  sara::GraphicsContext::instance().userThread().start();

  if (appObj == nullptr)
    return;
  auto app = reinterpret_cast<QApplication*>(appObj);
  app->exec();
}


static auto activeWindow() -> QWidget*
{
  return sara::GraphicsContext::instance().activeWindow();
}


auto createWindow(int w, int h) -> void*
{
  auto ctx = &sara::GraphicsContext::instance();
  const auto x = 0;
  const auto y = 0;
  QMetaObject::invokeMethod(ctx, "createWindow", Qt::BlockingQueuedConnection,
                            Q_ARG(int, sara::GraphicsContext::PAINTING_WINDOW),
                            Q_ARG(int, w), Q_ARG(int, h),
                            Q_ARG(const QString&, QString{"Sara"}),
                            Q_ARG(int, x), Q_ARG(int, y));
  return reinterpret_cast<void*>(ctx->activeWindow());
}

auto closeWindow(void* w) -> void
{
  auto ctx = &sara::GraphicsContext::instance();
  QMetaObject::invokeMethod(ctx, "closeWindow", Qt::BlockingQueuedConnection,
                            Q_ARG(QWidget*, reinterpret_cast<QWidget*>(w)));
}

void resizeWindow(int width, int height)
{
  QMetaObject::invokeMethod(activeWindow(), "resizeScreen",
                            Qt::BlockingQueuedConnection,
                            Q_ARG(int, width), Q_ARG(int, height));
}


auto drawPoint(int x, int y, const Color* c) -> void
{
  QMetaObject::invokeMethod(
      activeWindow(), "drawPoint", Qt::QueuedConnection, Q_ARG(int, x),
      Q_ARG(int, y), Q_ARG(const QColor&, QColor(c->r, c->g, c->b, c->a)));
}

void drawLine(int x1, int y1, int x2, int y2, const Color* c, int penWidth)
{
  QMetaObject::invokeMethod(
      activeWindow(), "drawLine", Qt::QueuedConnection, Q_ARG(int, x1),
      Q_ARG(int, y1), Q_ARG(int, x2), Q_ARG(int, y2),
      Q_ARG(const QColor&, QColor(c->r, c->g, c->b, c->a)),
      Q_ARG(int, penWidth));
}

void drawRect(int x, int y, int w, int h, int r, int g, int b, int penWidth)
{
  QMetaObject::invokeMethod(
      activeWindow(), "drawRect", Qt::QueuedConnection, Q_ARG(int, x),
      Q_ARG(int, y), Q_ARG(int, w), Q_ARG(int, h),
      Q_ARG(const QColor&, QColor(r, g, b)), Q_ARG(int, penWidth));
}

void drawCircle(int xc, int yc, int radius, int r, int g, int b, int penWidth)
{
  QMetaObject::invokeMethod(activeWindow(), "drawCircle", Qt::QueuedConnection,
                            Q_ARG(int, xc), Q_ARG(int, yc), Q_ARG(int, radius),
                            Q_ARG(const QColor&, QColor(r, g, b)),
                            Q_ARG(int, penWidth));
}

void drawEllipse(int x, int y, int w, int h, int r, int g, int b, int penWidth)
{
  QMetaObject::invokeMethod(
      activeWindow(), "drawEllipse", Qt::QueuedConnection, Q_ARG(int, x),
      Q_ARG(int, y), Q_ARG(int, w), Q_ARG(int, h),
      Q_ARG(const QColor&, QColor(r, g, b)), Q_ARG(int, penWidth));
}

void drawOrientedEllipse(float cx, float cy, float r1, float r2, float degree,
                         int r, int g, int b, int penWidth)
{
  QMetaObject::invokeMethod(
      activeWindow(), "drawEllipse", Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(cx, cy)), Q_ARG(qreal, qreal(r1)),
      Q_ARG(qreal, qreal(r2)), Q_ARG(qreal, qreal(degree)),
      Q_ARG(const QColor&, QColor(r, g, b)), Q_ARG(int, penWidth));
}

void drawArrow(int x1, int y1, int x2, int y2, int r, int g, int b,
               int arrowWidth, int arrowHeight, int style, int width)
{
  QMetaObject::invokeMethod(
      activeWindow(), "drawArrow", Qt::QueuedConnection, Q_ARG(int, x1),
      Q_ARG(int, y1), Q_ARG(int, x2), Q_ARG(int, y2),
      Q_ARG(const QColor&, QColor(r, g, b)), Q_ARG(int, arrowWidth),
      Q_ARG(int, arrowHeight), Q_ARG(int, style), Q_ARG(int, width));
}

void drawText(int x, int y, const char* s, int r, int g, int b, int fontSize,
              double alpha, char italic, char bold, char underlined)
{
  QMetaObject::invokeMethod(
      activeWindow(), "drawText", Qt::QueuedConnection, Q_ARG(int, x),
      Q_ARG(int, y), Q_ARG(const QString&, QString{s}),
      Q_ARG(const QColor&, QColor(r, g, b)), Q_ARG(int, fontSize),
      Q_ARG(qreal, qreal(alpha)), Q_ARG(bool, italic), Q_ARG(bool, bold),
      Q_ARG(bool, underlined));
}

void drawImage(const unsigned char* rgbDataPtr, int w, int h, int xoff,
               int yoff, double fact)
{
  auto image = QImage{rgbDataPtr, w, h, w * 3, QImage::Format_RGB888};
  QMetaObject::invokeMethod(activeWindow(), "display",
                            Qt::BlockingQueuedConnection,
                            Q_ARG(const QImage&, image), Q_ARG(int, xoff),
                            Q_ARG(int, yoff), Q_ARG(double, fact));
}


void fillCircle(int x, int y, int radius, int r, int g, int b)
{
  QMetaObject::invokeMethod(activeWindow(), "fillCircle", Qt::QueuedConnection,
                            Q_ARG(int, x), Q_ARG(int, y), Q_ARG(int, radius),
                            Q_ARG(const QColor&, QColor(r, g, b)));
}

void fillEllipse(int x, int y, int w, int h, int r, int g, int b)
{
  QMetaObject::invokeMethod(activeWindow(), "fillEllipse", Qt::QueuedConnection,
                            Q_ARG(int, x), Q_ARG(int, y), Q_ARG(int, w),
                            Q_ARG(int, h),
                            Q_ARG(const QColor&, QColor(r, g, b)));
}

void fillRect(int x, int y, int w, int h, int r, int g, int b)
{
  QMetaObject::invokeMethod(activeWindow(), "fillRect", Qt::QueuedConnection,
                            Q_ARG(int, x), Q_ARG(int, y), Q_ARG(int, w),
                            Q_ARG(int, h),
                            Q_ARG(const QColor&, QColor(r, g, b)));
}

void clearWindow()
{
  QMetaObject::invokeMethod(activeWindow(), "clear", Qt::QueuedConnection);
}

int getKey()
{
  return sara::GraphicsContext::instance().userThread().getKey();
}

void setAntialiasing(bool on)
{
  auto ctx = &sara::GraphicsContext::instance();
  QMetaObject::invokeMethod(ctx->activeWindow(), "setAntialiasing",
                            Qt::QueuedConnection, Q_ARG(bool, on));
}


auto JpegImageReader_init(const char* filepath) -> void*
{
  auto reader = new sara::JpegFileReader(filepath);
  return reinterpret_cast<void*>(reader);
}

auto JpegImageReader_deinit(void* reader) -> void
{
  delete reinterpret_cast<sara::JpegFileReader*>(reader);
}

auto JpegImageReader_imageSizes(void* reader, int* w, int* h, int* c) -> void
{
  auto r = reinterpret_cast<sara::JpegFileReader*>(reader);
  std::tie(*w, *h, *c) = r->image_sizes();
}

auto JpegImageReader_readImageData(void* reader, unsigned char* dataPtr) -> void
{
  auto r = reinterpret_cast<sara::JpegFileReader*>(reader);
  r->read(dataPtr);
}


auto VideoStream_init(const char* filepath) -> void*
{
  auto reader = new sara::VideoStream{filepath};
  return reinterpret_cast<void*>(reader);
}

auto VideoStream_deinit(void* stream) -> void
{
  delete reinterpret_cast<sara::VideoStream*>(stream);
}

auto VideoStream_getFramePtr(void* stream) -> unsigned char *
{
  auto vstream = reinterpret_cast<sara::VideoStream *>(stream);
  auto frame = vstream->frame();
  auto framePtr = reinterpret_cast<unsigned char*>(frame.data());
  return framePtr;
}

auto VideoStream_getFrameWidth(void* stream) -> int
{
  auto vstream = reinterpret_cast<sara::VideoStream *>(stream);
  return vstream->width();
}

auto VideoStream_getFrameHeight(void* stream) -> int
{
  auto vstream = reinterpret_cast<sara::VideoStream *>(stream);
  return vstream->height();
}

auto VideoStream_readFrame(void *stream) -> int
{
  auto vstream = reinterpret_cast<sara::VideoStream *>(stream);
  return static_cast<int>(vstream->read());
}

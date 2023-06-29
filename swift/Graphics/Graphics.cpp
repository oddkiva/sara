#include "Graphics.hpp"

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsContext.hpp>
#include <DO/Sara/ImageIO/Details/ImageIOObjects.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <QApplication>


namespace sara = DO::Sara;

auto GraphicsContext::initQApp() -> void*
{
  qDebug() << "Instantiating QApplication...";
  auto app = new QApplication{argc, argv};
  return reinterpret_cast<void*>(app);
}

auto GraphicsContext::deinitQApp(void* app) -> void
{
  qDebug() << "Destroying QApplication...";
  delete reinterpret_cast<QApplication*>(app);
}

auto GraphicsContext::initContext()-> void *
{
  qDebug() << "Instantiating graphics context...";
  auto context = new sara::GraphicsContext{};
  context->makeCurrent();
  return reinterpret_cast<void*>(context);
}

auto GraphicsContext::deinitContext(void* context) -> void
{
  qDebug() << "Destroying graphics context...";
  delete reinterpret_cast<sara::GraphicsContext *>(context);
}

auto GraphicsContext::initWidgetList() -> void*
{
  qDebug() << "Instantiating widget list...";
  auto widgetList = new sara::WidgetList{};

  auto ctx = sara::GraphicsContext::current();
  if (ctx != nullptr)
    ctx->setWidgetList(widgetList);

  return reinterpret_cast<void*>(widgetList);
}

auto GraphicsContext::deinitWidgetList(void* widgetListObj) -> void
{
  qDebug() << "Destroying widget list...";
  auto widgetList = reinterpret_cast<sara::WidgetList*>(widgetListObj);
  delete widgetList;

  auto ctx = sara::GraphicsContext::current();
  if (ctx)
    ctx->setWidgetList(nullptr);
}


auto GraphicsContext_registerUserMainFunc(auto (*user_main)(void) -> void) -> void
{
  auto ctx = sara::GraphicsContext::current();
  if (ctx == nullptr)
    throw std::runtime_error{"Current graphics context is invalid!"};

  auto user_main_func = [=](int, char**) -> int {
    (*user_main)();
    return 0;
  };
  ctx->registerUserMain(user_main_func);
}

auto GraphicsContext_exec(void* appObj) -> void
{
  auto ctx = sara::GraphicsContext::current();
  if (ctx != nullptr)
    ctx->userThread().start();

  if (appObj == nullptr)
    return;
  auto app = reinterpret_cast<QApplication*>(appObj);
  app->exec();
}

#include "Graphics.hpp"

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsContext.hpp>
#include <DO/Sara/ImageIO/Details/ImageIOObjects.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <QApplication>


namespace sara = DO::Sara;

GraphicsContext::GraphicsContext()
{
  qDebug() << "Instantiating QApplication...";
  _qApp = new QApplication{argc, argv};

  qDebug() << "Instantiating graphics context...";
  _context = new sara::GraphicsContext{};
  if (_context == nullptr)
    throw std::runtime_error{"Failed to initialize graphics context!"};
  _context->makeCurrent();

  qDebug() << "Instantiating widget list...";
  _widgetList = new sara::WidgetList{};
  _context->setWidgetList(_widgetList);
}

GraphicsContext::~GraphicsContext()
{
  qDebug() << "Destroying widget list...";
  delete _widgetList;

  if (_context)
    _context->setWidgetList(nullptr);
  qDebug() << "Destroying graphics context...";
  delete _context;

  qDebug() << "Destroying QApplication...";
  delete _qApp;
}

auto GraphicsContext::registerUserMainFunc(auto(*user_main)(void)->void) -> void
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

auto GraphicsContext::exec() -> void
{
  if (_context != nullptr)
    _context->userThread().start();
  _qApp->exec();
}

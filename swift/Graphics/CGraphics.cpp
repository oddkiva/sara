#include "CGraphics.hpp"

#include <DO/Sara/Graphics.hpp>


namespace sara = DO::Sara;

int argc = 0;
char** argv = nullptr;

// void* GraphicsApplication_initialize(int* argc, char** argv)
void* GraphicsApplication_initialize()
{
  static sara::GraphicsApplication app{argc, argv};
  auto obj = reinterpret_cast<void*>(&app);
  return obj;
}

void GraphicsApplication_registerUserMainFunc(void* app_obj,
                                              void (*user_main)(void))
{
  if (app_obj == nullptr)
    return;
  auto app = reinterpret_cast<sara::GraphicsApplication*>(app_obj);

  auto user_main_func = [=](int, char**) -> int {
    (*user_main)();
    return 0;
  };

  app->register_user_main(user_main_func);
}

void GraphicsApplication_exec(void* app_obj)
{
  if (app_obj == nullptr)
    return;
  auto app = reinterpret_cast<sara::GraphicsApplication*>(app_obj);

  app->exec();
}

void* createWindow(int w, int h)
{
  auto window = reinterpret_cast<void*>(sara::create_window(w, h));
  return window;
}

void closeWindow(void *window)
{
  sara::close_window(reinterpret_cast<sara::Window>(window));
}

void millisleep(int ms)
{
  return sara::millisleep(ms);
}

int getKey()
{
  return sara::get_key();
}

void drawPoint(int x, int y, int r, int g, int b)
{
  sara::draw_point(x, y, sara::Color3ub(r, g, b));
}

#include "CGraphics.hpp"

#include <DO/Sara/Graphics.hpp>


namespace sara = DO::Sara;

int argc = 0;
char** argv = nullptr;

// void* initialize_graphics_application(int* argc, char** argv)
void* initialize_graphics_application()
{
  auto obj = reinterpret_cast<void*>(            //
      new sara::GraphicsApplication{argc, argv}  //
  );
  return obj;
}

void deinitialize_graphics_application(void* obj)
{
  if (obj == nullptr)
    return;

  auto app = reinterpret_cast<sara::GraphicsApplication*>(obj);
  delete app;
  app = nullptr;
}

void register_user_main(void* app_obj, void (*user_main)(void))
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

void exec_graphics_application(void* app_obj)
{
  if (app_obj == nullptr)
    return;
  auto app = reinterpret_cast<sara::GraphicsApplication*>(app_obj);

  app->exec();
}

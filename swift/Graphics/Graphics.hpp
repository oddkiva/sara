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

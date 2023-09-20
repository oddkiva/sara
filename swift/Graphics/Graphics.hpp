#pragma once

class QApplication;

namespace DO::Sara {
class GraphicsContext;
class WidgetList;
}

class GraphicsContext
{
public:
  GraphicsContext();
  ~GraphicsContext();

  auto registerUserMainFunc(auto (*userMain)(void)->void) -> void;

  auto exec() -> void;

private:
  int argc = 0;
  char** argv = nullptr;

  QApplication* _qApp = nullptr;
  DO::Sara::GraphicsContext* _context = nullptr;
  DO::Sara::WidgetList* _widgetList = nullptr;
  auto (*_userMain)(void) -> void = nullptr;
};

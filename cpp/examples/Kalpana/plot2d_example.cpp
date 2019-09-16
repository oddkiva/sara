#include <QApplication>

#include <DO/Kalpana.hpp>


int main(int argc, char **argv)
{
  using namespace DO::Kalpana;

  QApplication app{ argc, argv };

  auto X = np::linspace(-6.28, 6.28, 50);

  auto ax = new Canvas{};
  ax->resize(320, 240);
  ax->plot(X, np::sin(X), plt::style(Qt::red, 5., "--"));
  ax->plot(X, np::cos(X), plt::style(Qt::blue, 5., "-"));
  ax->plot(X, (-X.array().square()).exp(), plt::style(Qt::green, 5., "-"));

  ax->show();
  return app.exec();
}

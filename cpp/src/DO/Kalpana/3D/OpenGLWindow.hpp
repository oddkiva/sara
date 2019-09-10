#pragma once

#include <QtGui/QWindow>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLExtraFunctions>


class QOpenGLPaintDevice;


class OpenGLWindow : public QWindow,
                     protected QOpenGLExtraFunctions
{
  Q_OBJECT

public:
  explicit OpenGLWindow(QWindow* parent = nullptr);
  ~OpenGLWindow();

  virtual void render(QPainter* painter);
  virtual void render();

  virtual void initialize();

  void setAnimating(bool animating);

public slots:
  void renderLater();
  void renderNow();

protected:
  bool event(QEvent* event) override;
  void exposeEvent(QExposeEvent* event) override;

protected:
  bool m_animating;

  QOpenGLContext* m_context;
  QOpenGLPaintDevice* m_device;
};

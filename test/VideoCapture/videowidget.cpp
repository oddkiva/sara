#include "videowidget.hpp"
#include "videosurface.hpp"
#include <QtMultimedia>

VideoWidget::VideoWidget(QWidget *parent)
  : QWidget(parent)
  , surface(0)
{
  setAutoFillBackground(false);
  setAttribute(Qt::WA_NoSystemBackground, true);
  setAttribute(Qt::WA_PaintOnScreen, true);

  QPalette palette = this->palette();
  palette.setColor(QPalette::Background, Qt::black);
  setPalette(palette);

  setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

  surface = new VideoWidgetSurface(this);
}

VideoWidget::~VideoWidget()
{
  delete surface;
}

QSize VideoWidget::sizeHint() const
{
  return surface->surfaceFormat().sizeHint();
}

void VideoWidget::paintEvent(QPaintEvent *event)
{
  QPainter painter(this);

  if (surface->isActive()) {
    const QRect videoRect = surface->videoRect();

    if (!videoRect.contains(event->rect())) {
      QRegion region = event->region();
      region = region.subtracted(videoRect);

      QBrush brush = palette().background();

      foreach (const QRect &rect, region.rects())
        painter.fillRect(rect, brush);
    }

    surface->paint(&painter);
  } else {
    painter.fillRect(event->rect(), palette().background());
  }
}

void VideoWidget::resizeEvent(QResizeEvent *event)
{
  QWidget::resizeEvent(event);

  surface->updateVideoRect();
}
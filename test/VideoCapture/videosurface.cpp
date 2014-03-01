#include "videosurface.hpp"

#include <QtMultimedia>
#include <DO/ImageProcessing.hpp>
#include <queue>

VideoWidgetSurface::
VideoWidgetSurface(QWidget *widget, QObject *parent)
  : QAbstractVideoSurface(parent)
  , widget(widget)
  , imageFormat(QImage::Format_Invalid)
{
}

QList<QVideoFrame::PixelFormat>
VideoWidgetSurface::
supportedPixelFormats(QAbstractVideoBuffer::HandleType handleType) const
{
  if (handleType == QAbstractVideoBuffer::NoHandle) {
    return QList<QVideoFrame::PixelFormat>()
      << QVideoFrame::Format_RGB24
      << QVideoFrame::Format_RGB32
      << QVideoFrame::Format_ARGB32
      << QVideoFrame::Format_ARGB32_Premultiplied
      << QVideoFrame::Format_RGB565
      << QVideoFrame::Format_RGB555;
  } else {
    return QList<QVideoFrame::PixelFormat>();
  }
}

bool
VideoWidgetSurface::
isFormatSupported(const QVideoSurfaceFormat &format,
                  QVideoSurfaceFormat *similar) const
{
  Q_UNUSED(similar);

  const QImage::Format imageFormat = QVideoFrame::imageFormatFromPixelFormat(format.pixelFormat());
  const QSize size = format.frameSize();

  return imageFormat != QImage::Format_Invalid
    && !size.isEmpty()
    && format.handleType() == QAbstractVideoBuffer::NoHandle;
}

bool VideoWidgetSurface::start(const QVideoSurfaceFormat &format)
{
  const QImage::Format imageFormat = QVideoFrame::imageFormatFromPixelFormat(format.pixelFormat());
  const QSize size = format.frameSize();

  if (imageFormat != QImage::Format_Invalid && !size.isEmpty()) {
    this->imageFormat = imageFormat;
    imageSize = size;
    sourceRect = format.viewport();

    QAbstractVideoSurface::start(format);

    widget->updateGeometry();
    updateVideoRect();

    return true;
  } else {
    return false;
  }
}

void VideoWidgetSurface::stop()
{
  currentFrame = QVideoFrame();
  targetRect = QRect();

  QAbstractVideoSurface::stop();

  widget->update();
}

bool VideoWidgetSurface::present(const QVideoFrame &frame)
{
  if (surfaceFormat().pixelFormat() != frame.pixelFormat() ||
      surfaceFormat().frameSize() != frame.size()) {
    setError(IncorrectFormatError);
    stop();
    return false;
  } else {
    currentFrame = frame;
    widget->update(targetRect);
    return true;
  }
}

void VideoWidgetSurface::updateVideoRect()
{
  QSize size = surfaceFormat().sizeHint();
  size.scale(widget->size().boundedTo(size), Qt::KeepAspectRatio);

  targetRect = QRect(QPoint(0, 0), size);
  targetRect.moveCenter(widget->rect().center());
}

using namespace DO;
typedef Image<float> WorkBuffer;
WorkBuffer workBuffer;
Image<Rgb8> dst;
HighResTimer timer;
double elapsed;

void inPlaceColorRescale(WorkBuffer& work)
{
  // Find min, max pixel intensity values.
  WorkBuffer::value_type minPixVal, maxPixVal;
  findMinMax(minPixVal, maxPixVal, work);
  WorkBuffer::value_type rangeVal(maxPixVal - minPixVal);
  // Avoid numeric accuracy issue.
  rangeVal += 1e-6f;
  // Rescale intensity in range [0, 1];
  WorkBuffer::iterator work_pix = work.begin();
  for ( ; work_pix != work.end(); ++work_pix)
    *work_pix = (*work_pix-minPixVal) / rangeVal;
}

void VideoWidgetSurface::paint(QPainter *painter)
{
  if (!currentFrame.map(QAbstractVideoBuffer::ReadOnly))
    return;

  // Get transform.
  const QTransform oldTransform = painter->transform();
  if (surfaceFormat().scanLineDirection() == QVideoSurfaceFormat::BottomToTop)
  {
    painter->scale(1, -1);
    painter->translate(0, -widget->height());
  }

  // Time image processing.
  timer.restart();
  // Wrap image buffer.
  Image<Rgb8> wrapped_src(
    reinterpret_cast<Rgb8 *>(currentFrame.bits()),
    Vector2i(currentFrame.width(), currentFrame.height()) );
  // Sanity check.
  if ( workBuffer.width() != currentFrame.width()   ||
       workBuffer.height() != currentFrame.height() )
  {
    workBuffer.resize(currentFrame.width(), currentFrame.height());
    dst.resize(currentFrame.width(), currentFrame.height());
  }

  // Color-convert and copy buffer.
  WorkBuffer::iterator work_pix = workBuffer.begin();
  Image<Rgb8>::const_iterator const_src_pix = wrapped_src.begin();
  for ( ; work_pix != workBuffer.end(); ++work_pix, ++const_src_pix)
    convertColor(*work_pix, *const_src_pix);

  // Apply image processing.
  inPlaceDeriche(workBuffer, 5.f, 0, 0);
  inPlaceDeriche(workBuffer, 5.f, 0, 1);
  //inPlaceColorRescale(work);
  
  // Copy back to destination
  WorkBuffer::const_iterator const_work_pix = workBuffer.begin();
  Image<Rgb8>::iterator dst_pix = dst.begin();
  for ( ; dst_pix != dst.end(); ++dst_pix, ++const_work_pix)
    convertColor(*dst_pix, *const_work_pix);
  elapsed = timer.elapsedMs();
  qDebug() << "Image frame processing time = " << elapsed << " ms";

  // Blit image to the screen.
  timer.restart();
  QImage image(
    reinterpret_cast<uchar *>(dst.data()),
    dst.width(), dst.height(), dst.width()*3,
    imageFormat);
  painter->drawImage(targetRect, image, sourceRect);
  painter->setTransform(oldTransform);
  elapsed = timer.elapsedMs();
  qDebug() << "Image display time = " << elapsed << " ms";

  currentFrame.unmap();
}
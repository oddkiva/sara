#include "videowidget.hpp"

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);

  // Camera device.
  QCamera camera;

  // Widget which outputs the processed video.
  VideoWidget videoWidget;
  QSize size(1024, 768);
  videoWidget.resize(size);

  // Video surface which processes the video frame.
  QAbstractVideoSurface *surface = videoWidget.videoSurface();
  QVideoSurfaceFormat videosurfaceformat(size, QVideoFrame::Format_RGB24);
  camera.setViewfinder(surface);
  surface->start(videosurfaceformat);
  videoWidget.show();

  // Have a look in particular to "videosurface.cpp" in particular to member function
  // void VideoWidgetSurface::paint(QPainter *painter)

  // Fire the camera!
  camera.start();

  return app.exec();
};
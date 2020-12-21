from PySide2.QtCore import QTimer, Qt, qWarning, Slot
from PySide2.QtWidgets import QApplication, QScrollArea, QWidget
from PySide2.QtGui import QColor, QPainter, QPen, QPixmap


class ScrollArea(QScrollArea):

    def __init__(self, parent=None):
        super(ScrollArea, self).__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_DeleteOnClose)

    def closeEvent(self, event):
        if event.spontaneous():
            qWarning((
                "\n\nWarning: you closed a window unexpectedly!\n\n"
                "Warning: the graphical application cannot terminate "
                "by design:"
                "Warning: please kill the application again manually..."
            ))


class PaintingWindow(QWidget):

    def __init__(self, sizes, window_title="Sara", position=None, parent=None):
        w, h = sizes

        super(PaintingWindow, self).__init__()
        self._scroll_area = ScrollArea(parent=parent)
        self._pixmap = QPixmap(w, h)
        self._painter = QPainter(self._pixmap)
        self._painter.end()

        self.setParent(self._scroll_area)
        self.setFocusPolicy(Qt.WheelFocus)

        # Set event listener.
        self._event_listening_timer = QTimer(self)
        self._event_listening_timer.setSingleShot(True)
        self._event_listening_timer.timeout.connect(
            PaintingWindow.eventListeningTimerStopped
        )

        # Move widget.
        if position is not None:
            self._scroll_area.move(position[0], position[1])
        self._scroll_area.setWindowTitle(window_title)
        self._scroll_area.setWidget(self)
        self._scroll_area.setFocusProxy(self)

        # Maximize if necessary.
        if w >= QApplication.instance().desktop().width() or \
                h >= QApplication.instance().desktop().height():
            self._scroll_area.showMaximized()
        # Resize the scroll area with the size plus a two-pixel offset.
        else:
            self._scroll_area.resize(w + 2, h + 2)
        self.resize(w, h)

        # Initialize the pixmap.
        self._pixmap.fill()
        self.update()

        # Show the widget.
        self._scroll_area.show()

    @property
    def window_title(self):
        return self._scroll_area.windowTitle()

    @property
    def x(self):
        return self._scroll_area.pos().x();

    @property
    def y(self):
        return self._scroll_area.pos().y();

    def paintEvent(self, event):
        p = QPainter(self);
        p.drawPixmap(0, 0, self._pixmap);

    def draw_point(self, x, y, color):
        self._painter.begin(self._pixmap)
        self._painter.setPen(QColor(*color));
        self._painter.drawPoint(x, y);
        self._painter.end()
        self.update();

    def draw_line(self, x1, y1, x2, y2, color, pen_width):
        self._painter.setPen(QPen(QColor(*color), pen_width));
        self._painter.drawLine(x1, y1, x2, y2);
        self.update();

    @Slot()
    def eventListeningTimerStopped(self):
        pass


#  void PaintingWindow::drawPoint(const QPointF& p, const QColor& c)
#  {
#    m_painter.setPen(c);
#    m_painter.drawPoint(p);
#    update();
#  }
#
#  void PaintingWindow::drawLine(const QPointF& p1, const QPointF& p2,
#                                const QColor& c, int penWidth)
#  {
#    m_painter.setPen(QPen(c, penWidth));
#    m_painter.drawLine(p1, p2);
#    update();
#  }
#
#  void PaintingWindow::drawCircle(int xc, int yc, int r, const QColor& c,
#                                  int penWidth)
#  {
#    m_painter.setPen(QPen(c, penWidth));
#    m_painter.drawEllipse(QPoint(xc, yc), r, r);
#    update();
#  }
#
#  void PaintingWindow::drawCircle(const QPointF& center, qreal r,
#                                  const QColor& c, int penWidth)
#  {
#    m_painter.setPen(QPen(c, penWidth));
#    m_painter.drawEllipse(QPointF(center.x(), center.y()), r, r);
#    update();
#  }
#
#  void PaintingWindow::drawEllipse(int x, int y, int w, int h,
#                                   const QColor& c, int penWidth)
#  {
#    m_painter.setPen(QPen(c, penWidth));
#    m_painter.drawEllipse(x, y, w, h);
#    update();
#  }
#
#  void PaintingWindow::drawEllipse(const QPointF& center, qreal r1, qreal r2,
#                                   qreal degree, const QColor& c, int penWidth)
#  {
#    m_painter.save();
#    m_painter.setPen(QPen(c, penWidth));
#    m_painter.translate(center);
#    m_painter.rotate(degree);
#    m_painter.translate(-r1, -r2);
#    m_painter.drawEllipse(QRectF(0, 0, 2*r1, 2*r2));
#    m_painter.restore();
#    update();
#  }
#
#  void PaintingWindow::drawPoly(const QPolygonF& polygon, const QColor& c,
#                                int width)
#  {
#    m_painter.setPen(QPen(c, width));
#    m_painter.drawPolygon(polygon);
#    update();
#  }
#
#  void PaintingWindow::drawRect(int x, int y, int w, int h, const QColor& c,
#                                int penWidth)
#  {
#    m_painter.setPen(QPen(c, penWidth));
#    m_painter.drawRect(x, y, w, h);
#    update();
#  }
#
#  void PaintingWindow::drawText(int x, int y, const QString& text,
#                                const QColor& color, int fontSize,
#                                double orientation, bool italic, bool bold,
#                                bool underline)
#  {
#    QFont font;
#    font.setPointSize(fontSize);
#    font.setItalic(italic);
#    font.setBold(bold);
#    font.setUnderline(underline);
#
#    m_painter.save();
#    m_painter.setPen(color);
#    m_painter.setFont(font);
#
#    m_painter.translate(x, y);
#    m_painter.rotate(qreal(orientation));
#    m_painter.drawText(0, 0, text);
#    m_painter.restore();
#    update();
#  }
#
#  void PaintingWindow::drawArrow(int x1, int y1, int x2, int y2,
#                                 const QColor& col,
#                                 int arrowWidth, int arrowHeight, int style,
#                                 int width)
#  {
#    double sl;
#    double dx = x2-x1;
#    double dy = y2-y1;
#    double norm= qSqrt(dx*dx+dy*dy);
#    if (norm < 0.999) // null vector
#    {
#      m_painter.setPen(QPen(col, width));
#      m_painter.drawPoint(x1, y1);
#      update();
#      return;
#    }
#
#    QPainterPath path;
#    QPolygonF pts;
#
#    qreal dx_norm = dx / norm;
#    qreal dy_norm = dy / norm;
#    qreal p1x = x1 + dx_norm*(norm-arrowWidth) + arrowHeight/2.*dy_norm;
#    qreal p1y = y1 + dy_norm*(norm-arrowWidth) - arrowHeight/2.*dx_norm;
#    qreal p2x = x1 + dx_norm*(norm-arrowWidth) - arrowHeight/2.*dy_norm;
#    qreal p2y = y1 + dy_norm*(norm-arrowWidth) + arrowHeight/2.*dx_norm;
#    switch(style) {
#      case 0:
#        m_painter.setPen(QPen(col, width));
#        m_painter.drawLine(x1, y1, x2, y2);
#        m_painter.drawLine(x2, y2, int(p1x), int(p1y));
#        m_painter.drawLine(x2, y2, int(p2x), int(p2y));
#        break;
#      case 1:
#        pts << QPointF(p2x, p2y);
#        pts << QPointF(x2, y2);
#        pts << QPointF(p1x, p1y);
#        sl = norm-(arrowWidth*.7);
#        pts << QPointF(x1 + dx_norm*sl + dy_norm*width,
#                       y1 + dy_norm*sl - dx_norm*width);
#        pts << QPointF(x1 + dy_norm*width, y1 - dx_norm*width);
#        pts << QPointF(x1 - dy_norm*width, y1 + dx_norm*width);
#        pts << QPointF(x1 + dx_norm*sl - dy_norm*width,
#                       y1 + dy_norm*sl + dx_norm*width);
#        path.addPolygon(pts);
#        m_painter.fillPath(path, col);
#        break;
#      case 2:
#        pts << QPointF(p2x, p2y);
#        pts << QPointF(x2, y2);
#        pts << QPointF(p1x, p1y);
#        sl = norm-arrowWidth;
#        pts << QPointF(x1 + dx_norm*sl + dy_norm*width,
#                       y1 + dy_norm*sl - dx_norm*width);
#        pts << QPointF(x1 + dy_norm*width, y1-dx_norm*width);
#        pts << QPointF(x1 - dy_norm*width, y1+dx_norm*width);
#        pts << QPointF(x1 + dx_norm*sl - dy_norm*width,
#                       y1 + dy_norm*sl + dx_norm*width);
#        path.addPolygon(pts);
#        m_painter.fillPath(path, col);
#        break;
#      default:
#        break;
#    }
#
#    update();
#  }
#
#  void PaintingWindow::display(const QImage& image, int xoff, int yoff,
#                               double fact)
#  {
#    m_painter.translate(xoff, yoff);
#    m_painter.scale(qreal(fact), qreal(fact));
#    m_painter.drawImage(0, 0, image);
#    m_painter.scale(qreal(1./fact), qreal(1./fact));
#    m_painter.translate(-xoff, -yoff);
#    update();
#  }
#
#  void PaintingWindow::fillCircle(int x, int y, int r, const QColor& c)
#  {
#    QPainterPath path;
#    path.addEllipse(qreal(x)-r/2., qreal(y)-r/2., qreal(r), qreal(r));
#    m_painter.fillPath(path, c);
#    update();
#  }
#
#  void PaintingWindow::fillCircle(const QPointF& p, qreal r, const QColor& c)
#  {
#    QPainterPath path;
#    path.addEllipse(p, r, r);
#    m_painter.fillPath(path, c);
#    update();
#  }
#
#  void PaintingWindow::fillEllipse(int x, int y, int w, int h,
#                                   const QColor& c)
#  {
#    QPainterPath path;
#    path.addEllipse(qreal(x), qreal(y), qreal(w), qreal(h));
#    m_painter.fillPath(path, c);
#    update();
#  }
#
#  void PaintingWindow::fillEllipse(const QPointF& p, qreal rx, qreal ry,
#                                   qreal degree, const QColor& c)
#  {
#    m_painter.save();
#    m_painter.translate(p);
#    m_painter.rotate(degree);
#    m_painter.translate(-rx, -ry);
#    QPainterPath path;
#    path.addEllipse(0., 0., 2*rx, 2*ry);
#    m_painter.fillPath(path, c);
#    m_painter.restore();
#    update();
#  }
#
#  void PaintingWindow::fillPoly(const QPolygonF& polygon, const QColor& c)
#  {
#    QPainterPath path;
#    path.addPolygon(polygon);
#    m_painter.fillPath(path, c);
#    update();
#  }
#
#  void PaintingWindow::fillRect(int x, int y, int w, int h,
#                                const QColor& c)
#  {
#    m_painter.setPen(c);
#    m_painter.fillRect(x, y, w, h, c);
#    update();
#  }
#
#  void PaintingWindow::clear()
#  {
#    m_pixmap.fill();
#    update();
#  }
#
#  void PaintingWindow::setAntialiasing(bool on)
#  {
#    m_painter.setRenderHints(QPainter::Antialiasing, on);
#  }
#
#  void PaintingWindow::setTransparency(bool on)
#  {
#    if (on)
#      m_painter.setCompositionMode(QPainter::CompositionMode_Multiply);
#    else
#      m_painter.setCompositionMode(QPainter::CompositionMode_Source);
#  }
#
#  void PaintingWindow::saveScreen(const QString& filename)
#  {
#    m_pixmap.save(filename);
#  }
#
#  void PaintingWindow::resizeScreen(int width, int height)
#  {
#    if (m_pixmap.width() == width && m_pixmap.height() == height)
#      return;
#    /*
#       The following internal changes are critical to prevent Qt from crashing.
#       1. Tell QPainter 'painter_' to stop using using QPixmap 'pixmap_'.
#       2. Reinitialize the QPixmap with the new size.
#       3. Now we can re-allow QPainter 'painter_' to re-use QPixmap 'pixmap_'.
#     */
#    m_painter.end();
#    m_pixmap = QPixmap(width, height);
#    m_pixmap.fill();
#    m_painter.begin(&m_pixmap);
#
#    // Resize the window and the scroll area as follows.
#    resize(width, height);
#    if (width > qApp->desktop()->width() || height > qApp->desktop()->height())
#    {
#      width = 800;
#      height = 600;
#    }
#    self._scroll_area->resize(width+2, height+2);
#  }
#
#  void PaintingWindow::waitForEvent(int ms)
#  {
#    m_eventListeningTimer.setInterval(ms);
#    m_eventListeningTimer.start();
#  }
#
#  void PaintingWindow::eventListeningTimerStopped()
#  {
#    emit sendEvent(no_event());
#  }
#
#  void PaintingWindow::mouseMoveEvent(QMouseEvent *event)
#  {
#    emit movedMouse(event->x(), event->y(), event->buttons());
#
#    if (m_eventListeningTimer.isActive())
#    {
#      m_eventListeningTimer.stop();
#      emit sendEvent(mouse_moved(event->x(), event->y(), event->buttons(),
#                     event->modifiers()));
#    }
#  }
#
#  void PaintingWindow::mousePressEvent(QMouseEvent *event)
#  {
##ifdef Q_OS_MAC
#    Qt::MouseButtons buttons = (event->modifiers() == Qt::ControlModifier &&
#                  event->buttons() == Qt::LeftButton) ?
#    Qt::MiddleButton : event->buttons();
#    emit pressedMouseButtons(event->x(), event->y(), buttons);
##else
#    emit pressedMouseButtons(event->x(), event->y(), event->buttons());
##endif
#    if (m_eventListeningTimer.isActive())
#    {
#      m_eventListeningTimer.stop();
#      emit sendEvent(mouse_pressed(event->x(), event->y(), event->buttons(),
#                     event->modifiers()));
#    }
#  }
#
#  void PaintingWindow::mouseReleaseEvent(QMouseEvent *event)
#  {
##ifdef Q_OS_MAC
#    Qt::MouseButtons buttons = (event->modifiers() == Qt::ControlModifier &&
#                                event->buttons() == Qt::LeftButton) ?
#      Qt::MiddleButton : event->buttons();
#    emit releasedMouseButtons(event->x(), event->y(), buttons);
##else
#    emit releasedMouseButtons(event->x(), event->y(), event->button());
##endif
#    if (m_eventListeningTimer.isActive())
#    {
#      m_eventListeningTimer.stop();
#      emit sendEvent(mouse_released(event->x(), event->y(),
#                                   event->buttons(), event->modifiers()));
#    }
#  }
#
#  void PaintingWindow::keyPressEvent(QKeyEvent *event)
#  {
#    emit pressedKey(event->key());
#    if (m_eventListeningTimer.isActive())
#    {
#      m_eventListeningTimer.stop();
#      emit sendEvent(key_pressed(event->key(), event->modifiers()));
#    }
#  }
#
#  void PaintingWindow::keyReleaseEvent(QKeyEvent *event)
#  {
#    emit releasedKey(event->key());
#    if (m_eventListeningTimer.isActive())
#    {
#      m_eventListeningTimer.stop();
#      emit sendEvent(key_released(event->key(), event->modifiers()));
#    }
#  }

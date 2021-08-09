from PySide2.QtCore import QPointF
from PySide2.QtGui import QColor, QFont, QImage, QPainter, QPen


def to_qimage(array):
    h, w, c = array.shape
    qimage = QImage(array.data, w, h, c * w, QImage.Format_RGB888)
    return qimage


def draw_line(array, p1, p2, color, pen_width, antialiasing=True):
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.Antialiasing, _antialiasing)
    p.setPen(QPen(QColor(*color), pen_width))
    p.drawLine(QPointF(*p1), QPointF(*p2))
    p.end()

def draw_point(array, x, y, color, antialiasing=True):
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setPen(QColor(*color));
    p.drawPoint(x, y);
    p.end()

def draw_line(array, p1, p2, color, pen_width, antialiasing=True):
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.Antialiasing, antialiasing)
    p.setPen(QPen(QColor(*color), pen_width))
    p.drawLine(QPointF(*p1), QPointF(*p2))
    p.end()

def draw_rect(array, top_left_corner, sizes, color, pen_width,
              antialiasing=True):
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.Antialiasing, antialiasing)
    p.setPen(QPen(QColor(*color), pen_width))
    p.drawRect(top_left_corner[0], top_left_corner[1],
                           sizes[0], sizes[1]);
    p.end()

def draw_circle(array, center, radius, color, pen_width, antialiasing=True):
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.Antialiasing, antialiasing);
    p.setPen(QPen(QColor(*color), pen_width))
    p.drawEllipse(QPointF(*center), radius, radius);
    p.end()

def draw_ellipse(array, center, r1, r2, angle_in_degrees, color, pen_width,
                 antialiasing=True):
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.Antialiasing, antialiasing)
    p.save()
    p.setPen(QPen(QColor(*color), pen_width))
    p.translate(QPointF(*center))
    p.rotate(angle_in_degrees)
    p.translate(-r1, -r2)
    p.drawEllipse(QRectF(0, 0, 2 * r1, 2 * r2))
    p.restore()
    p.end()

def draw_text(array, p, text, color, font_size, orientation, italic, bold,
              underline, antialiasing=True):
    font = QFont()
    font.setPointSize(font_size)
    font.setItalic(italic)
    font.setBold(bold)
    font.setUnderline(underline)

    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.Antialiasing, antialiasing)

    p.save()
    p.setPen(QColor(*color))
    p.setFont(font)

    p.translate(p[0], p[1])
    p.rotate(orientation)
    p.drawText(0, 0, text)
    p.restore()
    p.end()

def draw_image(array, image, offset, scale):
    surface = to_qimage(array)
    p = QPainter(surface)

    xoff, yoff = offset
    p.translate(xoff, yoff)
    p.scale(scale, scale)
    p.drawImage(0, 0, image)
    p.scale(1 / scale, 1 / scale)
    p.translate(-xoff, -yoff)
    p.end()

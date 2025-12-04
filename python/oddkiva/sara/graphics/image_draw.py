# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import numpy as np

from PySide6.QtCore import QPoint, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen


def to_qimage(array):
    h, w, c = array.shape
    qimage = QImage(array.data, w, h, c * w, QImage.Format_RGB888)
    return qimage

def draw_point(array, x, y, color, antialiasing=True):
    """ Draw a point.
    """
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.Antialiasing, antialiasing)
    p.setPen(QColor(*color));
    p.drawPoint(x, y);
    p.end()

def draw_line(array, p1, p2, color, pen_width, antialiasing=True):
    """ Draw a line.
    """
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.Antialiasing, antialiasing)
    p.setPen(QPen(QColor(*color), pen_width))
    p.drawLine(QPointF(*p1), QPointF(*p2))
    p.end()

def draw_rect(array, top_left_corner, sizes, color, pen_width,
              antialiasing=True):
    """ Draw a rectangle.
    """
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.Antialiasing, antialiasing)
    p.setPen(QPen(QColor(*color), pen_width))
    p.drawRect(top_left_corner[0], top_left_corner[1],
                           sizes[0], sizes[1]);
    p.end()

def draw_circle(array, center, radius, color, pen_width, antialiasing=True):
    """ Draw a circle.
    """
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.Antialiasing, antialiasing);
    p.setPen(QPen(QColor(*color), pen_width))
    p.drawEllipse(QPointF(*center), radius, radius);
    p.end()

def draw_ellipse(array, center, r1, r2, angle_in_degrees, color, pen_width,
                 antialiasing=True):
    """ Draw an ellipse.
    """
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

def draw_text(array: np.ndarray, p: QPoint, text: str, color: QColor,
              font_size: int, orientation: float, italic: bool, bold: bool,
              underline: bool, antialiasing: bool =True):
    """ Draw a string.
    """
    font = QFont()
    font.setPointSize(font_size)
    font.setItalic(italic)
    font.setBold(bold)
    font.setUnderline(underline)

    surface = to_qimage(array)
    painter = QPainter(surface)
    painter.setRenderHints(QPainter.Antialiasing, antialiasing)

    painter.save()
    painter.setPen(QColor(*color))
    painter.setFont(font)

    painter.translate(p[0], p[1])
    painter.rotate(orientation)
    painter.drawText(0, 0, text)
    painter.restore()
    painter.end()

def draw_image(array: np.ndarray, image: np.ndarray, offset: QPoint,
               scale: float):
    """ Draw an image.
    """
    surface = to_qimage(array)
    p = QPainter(surface)

    xoff, yoff = offset
    p.translate(xoff, yoff)
    p.scale(scale, scale)
    p.drawImage(0, 0, image)
    p.scale(1 / scale, 1 / scale)
    p.translate(-xoff, -yoff)
    p.end()

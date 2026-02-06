# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from typing import Iterable

import numpy as np

from PySide6.QtCore import QPointF, QRectF, QSizeF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPen,
)


def to_qimage(array: np.ndarray):
    h, w, c = array.shape
    qimage = QImage(array.data, w, h, c * w, QImage.Format.Format_RGB888)
    return qimage

def draw_point(array: np.ndarray, x, y, color, antialiasing: bool = True):
    """ Draw a point.
    """
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.RenderHint.Antialiasing, antialiasing)
    p.setPen(QColor(*color));
    p.drawPoint(x, y);
    p.end()

def draw_line(array: np.ndarray, p1, p2, color, pen_width,
              antialiasing: bool = True):
    """ Draw a line.
    """
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.RenderHint.Antialiasing, antialiasing)
    p.setPen(QPen(QColor(*color), pen_width))
    p.drawLine(QPointF(*p1), QPointF(*p2))
    p.end()

def draw_rect(array: np.ndarray, top_left_corner, sizes, color, pen_width,
              antialiasing: bool = True):
    """ Draw a rectangle.
    """
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.RenderHint.Antialiasing, antialiasing)
    p.setPen(QPen(QColor(*color), pen_width))
    p.drawRect(top_left_corner[0], top_left_corner[1],
                           sizes[0], sizes[1]);
    p.end()

def draw_circle(array: np.ndarray, center, radius, color, pen_width,
                antialiasing: bool = True):
    """ Draw a circle.
    """
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.RenderHint.Antialiasing, antialiasing)
    p.setPen(QPen(QColor(*color), pen_width))
    p.drawEllipse(QPointF(*center), radius, radius);
    p.end()

def draw_ellipse(array: np.ndarray, center, r1, r2, angle_in_degrees: float,
                 color: Iterable[int], pen_width: int,
                 antialiasing: bool =True):
    """ Draw an ellipse.
    """
    surface = to_qimage(array)
    p = QPainter(surface)
    p.setRenderHints(QPainter.RenderHint.Antialiasing, antialiasing)
    p.save()
    p.setPen(QPen(QColor(*color), pen_width))
    p.translate(QPointF(*center))
    p.rotate(angle_in_degrees)
    p.translate(-r1, -r2)
    p.drawEllipse(QRectF(0, 0, 2 * r1, 2 * r2))
    p.restore()
    p.end()

def draw_text(array: np.ndarray, p: Iterable[int], text: str,
              color: Iterable[int], font_size: int, orientation: float,
              italic: bool, bold: bool, underline: bool,
              antialiasing: bool = True):
    """ Draw a string.
    """
    font = QFont()
    font.setPointSize(font_size)
    font.setItalic(italic)
    font.setBold(bold)
    font.setUnderline(underline)

    surface = to_qimage(array)
    painter = QPainter(surface)
    painter.setRenderHints(QPainter.RenderHint.Antialiasing, antialiasing)

    painter.save()
    painter.setPen(QColor(*color))
    painter.setFont(font)

    painter.translate(p[0], p[1])
    painter.rotate(orientation)
    painter.drawText(0, 0, text)
    painter.restore()
    painter.end()

def draw_boxed_text(array: np.ndarray,
                    p: Iterable[float],
                    text: str,
                    box_color: Iterable[int | float],
                    font: QFont,
                    text_color: Iterable[int | float] = (0, 0, 0),
                    angle: float = 0.) -> None:
    font_metrics = QFontMetrics(font)
    text_offset = 1
    def calculate_text_box_size(text: str):
        label_text_rect = font_metrics.boundingRect(text)
        size = label_text_rect.size()
        w, h = size.width() + text_offset * 2 + 1, size.height()
        return w, h

    w, h = calculate_text_box_size(text)
    xoff = -5
    yoff = h * 0.5
    l, t = (int(p[0] + xoff + 0.5), int(p[1] + 0.5 + yoff))
    fill_rect(array, (l - text_offset, int(t - h + 0.5)), (w, h), box_color)
    draw_text(array, (l, t - 2 * text_offset), text, text_color,
              font.pointSize(), angle,
              font.italic(), font.bold(), font.underline())

def draw_image(array: np.ndarray, image: np.ndarray,
               offset: Iterable[int | float],
               scale: float):
    """ Draw an image.
    """
    surface = to_qimage(array)
    p = QPainter(surface)

    xoff, yoff = offset
    p.translate(xoff, yoff)
    p.scale(scale, scale)
    p.drawImage(0, 0, to_qimage(image))
    p.scale(1 / scale, 1 / scale)
    p.translate(-xoff, -yoff)
    p.end()


def fill_rect(array: np.ndarray,
              top_left_corner: Iterable[int],
              sizes: Iterable[int],
              color: Iterable[int | float]):
    surface = to_qimage(array)
    p = QPainter(surface)

    p.begin(surface)
    p.fillRect(
        QRectF(QPointF(*top_left_corner), QSizeF(*sizes)),
        QBrush(QColor(*color))
    )
    p.end()

from PySide2.QtGui import QImage
from PySide2.QtWidgets import QApplication

from do.sara.graphics.derived_qobjects.graphics_context import GraphicsContext
import do.sara.graphics.image_draw as image_draw


def millisleep(ms):
    ctx = GraphicsContext()
    ctx.user_thread.msleep(ms)

def get_key():
    user_thread = GraphicsContext().user_thread
    user_thread.get_key()
    return user_thread.key

def create_window(w, h):
    user_thread = GraphicsContext().user_thread
    user_thread.signals.create_window.emit(w, h)

def draw_point(x, y, color):
    user_thread = GraphicsContext().user_thread
    user_thread.signals.draw_point.emit(x, y, color)

def draw_line(p1, p2, color, pen_width=1):
    user_thread = GraphicsContext().user_thread
    user_thread.signals.draw_line.emit(p1, p2, color, pen_width)

def draw_rect(top_left_corner, sizes, color, pen_width=1):
    user_thread = GraphicsContext().user_thread
    user_thread.signals.draw_rect.emit(top_left_corner, sizes, color, pen_width)

def draw_circle(center, radius, color, pen_width=1):
    user_thread = GraphicsContext().user_thread
    user_thread.signals.draw_circle.emit(center, radius, color, pen_width)

def draw_ellipse(center, r1, r2, angle_in_degrees, color, pen_width=1):
    user_thread = GraphicsContext().user_thread
    user_thread.signals.draw_ellipse.emit(center, r1, r2, angle_in_degrees,
                                          color, pen_width)

def draw_text(p, text, color, font_size, orientation, italic, bold, underline):
    user_thread = GraphicsContext().user_thread
    user_thread.signals.draw_text.emit(p, text, color, font_size, orientation,
                                       italic, bold, underline)

def draw_image(image, offset=(0, 0), scale=1):
    user_thread = GraphicsContext().user_thread
    h, w, c = image.shape
    if c != 3:
        raise NotImplementedError()
    qimage = QImage(image.data, w, h, c * w, QImage.Format_RGB888)
    user_thread.signals.draw_image.emit(qimage, offset, scale)

def clear():
    user_thread = GraphicsContext().user_thread
    user_thread.signals.clear.emit()

def set_antialiasing(on = True):
    user_thread = GraphicsContext().user_thread
    user_thread.signals.set_antialiasing.emit(on)

def run_graphics(user_main):
    import sys

    app = QApplication(sys.argv)

    ctx = GraphicsContext()
    ctx.user_thread.register_user_main(user_main)
    ctx.user_thread.start()

    sys.exit(app.exec_())

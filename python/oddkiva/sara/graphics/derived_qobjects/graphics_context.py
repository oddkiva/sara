from PySide2.QtCore import QObject, Qt
from PySide2.QtWidgets import QApplication

from do.sara.graphics.derived_qobjects.painting_window import PaintingWindow
from do.sara.graphics.derived_qobjects.user_thread import UserThread


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class WindowManager(QObject):

    def __init__(self):
        self._widgets = []
        self._active_window = None

    def create_painting_window(self, w, h):
        pw =  PaintingWindow((w, h))
        self._widgets.append(pw)
        if len(self._widgets) == 1:
            self._active_window = pw
            self.connect_widget_to_user_thread(self._active_window)

    def connect_widget_to_user_thread(self, widget):
        if widget is None:
            return

        user_thread = GraphicsContext().user_thread

        # draw_xxx
        user_thread.signals.draw_point.connect(
            self._active_window.draw_point,
            type=Qt.QueuedConnection)
        user_thread.signals.draw_line.connect(
            self._active_window.draw_line,
            type=Qt.QueuedConnection)
        user_thread.signals.draw_rect.connect(
            self._active_window.draw_rect,
            type=Qt.QueuedConnection)
        user_thread.signals.draw_circle.connect(
            self._active_window.draw_circle,
            type=Qt.QueuedConnection)
        user_thread.signals.draw_ellipse.connect(
            self._active_window.draw_ellipse,
            type=Qt.QueuedConnection)
        user_thread.signals.draw_text.connect(
            self._active_window.draw_text,
            type=Qt.QueuedConnection)
        user_thread.signals.draw_image.connect(
            self._active_window.draw_image,
            type=Qt.BlockingQueuedConnection)

        user_thread.signals.clear.connect(
            self._active_window.clear,
            type=Qt.BlockingQueuedConnection)
        user_thread.signals.set_antialiasing.connect(
            self._active_window.set_antialiasing,
            type=Qt.BlockingQueuedConnection)

        widget.signals.pressed_key.connect(user_thread.pressed_key)

    @property
    def widgets(self):
        return self._widgets

    @property
    def active_window(self):
        return self._active_window


class GraphicsContext(metaclass=Singleton):

    def __init__(self):
        self._user_thread = UserThread()
        self._window_manager = WindowManager()

        # Create connections between signals and slots.
        self._user_thread.signals.create_window.connect(
            self._window_manager.create_painting_window,
            type=Qt.BlockingQueuedConnection)

    @property
    def user_thread(self):
        return self._user_thread

    @property
    def window_manager(self):
        return self._window_manager

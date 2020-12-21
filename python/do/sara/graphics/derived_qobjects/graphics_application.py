from PySide2.QtCore import QMetaObject, QObject, Qt, QGenericArgument
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

    def create_painting_window(self, w, h):
        pw =  PaintingWindow((w, h))
        self._widgets.append(pw)
        return pw

    @property
    def widgets(self):
        return self._widgets


class GraphicsContext(metaclass=Singleton):

    def __init__(self):
        self._user_thread = UserThread()
        self._window_manager = WindowManager()

        # Create connections between signals and slots.
        import ipdb; ipdb.set_trace()
        self._user_thread.signals.create_window.connect(
            self._window_manager.create_painting_window)

    @property
    def user_thread(self):
        return self._user_thread

    @property
    def window_manager(self):
        return self._window_manager


def millisleep(ms):
    ctx = GraphicsContext()
    ctx.user_thread.msleep(ms)


def create_window(w, h):
    user_thread = GraphicsContext().user_thread
    user_thread.signals.create_window.emit(w, h)


def user_main():
    create_window(800, 600)

    print('hello world!')
    i = 0
    while i < 100:
        print(i)
        i += 1
        millisleep(20)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    graphics_context = GraphicsContext()
    graphics_context.user_thread.register_user_main(user_main)
    graphics_context.user_thread.start()

    sys.exit(app.exec_())

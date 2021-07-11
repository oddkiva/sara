from PySide2.QtCore import QMutex, QObject, QThread, Qt, QWaitCondition, Signal
from PySide2.QtWidgets import QApplication


class UserThreadSignals(QObject):
    create_window = Signal(int, int)
    draw_point = Signal(int, int, object)
    draw_image = Signal(object, object, float)


class UserThread(QThread):

    def __init__(self, parent=None):
        super(UserThread, self).__init__(parent)
        self._user_main = None
        self._mutex = QMutex()
        self._condition = QWaitCondition()

        self._do_wait_for_click = False
        self._mouse_button = Qt.MouseButtons
        self._mouse_x = None
        self._mouse_y = None

        self._do_wait_for_key = False
        self._key = None

        self._event = None

        self.signals = UserThreadSignals()
        self.finished.connect(QApplication.instance().quit)

    def register_user_main(self, user_main):
        self._user_main = user_main

    def get_mouse(self):
        pass

    def get_key(self):
        self._mutex.lock()
        self._do_wait_for_key = True
        self._condition.wait(self._mutex)
        self._mutex.unlock()
        return self._key

    def pressed_key(self, key):
        self._mutex.lock()
        if self._do_wait_for_key:
            self._do_wait_for_key = False
            self._key = key
            self._condition.wakeOne()
        self._mutex.unlock()

    @property
    def key(self):
        return self._key

    def get_event(self, event):
        pass

    def listenToWindowEvents(self):
        pass

    def pressedMouseButtons(self, x, y, buttons):
        pass

    def closedWindow(self):
        pass

    def receivedEvent(self, event):
        pass

    def sendEvent(self, event, delayMs):
        pass

    def run(self):
        if self._user_main is None:
            return
        self._user_main()

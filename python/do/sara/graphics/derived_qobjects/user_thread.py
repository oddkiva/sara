from PySide2.QtCore import QMutex, QObject, QThread, Qt, QWaitCondition, Signal
from PySide2.QtWidgets import QApplication


class Communicate(QObject):
    create_window = Signal(int, int)


class UserThread(QThread):
    def __init__(self, parent=None):
        super(UserThread, self).__init__(parent)
        self._user_main = None
        self._mutex = QMutex()
        self._condition = QWaitCondition()

        self._doWaitForClick = False
        self._mouseButton = Qt.MouseButtons
        self._mouseX = None
        self._mouseY = None

        self._doWaitForKey = False
        self._key = None

        self._event = None

        self.signals = Communicate()
        self.finished.connect(QApplication.instance().quit)


    def register_user_main(self, user_main):
        self._user_main = user_main

    def get_mouse(self):
        pass

    def get_key(self):
        pass

    def get_event(self, event):
        pass

    def listenToWindowEvents(self):
        pass

    def pressedMouseButtons(self, x, y, buttons):
        pass

    def pressedKey(self, key):
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

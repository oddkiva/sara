from PySide2.QtCore import QThread, QMutex, QWaitCondition


class UserThread(QThread):

    def __init__(parent=None):
        super(UserThread, self).__init__(parent)
        self._user_main = None
        self._mutex = QMutex()
        self._condition = QWaitCondition()

        self._doWaitForClick = False
        self._mouseButton = Qt::MouseButtons
        self._mouseX = None
        self._mouseY = None

        self._doWaitForKey = False
        self._key = None

        self._event = None

    def registerUserMain(self, user_main):
        self._user_main = userMain

    def millisleep(self, msec)
        self.msleep(msec)

    def microsleep(self, usec):
        self.usleep(usec)

    def get_mouse(self):
        pass

    def get_key(self):
        pass

    def get_event(self, event);
        pass

    def listenToWindowEvents(self):
        pass

    def pressedMouseButtons(self, x, y, buttons):
        pass

    def pressedKey(self, key):
        pass

    def closedWindow(self):
        pass

    def receivedEvent(Event e):
        pass

    def sendEvent(self, event, delayMs);

    def run(self):
        pass

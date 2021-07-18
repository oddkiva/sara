import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('[{}] Elapsed: {} ms'.format(self.name,
                                           (time.time() - self.tstart) * 1e3))

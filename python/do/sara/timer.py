import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if not self.name:
            print('Elapsed: {}s'.format(time.time() - self.tstart))
        else:
            print('[{}] Elapsed: {}s'.format(self.name,
                                             time.time() - self.tstart))

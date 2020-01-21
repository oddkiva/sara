import six

from subprocess import Popen

import unittest

import zmq

from do.sara import IpcMedium


class TestIPC(unittest.TestCase):

    def test_ipc_with_cpp(self):
        ipc_medium = IpcMedium("MySharedMemory")

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")

        while True:
            print('[Python] Sending request')
            socket.send(b"1")

            image = ipc_medium.tensor("image")
            print('[Python] before image =\n', image)

            image[:] = -1
            print('[Python] after image =\n', image)

            message = socket.recv()
            print('[Python] Received reply {} '.format(message))


if __name__ == '__main__':
    unittest.main()

import six

from subprocess import Popen

import unittest

import zmq

from do.sara import IpcMedium


def get_image(ipc_medium):
    image_shape = ipc_medium.image_shape("image_shape")
    print('[Python] image_shape = ', ipc_medium.image_shape("image_shape"))

    image_data = ipc_medium.image_data("image_data")
    print('[Python] image_data = ', image_data)

    return image_data

class TestIPC(unittest.TestCase):

    def test_ipc_with_cpp(self):
        ipc_medium = IpcMedium("MySharedMemory")


        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")

        while True:
            print('[Python] Sending request')
            socket.send(b"1")


            image_data = get_image(ipc_medium)

            image_data[:] = -1

            message = socket.recv()
            print('[Python] Received reply {} '.format(message))


if __name__ == '__main__':
    unittest.main()

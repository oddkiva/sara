from oddkiva.sara import resize

import numpy as np


def test_resize():
    src = np.arange(16).reshape((4, 4))
    src = np.array([src, src, src]).astype(np.float32)
    print(src)

    dst = np.zeros((3, 4, 4)).astype(np.float32)
    
    resize(src, dst)
    print(dst)

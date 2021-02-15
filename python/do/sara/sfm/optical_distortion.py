import numpy as np
import matplotlib.pyplot as plt


# Important overlooked details: coordinates should to be normalized so that
# they are in [-1, 1].
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
xv, yv = np.meshgrid(x, y)

px, py = np.vstack((xv.ravel(), yv.ravel()))

# Barrel distortion.
k1, k2 = 0.47173, 0.0895379


def radial_correction(k1, k2, xd, yd, c = np.zeros(2)):
    cx, cy = c
    r_x = xd - cx
    r_y = yd - cy
    r2 = r_x ** 2 + r_y ** 2
    r4 = r2 ** 2
    distortion = k1 * r2 + k2 * r4
    xu = xd + distortion * r_x
    yu = yd + distortion * r_y
    return xu, yu

pxu, pyu = radial_correction(k1, k2, px, py)

plt.scatter(pxu, pyu)
plt.show()

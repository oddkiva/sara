from typing import Tuple
from oddkiva.brahma.torch import DEFAULT_DEVICE

import torch as T
import torch.nn as nn


def enumerate_coords(w: int, h: int) -> T.Tensor:
    x, y = T.meshgrid(T.arange(0, w), T.arange(0, h), indexing='xy')
    x, y = x.reshape((w * h,)), y.reshape((w * h,))
    p = T.stack((x, y))
    return p

def bilinear_interpolation_2d(
    image: T.Tensor,
    coords: T.Tensor
) -> Tuple[T.Tensor, T.Tensor]:
    x, y = coords[0, :], coords[1, :]

    # Calculate the corners for each coordinates.
    x0, x1 = T.floor(x), T.floor(x) + 1
    y0, y1 = T.floor(y), T.floor(y) + 1

    h, w = image.shape[0], image.shape[1]

    xmap0 = T.logical_and(0 <= x0, x0 <= w - 1)
    xmap1 = T.logical_and(0 <= x1, x1 <= w - 1)
    ymap0 = T.logical_and(0 <= y0, y0 <= h - 1)
    ymap1 = T.logical_and(0 <= y1, y1 <= h - 1)

    # The interpolation can happen only if all the 4 corners are in the image
    # domain
    all_corners_in_image_domain = T.logical_and(
        T.logical_and(xmap0, xmap1),
        T.logical_and(ymap0, ymap1)
    )
    ixs_where_all_corners_in_image_domain, = T.where(
        all_corners_in_image_domain
    )

    # Filter the coordinates
    xf = x[ixs_where_all_corners_in_image_domain]
    yf = y[ixs_where_all_corners_in_image_domain]
    x0f = x0[ixs_where_all_corners_in_image_domain]
    x1f = x1[ixs_where_all_corners_in_image_domain]
    y0f = y0[ixs_where_all_corners_in_image_domain]
    y1f = y1[ixs_where_all_corners_in_image_domain]

    x0i = x0f.int()
    x1i = x1f.int()
    y0i = y0f.int()
    y1i = y1f.int()

    image_flat = image.flatten()
    v00 = image_flat[y0i * w + x0i]
    v10 = image_flat[y0i * w + x1i]
    v01 = image_flat[y1i * w + x0i]
    v11 = image_flat[y1i * w + x1i]

    ax0, ax1 = x1f - xf, xf - x0f
    ay0, ay1 = y1f - yf, yf - y0f

    values = \
        ax0 * ay0 * v00 + ax1 * ay0 * v10 + \
        ax0 * ay1 * v01 + ax1 * ay1 * v11

    return values, ixs_where_all_corners_in_image_domain

def homogeneous(xy: T.Tensor) -> T.Tensor:
    n = xy.shape[1]
    ones = T.ones(n)
    xyh = T.ones((3, n))
    xyh[:2] = xy
    return xyh

def euclidean(xyh: T.Tensor) -> T.Tensor:
    xy1 = (xyh / xyh[2, :])
    return xy1[:2]

class BilinearInterpolation2d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, coords):
        return bilinear_interpolation_2d(x, coords)


class Homography(nn.Module):

    def __init__(self):
        super().__init__()
        self.interp = BilinearInterpolation2d()
        self.homography = nn.Parameter(T.zeros((3, 3)))

    def forward(self, image: T.Tensor):
        h, w = image.shape[2:]
        p2 = enumerate_coords(w, h)

        # Apply the homography
        p3 = homogeneous(p2).float().to(image.device)
        H = self.homography
        Hp3 = T.matmul(H, p3)
        Hp2 = euclidean(Hp3)

        image_warped = T.zeros_like(image)

        N, C = image.shape[:2]
        for n in range(N):
            for c in range(C):
                values, ixs = self.interp.forward(image[n, c], Hp2)
                plane_warped_c = image_warped[n, c].flatten()
                plane_warped_c[ixs] = values

        return image_warped

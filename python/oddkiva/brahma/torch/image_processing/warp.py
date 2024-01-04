import torch as T


def enumerate_coords(w: int, h: int) -> T.Tensor:
    x, y = T.meshgrid(T.arange(0, w), T.arange(0, h), indexing='xy')
    x, y = x.reshape((w * h,)), y.reshape((w * h,))
    p = T.stack((x, y))
    return p


def bilinear_interpolation_2d(image: T.Tensor, coords: T.Tensor) -> T.Tensor:
    x, y = coords[0, :], coords[1, :]

    # Calculate the corners for each coordinates.
    x0, x1 = T.floor(x), T.ceil(x)
    y0, y1 = T.floor(y), T.ceil(y)

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
    ixs_where_all_corners_in_image_domain = T.where(
        all_corners_in_image_domain
    )

    # Filter the coordinates
    x = x[ixs_where_all_corners_in_image_domain]
    y = y[ixs_where_all_corners_in_image_domain]
    x0 = x0[ixs_where_all_corners_in_image_domain].int()
    x1 = x1[ixs_where_all_corners_in_image_domain].int()
    y0 = y0[ixs_where_all_corners_in_image_domain].int()
    y1 = y1[ixs_where_all_corners_in_image_domain].int()

    image_flat = image.flatten()
    v00 = image_flat[y0 * w + x0]
    v10 = image_flat[y0 * w + x1]
    v01 = image_flat[y1 * w + x0]
    v11 = image_flat[y1 * w + x1]

    ax0, ax1 = x1 - x, x - x0
    ay0, ay1 = y1 - y, y - y0

    values = \
        ax0 * ay0 * v00 + ax1 * ay0 * v10 + \
        ax0 * ay1 * v01 + ax1 * ay1 * v11

    return values, T.stack((x, y))

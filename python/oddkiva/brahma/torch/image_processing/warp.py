import torch as T


def enumerate_coords(w: int, h: int) -> T.Tensor:
    x, y = T.meshgrid(T.range(0, w - 1), T.range(0, h - 1))
    x, y = x.reshape((w * h,)), y.reshape((w * h,))
    p = T.stack((x, y))
    return p


def which_coords_is_in_domain(x, xmin, xmax):
    return T.logical_and(xmin <= x, x < xmax)


def bilinear_interpolation_2d(image: T.Tensor, coords: T.Tensor) -> T.Tensor:
    x, y = coords[0, :], coords[1, :]

    # Calculate the corners for each coordinates.
    x0, x1 = T.floor(x), T.ceil(x)
    y0, y1 = T.floor(y), T.ceil(y)

    h, w = image.shape[0], image.shape[1]

    xmap0 = which_coords_is_in_domain(x0, 0, w - 1)
    xmap1 = which_coords_is_in_domain(x1, 0, w - 1)
    ymap0 = which_coords_is_in_domain(y0, 0, h - 1)
    ymap1 = which_coords_is_in_domain(y1, 0, h - 1)

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
    x0 = x0[ixs_where_all_corners_in_image_domain]
    x1 = x1[ixs_where_all_corners_in_image_domain]
    y0 = y0[ixs_where_all_corners_in_image_domain]
    y1 = y1[ixs_where_all_corners_in_image_domain]

    ax0, ax1 = x1 - x, x - x0
    ay0, ay1 = y1 - y, y - y0

    values = \
        ax0 * ay0 * image[y0][x0] + ax1 * ay0 * image[y0][x1] + \
        ax0 * ay1 * image[y1][x0] + ax1 * ay1 * image[y1][x1]

    out_image = T.zeros(image.shape, device=image.device)
    out_image.flatten()[y * w + x] = values

    return out_image

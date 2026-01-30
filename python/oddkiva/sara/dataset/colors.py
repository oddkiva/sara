# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import numpy as np


def generate_label_colors(
    label_count: int,
) -> np.ndarray:
    import matplotlib.colors as mcolors

    colors = np.array([
        mcolors.hex2color(c)
        for _, c in mcolors.CSS4_COLORS.items()
    ])
    colors = (colors * 255).astype(np.int32)
    colors = colors[:label_count]

    return colors

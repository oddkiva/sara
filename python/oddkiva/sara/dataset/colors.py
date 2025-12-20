# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import numpy as np


def generate_label_colors(
    label_count: int,
    colormap: str = 'rainbow'
) -> np.ndarray:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, label_count))
    colors = (colors[:, :3] * 255).astype(np.int32)
    return colors

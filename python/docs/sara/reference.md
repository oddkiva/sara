# Reference

::: oddkiva.sara.timer.Timer

## Graphics API

The Graphics API is built around PySide6.

### PySide6 Backend

The backend consists essentially derived QObject classes.

Underneath, we implement a graphics context class `GraphicsContext` that controls:

- the window management that *lives on the main thread* and
- the user thread that lives on a child thread, which requests rendering calls
  via Qt's signal-slot communication mechanism.

::: oddkiva.sara.graphics.derived_qobjects.graphics_context.GraphicsContext
::: oddkiva.sara.graphics.derived_qobjects.user_thread.UserThread
::: oddkiva.sara.graphics.derived_qobjects.painting_window.PaintingWindow

### Draw API

::: oddkiva.sara.graphics.image_draw.draw_point
::: oddkiva.sara.graphics.image_draw.draw_line
::: oddkiva.sara.graphics.image_draw.draw_rect
::: oddkiva.sara.graphics.image_draw.draw_circle
::: oddkiva.sara.graphics.image_draw.draw_ellipse
::: oddkiva.sara.graphics.image_draw.draw_text
::: oddkiva.sara.graphics.image_draw.draw_image

### Example

```python
--8<-- "oddkiva/sara/graphics/examples/hello_sara.py"
```


## Feature Detection and Matching

The Python API essentially wraps the optimized C++ code.

### Example

The code example shows how to use SIFT feature detection and matching frame by
frame.

```python
--8<-- "oddkiva/sara/features/examples/feature_matching.py"
```

#### Usage

```bash
python3 oddkiva/sara/feeatures/examples/feature_matching.py \
    https://sample.vodobox.net/skate_phantom_flex_4k/skate_phantom_flex_4k.m3u8?ref=developerinsider.co
```

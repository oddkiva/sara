# Graphics API

The Graphics API is built around PySide6.

## PySide6 Backend

The backend consists essentially derived QObject classes.

Underneath, we implement a graphics context class `GraphicsContext` that controls:

- the window management that *lives on the main thread* and
- the user thread that lives on a child thread, which requests rendering calls
  via Qt's signal-slot communication mechanism.

Note that there can be only one `GraphicsContext` object as we use the
*Singleton* pattern.

::: oddkiva.sara.graphics.derived_qobjects.graphics_context.GraphicsContext
::: oddkiva.sara.graphics.derived_qobjects.user_thread.UserThread
::: oddkiva.sara.graphics.derived_qobjects.painting_window.PaintingWindow

## Draw API

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

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

::: oddkiva.sara.graphics.derived_qobjects.graphics_context
::: oddkiva.sara.graphics.derived_qobjects.user_thread
::: oddkiva.sara.graphics.derived_qobjects.painting_window

## Draw API

::: oddkiva.sara.graphics.image_draw

### Example

```python
--8<-- "oddkiva/sara/graphics/examples/hello_sara.py"
```

import inspect
import importlib


def register():
    def decorator(foo):
        if inspect.isclass(foo):
            print("Registering class:", foo)
            print("class name:", foo.__name__)
            print("module:", importlib.import_module(foo.__module__))

        else:
            raise NotImplementedError()

        return foo

    return decorator

@register()
class Foo:

    def __init__(self):
        self._bar = None


def test_class_shenanigans():
    print("hello")

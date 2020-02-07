from PySide2 import QtGui

import vulkan as vk


VALIDATION_LAYERS = [
    'VK_LAYER_LUNARG_standard_validation'
]

ENABLED_VALIDATION_LAYERS = True


class InstanceProcAddr:

    def __init__(self, func):
        self.__func = func

    def __call__(self, *args, **kwargs):
        func_name = self.__func.__name__
        func = vk.vkGetInstanceProcAddr(args[0], func_name)
        if func:
            return func(*args, **kwargs)
        else:
            return vk.VK_ERROR_EXTENSION_NOT_PRESENT


@InstanceProcAddr
def vkCreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    pass


@InstanceProcAddr
def vkDestroyDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    pass


def debug_callback(*args):
    print('DEBUG: {} {}'.format(args[5], args[6]))
    return 0


class QueueFamilyIndices:

    def __init__(self):
        self.graphicsFamily = -1

    @property
    def is_complete(self):
        return self.graphicsFamily >= 0


class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__()
        self.setWidth(1280)
        self.setHeight(720)
        self.setTitle('Vulkan Python - PySide2')

        self.__instance = None
        self.__callback = None

        self.initVulkan()

    def __del__(self):
        if self.__callback:
            vkDestroyDebugReportCallbackEXT(self.__instance, self.__callback,
                                            None)

        if self.__instance:
            vk.vkDestroyInstance(self.__instance, None)
            print('instance destroyed')

    def initVulkan(self):
        self.__create_instance()
        self.__setup_debug_callback()
        self.__pick_physical_device()

    def __create_instance(self):
        app_info = vk.VkApplicationInfo(
            pApplicationName='Python VK',
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName='pyvulkan',
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION
        )

        extensions = [
            e.extensionName
            for e in vk.vkEnumerateInstanceExtensionProperties(None)
        ]

        instance_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info,
            enabledLayerCount=0,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )

        self.__instance = vk.vkCreateInstance(instance_info, None)

    def __setup_debug_callback(self):
        if not ENABLED_VALIDATION_LAYERS:
            return

        create_info = vk.VkDebugReportCallbackCreateInfoEXT(
            flags=vk.VK_DEBUG_REPORT_WARNING_BIT_EXT |
            vk.VK_DEBUG_REPORT_ERROR_BIT_EXT,
            pfnCallback=debug_callback
        )

        self.__callback = vkCreateDebugReportCallbackEXT(self.__instance,
                                                         create_info, None)

    def __pick_physical_device(self):
        physical_devices = vk.vkEnumeratePhysicalDevices(self.__instance)
        import ipdb; ipdb.set_trace()

        for device in physical_devices:
            if self.__is_device_suitable(device):
                self.__physical_device = device
                break

        assert self.__physical_device is not None

    def __is_device_suitable(self, device):
        indices = self.__find_queue_families(device)
        return indices.is_complete

    def __find_queue_families(self, device):
        indices = QueueFamilyIndices()
        family_properties = vk.vkGetPhysicalDeviceQueueFamilyProperties(device)
        for i, prop in enumerate(family_properties):
            if prop.queueCount > 0 and \
                    prop.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                indices.graphicsFamily = i

            if indices.is_complete:
                break

        return indices


if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)
    win = HelloTriangleApplication()
    win.show()

    def cleanup():
        global win
        del win

    app.aboutToQuit.connect(cleanup)

    sys.exit(app.exec_())

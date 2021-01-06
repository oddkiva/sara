Reading Vulkan tutorial
=======================

1. Create a Vulkan instance
2. Get a Vulkan physical device, typically a GPU or any accelerator.
3. Check that the physical device supports a queue family to which we can submit
   drawing commands.
4. Check that the physical device supports a queue family to which we can submit
   presentation commands (show the image buffer to the window).
   The presentation are not considered as part of Vulkan core but as a Vulkan
   extensions.
   - Create a Vulkan surface, this is platform-dependent.
   - We specify the image format of the surface (pixel format and colorspace)
5. Create a logical device that is associated to the physical device.
6. Create a swap chain, which is a queue of images ready to be presented to the
   screen.
   - To be able to use any image in the swap chain, we need to create an image
     view object to manipulate the data in each image.

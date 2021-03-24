Framebuffer
===========

As explained in a nice StackOverflow post, a framebuffer in OpenGL and in Vulkan
is not a buffer object but rather be viewed as a **collection of references to
buffer objects**. Here, a buffer should be understood as an array of bytes.

Programmatically a framebuffer can be seen as a **C struct** object as follows.

.. code-block:: cpp

   struct Framebuffer
   {
     void *color_buffer = nullptr;
     void *depth_buffer = nullptr;
     void *stencil_buffer = nullptr;
   };

   using FramebufferId = Framebuffer *;

Pursuing this explanation, we can understand:

- a color attachment as a pointer to the color array object.
- *binding a color attachment to the framebuffer object* corresponds to assigning
  the pointer of a color array object to the `color_buffer` data member.

And likewise for *depth* and *stencil* attachments.

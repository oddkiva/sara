Graphics
========

To run the code snippets below, create file `CMakeLists.txt` and place it in the same folder as where the source code lives::

  cmake_minimum_required(VERSION 2.6)

  find_package(DO COMPONENTS Core Graphics REQUIRED)
  include(${DO_USE_FILE})

  add_executable(graphics_example graphics_example.cpp)
  set_target_properties(graphics_example PROPERTIES
                        COMPILE_FLAGS "${ENABLE_CXX11} -DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}"
                        COMPILE_DEFINITIONS DO_STATIC)
  target_link_libraries(${_project_name} ${DO_LIBRARIES})


Note that the examples are also available in the `examples` directory.


Tutorial 1: Quick-start
-----------------------

The example below shows how to:

#. open a window,
#. draw something,
#. wait for a mouse click,
#. close the window.

.. literalinclude:: ../../../examples/Graphics/example_1.cpp
   :encoding: latin-1
   :language: cpp


You should be able to see something like below:

.. image:: example_1.png


Tutorial 2: Window Management (1/3)
-----------------------------------

.. literalinclude:: ../../../examples/Graphics/example_2.cpp
   :encoding: latin-1
   :language: cpp


Tutorial 3: Window Management (2/3)
-----------------------------------

.. literalinclude:: ../../../examples/Graphics/example_3.cpp
   :encoding: latin-1
   :language: cpp


Tutorial 4: Window Management (3/3)
-----------------------------------

.. literalinclude:: ../../../examples/Graphics/example_4.cpp
   :encoding: latin-1
   :language: cpp


Tutorial 5: Drawing with Integer Coordinates
--------------------------------------------

.. literalinclude:: ../../../examples/Graphics/example_5.cpp
   :encoding: latin-1
   :language: cpp


Tutorial 6: Drawing with Floating-Point Coordinates
---------------------------------------------------

.. literalinclude:: ../../../examples/Graphics/example_6.cpp
   :encoding: latin-1
   :language: cpp
   
Tutorial 7: Display Images (1/3)
--------------------------------

.. literalinclude:: ../../../examples/Graphics/example_7.cpp
   :encoding: latin-1
   :language: cpp


Tutorial 8: Display Images (2/3)
--------------------------------

.. literalinclude:: ../../../examples/Graphics/example_8.cpp
   :encoding: latin-1
   :language: cpp


Tutorial 9: Display Images (3/3)
--------------------------------

.. literalinclude:: ../../../examples/Graphics/example_9.cpp
   :encoding: latin-1
   :language: cpp
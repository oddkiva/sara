Imaging via Perspective Projection
==================================

Previously we have shown how to go from the camera coordinate system to the
world coordinate system which is defined by a global rigid transform
:math:`(\mathbf{R}, \mathbf{t})`

.. math::

  \mathbf{x}_W = \mathbf{R} \mathbf{x}_C + \mathbf{t}

Going back to the camera coordinates can be done as

.. math::

  \mathbf{x}_C = \mathbf{R}^T \mathbf{x}_W - \mathbf{R}^T \mathbf{t}

and thus the inverse rigid body transform is :math:`(\mathbf{R}^T, -\mathbf{R}^T
\mathbf{t})`:

Let us make two useful remarks:

- The inverse rotation matrix :math:`\mathbf{R}^T` stores the coordinates the
  world basis vectors expressed in the camera coordinate system.
- The translation :math:`-\mathbf{R}^T \mathbf{t}` stores the coordinates of the
  world origin expressed in the camera coordinate system.

In the camera pinhole model, lights reflected by 3D objects that passes through
the camera aperture have a straight line trajectory.

From a geometrical point of view, the camera forms the image by projecting every
3D material point with *metric* coordinates :math:`(x_C, y_C, z_C)` onto the
camera film plane defined by :math:`z_C = f` where :math:`f` is the focal
length.

The corresponding point imaged in the camera film has 3D *metric* coordinates
:math:`(u_C, v_C, f)` also expressed in the camera coordinate system.

Now in geometric computer vision, we prefer working in the *normalized* camera
coordinate system where we assume the focal length is :math:`f = 1`. This
amounts to dividing the coordinates by :math:`f = 1`. In other words, the base
length unit in the normalized camera coordinate system is the focal length
instead of the metre. And we use the camera calibration matrix to rescale the
normalized coordinates back to metric coordinates.

In the camera pinhole model, similar triangles holds and thus

.. math::

   \frac{u_C}{x_C} = \frac{v_C}{y_C} = \frac{1}{z_C} \\

i.e., in vector form

.. math::

   z_C \left[ \begin{array}{c} u_C \\ v_C \\ 1 \end{array} \right] =
   \mathbf{x}_C


By reinjecting the world coordinates we see that

.. math::

   z_C \left[ \begin{array}{c} u_C \\ v_C \\ 1 \end{array} \right] =
   \underbrace{\left[ \begin{array}{c|c}
     \mathbf{R}^T & -\mathbf{R}^T t
   \end{array} \right]}_{\mathbf{C}}

   \left[ \begin{array}{c} \mathbf{x}_W \\ 1 \end{array} \right] .

Thus we have introduced by construction a perspective projection matrix
:math:`\mathbf{C}`, in which homogeneous coordinates appear naturally.

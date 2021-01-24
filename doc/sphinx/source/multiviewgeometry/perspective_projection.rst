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

- The inverse rotation matrix :math:`\mathbf{R}^T` stores the basis
  coordinates the world axes expressed in the camera coordinate system.
- The translation :math:`-\mathbf{R}^T \mathbf{t}` stores the coordinate of the
  world origin expressed in the camera coordinate system

Now a camera is a physical projector that captures incoming lights reflected by
3D material points into the camera film.

Following the principles of geometric optics, the imaging process consists of
projecting 3D points :math:`(x_C, y_C, z_C)` into the film which can be viewed
as an image plane defined by :math:`z_C = 1`.

The imaged point in the camera film has film coordinates :math:`(u_C, v_C, 1)` in the
camera coordinate system. Using the basic proportionality theorem, it follows
that

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

Thus we have introduced by construction a perspective projection matrix :math:`\mathbf{C}`, in which homogeneous coordinates appear naturally.

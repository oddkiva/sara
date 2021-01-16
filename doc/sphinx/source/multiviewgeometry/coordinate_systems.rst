Changing coordinate systems
===========================

Let us detail how we change coordinate from the world coordinate system to the
camera coordinates.

Firstly we denote the center of the world coordinate system by :math:`O_W` and
the center of the camera coordinate system :math:`O_C`.

Let :math:`M` denote a 3D point of the world, then


.. math::

   \overrightarrow{O_W M} = \overrightarrow{O_W O_C} + \overrightarrow{O_\textrm{C}M} \\

   \overrightarrow{O_W M} = \overrightarrow{O_W O_C} + x_C \mathbf{i}_C + y_C \mathbf{j}_C + z_C \mathbf{k}_C\\

   \overrightarrow{O_W M} = \overrightarrow{O_W O_C} +
     \left[ \begin{array}{c|c|c}
     \mathbf{i}_C & \mathbf{j}_C & \mathbf{k}_C
     \end{array} \right]

     \left[ \begin{array}{c} x_C \\ y_C \\ z_C \end{array} \right] \\

We recognise the global rigid body motion :math:`(\mathbf{R}, \mathbf{t})`
where:

.. math::
   \mathbf{t} = \overrightarrow{O_W O_C} \\

   \mathbf{R} = \left[ \mathbf{i}_C | \mathbf{j}_C | \mathbf{k}_C \right] \\

In other words, the rotation matrix :math:`\mathbf{R}` whose column vectors are
**the basis vectors of camera coordinate system expressed in the bus coordinate
system**.


Euler Angles
============

The Euler angles :math:`(\psi, \theta, \phi)` are one parameterisation to
describe a rotation matrix.

Consider an airplane and the local coordinate system attached to it. The usual
convention of the local coordinates used in aerospace is as follows:
- The x-axis is the longitudinal axis pointing to the nose.
- The y-axis is the transversal axis pointing to the right.
- The z-axis is the vertical axis pointing downwards

In terms of rotation,
- The z-axis is the yaw axis (NO head movement)
- The y-axis is the pich axis (YES head movement).
- The x-axis is the roll axis ("Indian" MAYBE head movement).

Such a rotation is understood the composition of three elementary rotations in
the following order:

- Yaw about the :math:`z` axis by an angle :math:`\psi`,
- Pitch about the current :math:`y'` axis by an angle :math:`\theta`,
- Roll about the current :math:`X''` axis by an angle :math:`\phi`

The rotation matrix must be expressed with respect to the initial axes and it
can be show that:

.. math::

   R(\psi, \theta, \phi) = R_z(\psi) R_y(\theta) R_x(\phi)

Proof
-----

The composition of rotations is

.. math::

   R(\psi, \theta, \phi) = R_{x''}(\phi) R_{y'}(\theta) R_{z}(\psi)

To obtain :math:`R_{y'}`, we need to understand that

.. math::

   R_{y'}(\theta) = R_{z}(\theta) R_{y}(\theta) R_{z}(\theta)^T

Likewise:

.. math::

   R_{x''}(\psi) = R R_{y}(\theta) R^T

where

.. math::

   R = R_z(\psi) R_{y}(\theta)

By multiplying the three rotations, the inverse rotations will disappear and we get
the formula shown above.

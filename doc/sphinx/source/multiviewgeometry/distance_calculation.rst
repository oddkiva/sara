Distance Calculation From a Camera
==================================

This section provides an easily implementable method to calculate distance from
a camera. To calculate distances from a camera requires the knowledge of the
internal camera parameters.

Motivating Example
------------------

Consider a camera at a fixed height :math:`h` above the ground slight looking
down due to a nonzero pitch angle :math:`\theta`.

Consider a reference coordinate system centered in the camera center with the z-axis
still parallel to the ground plane where.

Let :math:`M` denote a point on the ground in this reference coordinate system.

.. math::

   M = x\ \mathbf{i} + h\ \mathbf{j} + z\ \mathbf{k}

The basis vectors of the reference coordinate system have the following
coordinates in the local camera coordinate system.

.. math::

   \begin{array}{ccl}
   \mathbf{i} &=&  \mathbf{i}_C \\
   \mathbf{j} &=&  \cos\theta\ \mathbf{j}_C + \sin\theta\ \mathbf{k}_C \\
   \mathbf{k} &=& -\sin\theta\ \mathbf{j}_C + \cos\theta\ \mathbf{k}_C
   \end{array}

Rewriting the coordinates of :math:`M` in the camera coordinates system

.. math::

   \begin{array}{ccl}
   M &=& x\ \mathbf{i}_C +
         h (\cos\theta\ \mathbf{j}_C + \sin\theta\ \mathbf{k}_C) +
         z (-\sin\theta\ \mathbf{j}_C + \cos\theta\ \mathbf{k}_C) \\

   M &=& x\ \mathbf{i}_C +
         (h \cos\theta - z \sin\theta)\ \mathbf{j}_C +
         (h \sin\theta + z\cos\theta)\ \mathbf{k}_C \\
   \end{array}

Thus in the camera coordinate system

.. math::

   \begin{array}{ccl}
   x_C &=& x \\
   y_C &=& h \cos\theta - z \sin\theta \\
   z_C &=& h \sin\theta + z\cos\theta \\
   \end{array}


The point :math:`M` is projected to the normalized camera plane. Let :math:`(u,
v)` denote its normalized camera coordinates.

Using the basic proportionality theorem in geometric optics:

.. math::

   \frac{u}{x_C} = \frac{v}{y_C} = \frac{1}{z_C}


We can inject the equations to calculate :math:`z`:

.. math::

   v = \frac{y_C}{z_C} \\
   v = \frac{h \cos\theta - z \sin\theta}{h \sin\theta + z\cos\theta} \\
   v (h \sin\theta + z\cos\theta) = h \cos\theta - z \sin\theta \\

Reordering in :math:`z`:

.. math::

   z (v\cos\theta + \sin\theta) = h (\cos\theta - v \sin\theta) \\
   z = h \frac{\cos\theta - v \sin\theta}{\sin\theta + v\cos\theta} \\


Generalization
--------------

We can generalize elegantly the reasoning by means of linear algebra. And it
will not matter which angular direction :math:`(\psi, \theta, \phi)` the camera
is looking at.

Let us assume that the road is *planar*. Following the usual convention in the
automotive industry, without loss of generality, any point on the ground is at
height :math:`z = 0`. Note this observation stays valid whether the vehicle
climbs or descends does not matter as long as the road is *planar*.

Suppose that we know the camera pose parameterised by the rigid body transform
:math:`(\mathbf{R}, \mathbf{t})` w.r.t. the vehicle coordinate system.

As said earlier, a ground point :math:`M` has more natural coordinates
:math:`\mathbf{x} = (x, y, 0)` in the vehicle coordinate system. Let
:math:`\mathbf{x}'= (x', y', z')` denote its coordinates in the camera
coordinate system. The rigid body transform relates the two vector quantities as
follows

.. math::

   \mathbf{x} = \mathbf{R} \mathbf{x}' + \mathbf{t} \\

Then we can introduce the camera matrix

.. math::

   \mathbf{C} = \left[ \begin{array}{c|c}
     \mathbf{R}^T & -\mathbf{R}^T t
   \end{array} \right]

which encodes the inverse rigid body transform :math:`(\mathbf{R}', \mathbf{t}')`
where:

.. math::

   \mathbf{R}' = \mathbf{R}^T \\
   \mathbf{t}' = -\mathbf{R}^T \mathbf{t}

We can calculate the camera coordinates :math:`\mathbf{x}'` from the reference
coordinates :math:`\mathbf{x}` as follows

.. math::

   \mathbf{x}' = \mathbf{C} \mathbf{x} \\

If the ground point is visible in the image at the following normalized
coordinates :math:`(u, v)`. then using the basic proportionality theorem in
geometric optics:

.. math::

   \frac{u}{x'} = \frac{v}{y'} = \frac{1}{z'},

We can derive a system of two equations.

.. math::

   \left\{ \begin{array}{lll}
   u z' - x' &=& 0 \\
   v z' - y' &=& 0 \\
   \end{array} \right.

Expanding the matrix operation as a linear system:

.. math::

   \mathbf{x}' = \mathbf{C} \mathbf{x} \\

yields

.. math::

   \left\{ \begin{array}{lll}
   x' &=& r'_{11} x + r'_{12} y + r'_{13} z + t'_{1}\\
   y' &=& r'_{21} x + r'_{22} y + r'_{23} z + t'_{2}\\
   z' &=& r'_{31} x + r'_{32} y + r'_{33} z + t'_{3}\\
   \end{array} \right.

Injecting these equations in the system of two equations yields

.. math::

   \left\{ \begin{array}{lll}
   u (r'_{31} x + r'_{32} y + r'_{33} z + t'_{3}) -
     (r'_{11} x + r'_{12} y + r'_{13} z + t'_{1})  &=& 0 \\

   v (r'_{31} x + r'_{32} y + r'_{33} z + t'_{3}) -
     (r'_{21} x + r'_{22} y + r'_{23} z + t'_{2}) &=& 0\\
   \end{array} \right.

Reordering

.. math::

   \left\{ \begin{array}{lll}
   (u r'_{31} - r'_{11}) x - (u r'_{32} - r'_{12}) y + (u r'_{33} - r'_{13}) z
   &=& t'_{1} - u t'_{3} \\

   (v r'_{31} - r'_{21}) x - (v r'_{32} - r'_{22}) y + (v r'_{33} - r'_{23}) z
   &=& t'_{2} - v t'_{3}  \\
   \end{array} \right.

Because we are dealing with a ground point, :math:`z = 0` and we obtain an
invertible linear system:

.. math::

   \left\{ \begin{array}{lll}
   (u r'_{31} - r'_{11}) x - (u r'_{32} - r'_{12}) y &=& t'_{1} - u t'_{3} \\
   (v r'_{31} - r'_{21}) x - (v r'_{32} - r'_{22}) y &=& t'_{2} - v t'_{3} \\
   \end{array} \right.

This will determine the missing coordinates :math:`x` and :math:`y`, which is
what we want.

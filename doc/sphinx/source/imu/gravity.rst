.. _sec-gravity-vector:

Accelerometer
*************

Actually What Does It Measure?
##############################

In this introductory section, I report my own experience as a beginner in IMU
data analysis and how I came up with my own intuition. The accelerometer is more
difficult to interpret than the gyroscope and the measurements noise make
acceleration readings hard to interpret.

When I started analysing data from IMU data from a recorded journey by
bicyle or by car, I was puzzled by what the IMU actually measures.

Initially I thought that the acceleration recorded by the IMU would correspond
to the sum of all external forces applied on the object, because of Newton's
second law. My questions:

- Why do we see recorded accelerations close to the gravity vector plus some
  deltas? Is it just the gravity vector and other things or just noise?
- If research papers write that the accelerometers measures the gravity vectors,
  why is the norm of recorded acceleration sometimes only about :math:`7.78` or
  :math:`8.4` on a car and not equal to :math:`9.81`?

These readings were puzzling to me and conflicting with my intuition.

Later, I found the following document and post, videos quite informative:

- http://www.chrobotics.com/docs/AN-1008-SensorsForOrientationEstimation.pdf
- https://physics.stackexchange.com/questions/402611/why-an-accelerometer-shows-zero-force-while-in-free-fall
- https://www.youtube.com/watch?v=-om0eTXsgnY

Previously I downloaded an accelerometer smartphone app and play with it. The
experiments will contradict this intuition. The smartphone lying on the table
would record vertical acceleration vector **pointing up to the sky** and with
magnitude close to :math:`9.81 m/s^2`. Nothing surprising until we start
questioning ourselves as for what the IMU actually measures.

Physics Interpretation
######################

Newton's Law
------------

Putting all these thoughts, experiments, and the essential bits of the different
readings together, it turns out indeed that the accelerometer measures the sum
of all external forces **except the gravitational force** acting upon the IMU
body.

.. math::

   m \mathbf{a}^{\text{IMU}} = \Sigma \mathbf{F} - m \mathbf{g}

Then with this formula, when the smartphone is lying on the table, we see that
what the IMU measures is the reaction force of the table. Because the smartphone
is immobile, its acceleration is zero. So this reaction force has to cancel the
gravitational force to prevent the smartphone from "falling down". That is the
reason why the IMU records acceleration vectors pointing up (positive elevation
:math:`a_z^{\text{IMU}} > 0`) and close to :math:`9.81 m/s^2`.

When the smartphone is falling down on the floor, the only external force
acting on it is the gravitational force. As we just the accelerometer records all
forces except the gravitational force and this explains why the recorded
accelerations are close to zero.

For example, in some recorded IMU data, the accelerations recorded on a car
every so often has a lower magnitude than the gravity vectord a possible
explanation is that the car springs are being compressed or elongated further
when the car is bumping on the road, causing the car elevation to fluctuate
slightly more.

Accelerometer as a Mass on a Spring
-----------------------------------

The stackoverflow post clarifies these observations by imagining an accelerometer
as a mass on a spring. So the accelerometric measure is essentially derived from
the measured elongation or compression of the spring.

Thus (knowing the spring constant :math:`k`) we can calculate the corresponding
reaction force of the table that maintains the mechanical equilibrium of the
smartphone.

Under a free fall, if we leave aside the aerodynamic resistance, there is no
reaction force is acting upon the spring and the spring is in a relaxed state,
(no elongation no compression). Have a look at the figures drawn in the
stackoverflow post.

Challenges
##########

The trouble with my current understanding is all the external forces should
include at the very least:

- the vehicle spring forces that supports the vehicle chassis on the ground.

  Complications happen when the vehicle bump every so often on non smooth
  surfaces. And it is not true anymore to assume that the car moves at constant
  altitude, and a constant pitch or roll angles, which we cannot really ignore
  in the IMU.

- the engine force that moves the vehicle.

  Another complication is that the engine makes the vehicle vibrate, thus
  creating non-negligible noise that cannot be easily distinguished from a
  truly small engine force

- the gravitational force applied to the vehicle chassis, which the IMU is
  attached to and thus forming altogether a solid object.

- the three fictitious forces since the IMU body frame is a non-inertial frame with
  respect to the world frame:

  - the Coriolis force

    .. math::

       -2m\ \mathbf{\Omega}_{\text{imu} / \text{world}} \times \mathbf{v}|_{\text{imu}}

  - the centrifugal acceleration

    .. math::

       -m\ \mathbf{\Omega}_{\text{imu} / \text{world}} \times
          (\mathbf{\Omega}_{\text{imu} / \text{world}} \times
          \mathbf{v}|_{\text{imu}})

  - the centrifugal force

    .. math::

       -m\ \frac{d \mathbf{v}|_{\text{imu}}}{dt}

  See https://en.wikipedia.org/wiki/Fictitious_force for details.

Add to this list some more subtle forces which can be neglected but are worth
mentioning. Because the earth is rotating, we may need to take into account:

- the Euler force from the rotating earth
- the Coriolis acceleration from the rotating earth
- the centrifugal force from the rotating earth.

They are easily calculable provided we know the GPS coordinates :math:`(\theta,
\phi)` (respectively the longitude and latitude angles), of the vehicle for In
the long run the effect of these forces may become non negligible.

I can recommend the reader to this excellent wikicoastal.org page that explains
these external force: http://www.coastalwiki.org/wiki/Coriolis_acceleration

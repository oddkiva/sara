.. _sec-gravity-vector:

Accelerometer
*************

Actually... What Does It Measure?
#################################

In this introductory section, I write about my own experience as a beginner in
IMU data analysis and how I came up with my own intuition.

The acceleration data is more difficult to interpret than the gyroscopic one and
the measurement noise makes them all the more difficult to interpret. Analyzing
for the first time IMU data recorded during a journey by bicyle or by car made
me ponder what the IMU actually measures.

Initially I thought that the acceleration recorded by the IMU would correspond
to the sum of all external forces applied on the object, because of Newton's
second law. I was asking myself:

- Why do we see recorded accelerations close to the gravity vector plus some
  deltas? Are the deltas other forces or just noise?
- If research papers write that the accelerometers measures the gravity vector,
  why is the norm of recorded acceleration sometimes only about :math:`7.78` or
  :math:`8.4` on a car and not equal to :math:`9.81`?

These readings were puzzling to me and conflicting with my intuition.

Later, I found the following documents, posts and videos that are quite
informative:

- http://www.chrobotics.com/docs/AN-1008-SensorsForOrientationEstimation.pdf
- https://physics.stackexchange.com/questions/402611/why-an-accelerometer-shows-zero-force-while-in-free-fall
- https://www.youtube.com/watch?v=-om0eTXsgnY

A while back ago, I downloaded an accelerometer smartphone app and played with
it. The smartphone lying on the table would record acceleration vector
**pointing up to the sky** and with magnitude close to :math:`9.81 m/s^2`.
Nothing surprised me back then until I started asking myself what the IMU
actually measures.

Physics Interpretation
######################

With Newton's Laws
------------------

Putting all these thoughts, experiments, and the essential bits of the different
readings together, the accelerometer does measure the sum of all external forces
**except the gravitational force** acting upon the IMU body.

.. math::

   m \mathbf{a}^{\text{IMU}} = \Sigma \mathbf{F} - m \mathbf{g}

Thus with this formula, when the smartphone is lying on the table, we realize
that what the IMU measures is **the reaction force of the table**.

Empirical Validation with the Smartphone App
""""""""""""""""""""""""""""""""""""""""""""

Indeed, because the smartphone is immobile, its acceleration is zero.
Necessarily this reaction force has to cancel the gravitational force to prevent
the smartphone from "falling down". The sum of all forces is (1) the reaction
force of the table and (2) the gravitational force and applying the formula
above, we should indeed expect the IMU to record acceleration vectors pointing
up. In other words, they have positive elevation :math:`a_z > 0` and a magnitude
close to :math:`9.81\ m/s^2`.

If we play with the smartphone app and rotate it, this should convince us and
also validate the formula above.

Empirical Validation on the Free-Fall Case
""""""""""""""""""""""""""""""""""""""""""

Let us now trust the formula above and the explanations in the YouTube video.

When the smartphone is falling down on the floor, the only external force
acting on it is the gravitational force. As we said it earlier, the
accelerometer records all forces except the gravitational force and this
explains why the recorded accelerations are close to zero.

In my experience, the acceleration magnitudes recorded during a car drive can
have a lower magnitude than the gravity vector. These usually coincides when the
car is bumping on the road. The car elevation fluctuates. A possible explanation
is due to the car springs which are being elongated then compressed.

And my personal interpretation is that these phases where the car springs begin
to compress can be viewed as "mini free-fall" situations.

Accelerometer as a Mass on a Spring
-----------------------------------

The abovementioned stackoverflow post explains why the acceleration recorded by
the IMU excludes the gravitational force. It suggests to imagine the
accelerometer as a mass on a spring. Then the acceleration measurements are
essentially derived from the measured elongation or compression of the spring.

Thus, knowing the spring constant :math:`k`, we can calculate the corresponding
reaction force of the table that maintains the mechanical equilibrium of the
smartphone.

Let us visualize the smartphone IMU lying on the table as a mass-spring system
lying on the table. This system is at mechanical equilibrium. Let us list the
external forces acting upon this system:

- The mass is being pushed:

  - down by the gravitational force :math:`-m_\text{mass} g`.
  - up by the spring force :math:`k \Delta x`

- The spring is compressed due the mass weight and being pushed:

  - down by the mass reaction force :math:`-m_\text{mass} g`
  - down by its own mass :math:`-m_\text{spring} g`
  - up by the reaction force of the table :math:`R_\text{table}`.

Now the mass-spring system as a whole has two forces acting on it only:

  - up by the reaction force of the table :math:`R_\text{table}`.
  - down by the mass reaction force :math:`-(m_\text{mass} + m_\text{spring}) g`

Because the mass-spring system is at mechanical equilibrium

.. math::

   k \Delta x = m_\text{mass} g \\

   R_\text{table} = (m_\text{mass} + m_\text{spring}) g \\

The gravity vector is calculated directly from the spring displacement from
its relaxed position:

.. math::

   g = \frac{k}{m_\text{mass}} \Delta x\\

We can also measure the reaction force of the table as

.. math::

   R_\text{table} = k (1 + \frac{m_\text{spring}}{m_\text{mass}}) \Delta x \\

Under a free fall (and neglecting the aerodynamic resistance), we should observe
that the spring is in a relaxed state and that that means no elongation and no
compression, thus :math:`\Delta x \approx 0`. And the accelerometer will measure
accelerations close to zero since :math:`\Delta x \approx 0`.

See the illustrations drawn in the StackOverflow post for complementary
information.


Challenges
##########

The trouble with my current understanding is that the external forces should
include at the very least:

- the vehicle spring forces that supports the vehicle chassis on the ground.

  Complications happen when the vehicle bump every so often on non smooth
  surfaces. And it is not true anymore to assume that the car moves at constant
  altitude, and a constant pitch or roll angles, which we cannot really ignore
  in the IMU.

- the engine force that moves the vehicle forward.

  Another complication is that the engine makes the vehicle vibrate, thus
  creating non-negligible noise that is not easily distinguished from a
  truly small engine force.

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
these external forces: http://www.coastalwiki.org/wiki/Coriolis_acceleration

.. _sec-ellipse-intersection:

Ellipse Intersections
*********************

This section details the implementation of ellipse intersections in **Sara**.
This section is extracted from the appendix of my thesis
:cite:`Ok:2013:phdthesis` and has been retouched a bit.

:cite:`Eberly` provides a comprehensive study on the computation of ellipses
intersection, namely the computation of its area and its intersection points.
This is non a trivial geometric problem. We complement :cite:`Eberly`’s study
with additional technical details about the area computation of two intersecting
ellipses.

Origin-Centered Axis-Aligned Ellipses
-------------------------------------

Let :math:`\mathcal{E}` be a ellipse with semi-major axis :math:`a` and
semi-minor axis :math:`b`, *i.e.*, :math:`\, a \geq b > 0`. Let us first suppose
that :math:`\mathcal{E}` is centered at the origin and is axis-aligned oriented,
*i.e*., such that the axis :math:`a` is along the :math:`x`-axis and the axis
:math:`b` along the :math:`y`-axis. Then,

.. important::
   the equation of ellipse :math:`\mathcal{E}` is

   .. math::
      :label: eq-elleq

      \frac{x^2}{a^2} + \frac{y^2}{b^2} = 1


Ellipse Area
~~~~~~~~~~~~

.. _figriemannsum:
.. figure:: ../figures/ellipsearea.png
    :align: center
    :scale: 90%

    Riemann sum approximating the upper quadrant area of the ellipse.

Using the symmetry in the ellipse, the area of ellipse :math:`\mathcal{E}` is
:math:`4` times the upper quadrant area of the ellipse, *i.e.*,

.. math:: \mathop{\mathrm{area}}(\mathcal{E}) = 4 \int_{0}^{a} y(x) \mathop{\mathrm{d}x} = 4 b \int_{0}^{a} \sqrt{1 - \frac{x^2}{a^2}} \mathop{\mathrm{d}x} = \pi a b

The integral is the limit of the Riemann sum as illustrated in Figure
[:ref:`figriemannsum`].

Let us detail the computation. We use the :math:`\mathcal{C}^1`-diffeomorphism
change of variable :math:`\frac{x}{a} = \sin \theta` which is valid for
:math:`[0, a] \rightarrow [0, \pi/2]`. We recall that a
:math:`\mathcal{C}^1`-diffeomorphism is an invertible differentiable function
with continuous derivative.

The differential is :math:`\mathop{\mathrm{d}x} = a \cos(\theta)
\mathop{\mathrm{d}\theta}` and hence,

.. important::
   .. math::

      \begin{aligned}
      \mathop{\mathrm{area}}(\mathcal{E})
        &= 4ab \int_{0}^{\pi/2} \cos^2(\theta) \mathop{\mathrm{d}\theta} \\
        &= 4ab \int_{0}^{\pi/2} \frac{1 + \cos(2\theta)}{2} \mathop{\mathrm{d}\theta} \\
        &= 4ab \left[ \frac{\theta}{2} + \frac{\sin(2\theta)}{4} \right]_{0}^{\pi/2} \\
        &= \pi a b.
      \end{aligned}

Area of an Elliptical Sector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this part, we review the computation of the area of an ellipse sector. We
complement :cite:`Eberly` with a bit more details.

.. _fig-ellsector:
.. figure:: ../figures/ellipticalsector.png
    :align: center
    :width: 90.0%

    The ellipse sector delimited by the polar angles :math:`(\theta_1,
    \theta_2)` is colored in blue

The elliptic sector area is delimited in polar coordinates by :math:`[\theta_1,
\theta_2]` (with :math:`\theta_1 < \theta_2`) as illustrated in
Figure [:ref:`fig-ellsector`]. Using polar coordinates, it equals to the
following nonnegative integral

.. math::

    A(\theta_1, \theta_2) = \frac{1}{2} \int_{\theta_1}^{\theta_2} r^2
    \mathop{\mathrm{d}\theta}.

The change of variable in polar coordinates is :math:`x = r \cos\theta` and
:math:`y = r \sin\theta` and, thus with Equation :eq:`eq-elleq`,
:math:`\displaystyle\frac{r^2 \cos^2(\theta)}{a^2} + \frac{r^2
\sin^2(\theta)}{b^2} = 1`, therefore


.. math::

    \displaystyle r^2 = \frac{a^2 b^2}{b^2 \cos^2(\theta) + a^2 \sin^2(\theta)}.

Plugging the formula of :math:`r` in the integral,

.. math::

   A(\theta_1, \theta_2) = \frac{a^2b^2}{2} \int_{\theta_0}^{\theta_1}
     \frac{\mathop{\mathrm{d}\theta}}{b^2 \cos^2(\theta) + a^2 \sin^2(\theta)}

Now the integrand
:math:`\frac{\mathop{\mathrm{d}\theta}}{b^2 \cos^2(\theta) + a^2 \sin^2(\theta)}`
is invariant by the transformation :math:`\theta \mapsto \theta+\pi`,
*i.e.*,

.. math::

   \frac{\mathop{\mathrm{d}\theta}}       {b^2 \cos^2(\theta) + a^2 \sin^2(\theta)} =
     \frac{\mathop{\mathrm{d}(\theta+\pi)}} {b^2 \cos^2(\theta+\pi) + a^2 \sin^2(\theta+\pi)}.

According to Bioche’s rule, a relevant change of variable is the
:math:`\mathcal{C}^1`-diffeomorphism change of variable
:math:`t = \tan(\theta)` which is valid for
:math:`]-\pi/2, \pi/2[ \rightarrow ]-\infty, \infty[`. Let us first
rewrite

.. math::

    \begin{aligned}
    A(\theta_1, \theta_2)
     &= \frac{a^2b^2}{2} \int_{\theta_1}^{\theta_2}
        \frac{\mathop{\mathrm{d}\theta}}{b^2 \cos^2(\theta) + a^2 \sin^2(\theta)}\\
     &= \frac{a^2b^2}{2} \int_{\theta_1}^{\theta_2}
        \frac{\frac{\mathop{\mathrm{d}\theta}}{\cos^2(\theta)}}{b^2  + a^2 \tan^2(\theta)}\\
     &= \frac{\cancel{a^2}b^2}{2} \int_{\theta_1}^{\theta_2}
        \frac{\frac{\mathop{\mathrm{d}\theta}}{\cos^2(\theta)}}{\cancel{a^2} (b/a)^2  +
        \tan^2(\theta))}\\
    \end{aligned}

Differentiating :math:`t=\tan\theta`,
:math:`\mathop{\mathrm{d}t} = \frac{\mathop{\mathrm{d}\theta}}{\cos^2(\theta)}`,
thus

.. math::

   \begin{aligned}
     A(\theta_1, \theta_2)
     &= \frac{b^2}{2} \int_{\tan\theta_1}^{\tan\theta_2}
        \frac{\mathop{\mathrm{d}t}}{(b/a)^2  + t^2}\\
     &= \frac{b^{\cancel{2}}}{2} \left[ \frac{a}{\cancel{b}}
        \arctan\left(\frac{a}{b} t\right)
        \right]_{\tan\theta_1}^{\tan\theta_2}\\
     &= \frac{ab}{2} \left[ \arctan\left(\frac{a}{b} t\right)
        \right]_{\tan\theta_1}^{\tan\theta_2} \\
     &= \frac{ab}{2} \left( \arctan\left(\frac{a}{b} \tan\theta_2\right) -
        \arctan\left(\frac{a}{b} \tan\theta_1\right) \right)\end{aligned}

Hence,

.. math::

   A(\theta_1, \theta_2)
     = \frac{ab}{2} \left( \arctan\left(\frac{a}{b} \tan\theta_2\right) -
       \arctan\left(\frac{a}{b} \tan\theta_1\right) \right)

.. warning::

    The integral is properly defined for
    :math:`(\theta_1, \theta_2) \in ]-\pi/2, \pi/2[`. But, using symmetry
    properties of the ellipse, we can easily retrieve the elliptical sector
    for any :math:`(\theta_1, \theta_2) \in ]-\pi, \pi[`.

Alternatively, :cite:`Eberly` provides a more convenient antiderivative because
it is defined in :math:`]-\pi, \pi]` as follows

.. math::

   F(\theta) = \frac{ab}{2}
     \left[
         \theta
       - \arctan \left( \frac{(b-a) \sin 2\theta}{(b+a) + (b-a)\cos 2 \theta}
                 \right)
     \right].

Hence, the elliptic sector area equals to the following *nonnegative*
quantity

.. important::
   .. math::

      \forall (\theta_1, \theta_2) \in ]-\pi, \pi], \ A(\theta_1, \theta_2) =
      \left| F(\theta_2) - F(\theta_1) \right|.

Area Bounded by a Line Segment and an Elliptical Arc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _figellsector2:
.. figure:: ../figures/ellipticalsector2.png
    :width: 90.0%

    The ellipse sector bounded by a line segment and the elliptical arc
    :math:`(\theta_1, \theta_2)` is colored in blue.

We are interested in computing the elliptic portion by a line segment
and the elliptical arc :math:`(\theta_1, \theta_2)` such that

.. math:: |\theta_2 - \theta_1| \leq \pi

This condition *is* important as a such elliptic portion always
corresponds to the blue elliptic portion in
Figure [:ref:`figellsector2`]. Let us denote the area of such portion
by :math:`B(\theta_1, \theta_2)`. Geometrically, we see that, if
:math:`|\theta_2 - \theta_1| \leq \pi`, then

.. math::

   \begin{aligned}
     B(\theta_1, \theta_2)
     &= \mathop{\mathrm{area}}(\mathrm{sector(\theta_1, \theta_2)})
      - \mathop{\mathrm{area}}(\mathrm{triangle(\theta_1, \theta_2)})\\
     &= A(\theta_1, \theta_2) - \frac{1}{2} |x_2y_1 - x_1y_2|\end{aligned}

where :math:`(x_i,y_i) = (r_i\cos\theta_i, r_i\sin\theta_i)` and
:math:`\displaystyle r_i = \frac{ab}{\sqrt{b^2 \cos^2(\theta_i)+a^2
\sin^2(\theta_i)}}` for :math:`i \in \{1,2\}`.

Note that the other portion corresponding to the red one in
Figure `3 <#fig:ellsector2>`__ has an area which equals to
:math:`\pi a b - B(\theta_1, \theta_2) \geq B(\theta_1, \theta_2)` if
:math:`|\theta_2 - \theta_1| \leq \pi`.

To summarize, our portion of interest, illustrated by the blue elliptic
portion in Figure `3 <#fig:ellsector2>`__, has an area which equals to

.. important::
   For any :math:`(\theta_1, \theta_2) \in ]-\pi, \pi]`,

   .. math::
        \ B(\theta_1, \theta_2) =
        \left\{
        \begin{array}{cl}
          \displaystyle A(\theta_1, \theta_2) - \frac{1}{2} |x_2y_1 - x_1y_2| &
          \textrm{if} \  |\theta_2 - \theta_1| \leq \pi \\
          \displaystyle \pi a b - A(\theta_1, \theta_2)  + \frac{1}{2} |x_2y_1 - x_1y_2| &
          \textrm{otherwise}
        \end{array}
        \right. .

General Ellipse Parameterization
--------------------------------

The previous sections has provided the basis for area of intersecting
ellipses. However, ellipses are neither centered at the origin nor
aligned with the axes of the reference frame in general. Therefore, an
ellipse :math:`\mathcal{E}` is entirely defined by the following
geometric information

-  a center :math:`\mathbf{x}_{\mathcal{E}}`,
-  axis radii :math:`(a_{\mathcal{E}}, b_{\mathcal{E}})`,
-  an orientation :math:`\theta_{\mathcal{E}}`, *i.e.*, the oriented angle
   between the :math:`x`-axis and the axis of radius :math:`a_{\mathcal{E}}`.

or more concisely by the pair
:math:`(\mathbf{x}_{\mathcal{E}}, \mathbf{\Sigma}_{\mathcal{E}})` where
the positive definite matrix
:math:`\mathbf{\Sigma}_{\mathcal{E}} \in \mathcal{S}^{++}_2`
is such that

.. math::
    :label: eq-sigma_eps

    \mathbf{\Sigma}_{\mathcal{E}} = \mathbf{R}_{\mathcal{E}} \mathbf{D}_{\mathcal{E}} \mathbf{R}_{\mathcal{E}}^T

where :math:`\mathbf{R}_{\mathcal{E}}` is a rotation matrix defined as

.. math::

   \mathbf{R}_{\mathcal{E}} \overset{\textrm{def}}{=}
   \begin{bmatrix}
     \cos\theta_{\mathcal{E}} & -\sin\theta_{\mathcal{E}}\\
     \sin\theta_{\mathcal{E}} &  \cos\theta_{\mathcal{E}}
   \end{bmatrix}

and :math:`\mathbf{D}_{\mathcal{E}}` is the diagonal matrix defined as

.. math::

   \mathbf{D}_{\mathcal{E}} \overset{\textrm{def}}{=}
   \begin{bmatrix}
     1/b_{\mathcal{E}}^2 & 0\\
     0 & 1/a_{\mathcal{E}}^2 & \\
   \end{bmatrix}

Note that Equation :eq:`eq-sigma_eps` is the singular value decomposition of
:math:`\mathbf{\Sigma}_{\mathcal{E}}` if the axis radii satisfy
:math:`a_{\mathcal{E}} \geq b_{\mathcal{E}}`. Thus more generally,

.. important::

   The ellipse :math:`\mathcal{E}` is characterized by the equation

   .. math::

      (\mathbf{x}-\mathbf{x}_{\mathcal{E}})^T \mathbf{\Sigma}_{\mathcal{E}} (\mathbf{x}- \mathbf{x}_{\mathcal{E}}) = 1

Or

.. math:: \mathbf{x}^T \mathbf{A}_{\mathcal{E}} \mathbf{x}+ \mathbf{b}_{\mathcal{E}}^T \mathbf{x}+ c_{\mathcal{E}} = 0

where :math:`\mathbf{A}_{\mathcal{E}} = \mathbf{\Sigma}_{\mathcal{E}}`,
:math:`\mathbf{b}_{\mathcal{E}} = 2 \mathbf{\Sigma}_{\mathcal{E}} \mathbf{x}_{\mathcal{E}}`
and
:math:`c_{\mathcal{E}} = \mathbf{x}_{\mathcal{E}}^T \mathbf{\Sigma}_{\mathcal{E}} \mathbf{x}_{\mathcal{E}} - 1`.
Denoting :math:`\mathbf{x}^T = [x, y]`, ellipse :math:`\mathcal{E}` can
be defined algebraically as

.. math:: E(x,y) = e_1 x^2 + e_2xy + e_3y^2 + e_4x + e_5y + e_6 = 0,

where
:math:`\mathbf{A}_{\mathcal{E}} = \begin{bmatrix} e_1 & e_2/2 \\ e_2/2 & e_3 \end{bmatrix}`,
:math:`\mathbf{b}_{\mathcal{E}}^T = [e_4, e_5]` and
:math:`c_{\mathcal{E}} = e_6`. This algebraic form is the convenient one
that we will use in order to compute the intersection points of two
intersecting ellipses.

Intersection Points of Two Ellipses
-----------------------------------

We explain how we can retrieve the intersection points of two ellipses. Our
presentation complements :cite:`Eberly`.

First let :math:`(\mathcal{E}_i)_{1 \leq i \leq 2}` be two ellipses defined as

.. math::
    :label: eq-twoellipses

    (x,y) \in \mathcal{E}_i \iff
    E_i(x,y) = e_{i1} x^2 + e_{i2} xy + e_{i3} y^2 + e_{i4} x + e_{i5} y + e_{i6} = 0

The intersection points of ellipses :math:`(\mathcal{E}_i)_{1 \leq i \leq 2}`
satisfy Equation :eq:`eq-twoellipses` for :math:`i \in \{1, 2\}`, *i.e.*, the
following equation system holds for intersection points

.. math::
    :label: eq-system

    \left\{ \begin{matrix} E_1(x,y) = 0 \\ E_2(x,y) = 0 \end{matrix} \right.

Now let us rewrite :math:`E_i(x,y)` as a quadratic polynomial in :math:`x`, *i.e.*,

.. math::

    E_i(x,y) = e_{i1} x^2
               + (e_{i2} y + e_{i4}) x
               + (e_{i3} y^2 + e_{i5} y + e_{i6}) = 0

Conveniently we define auxiliary polynomials in :math:`y`

.. math::

    \begin{aligned}
      p_0(y) &= e_{13} y^2 + e_{15} y + e_{16} &
      q_0(y) &= e_{23} y^2 + e_{25} y + e_{26} \\
      p_1(y) &= e_{12} y + e_{14} &
      q_1(y) &= e_{22} y + e_{24} \\
      p_2(y) &= e_{11} &
      q_2(y) &= e_{21}
    \end{aligned}

Introducing the polynomials above, Equation :eq:`eq-twoellipses` is rewritten as

.. math::

   \left\{
   \begin{matrix}
   p_2(y) x^2 + p_1(y) x + p_0(y) = 0 \\
   q_2(y) x^2 + q_1(y) x + q_0(y) = 0
   \end{matrix}
   \right.


Suppose we know the :math:`y`-coordinate of an intersection point, we can
calculate the :math:`x`-coordinate of this intersection point.

Indeed we multiply the first equation by :math:`q_2(y)` and the second equation
by :math:`p_2(y)`.

.. math::

   \left\{
   \begin{matrix}
   q_2(y) \times \left( p_2(y) x^2 + p_1(y) x + p_0(y) \right)= 0\times q_2(y)\\
   p_2(y) \times \left( q_2(y) x^2 + q_1(y) x + q_0(y) \right)= 0\times p_2(y)
   \end{matrix}
   \right.

Then subtracting the first equation from the second equation, the monomial
:math:`x^2` disappears. Thus:

.. important::

   .. math::
      :label: eq:xinter

      x = \frac{p_0(y)q_2(y) - p_2(y)q_0(y)}{p_1(y)q_2(y) - p_2(y)q_1(y)}.

Furthermore, Equation :eq:`eq-system` is equivalent to the following augmented
equation system

.. math::

   \left\{
     \begin{array}{rl}
               E_1(x,y) &= 0 \\
       x\times E_1(x,y) &= 0 \\
               E_2(x,y) &= 0 \\
       x\times E_2(x,y) &= 0 \\
     \end{array}
   \right.,

And we see more clearly in matrix notation that

.. important::

   :math:`[1, x, x^2, x^3]^T` is in the nullspace of :math:`\mathbf{B}(y)`,
   where :math:`\mathbf{B}(y)` is defined as

   .. math::
       :label: eq-system2

       \underbrace{
         \begin{bmatrix}
           p_{0}(y) & p_{1}(y) & p_{2}(y) & 0  \\
           0  & p_{0}(y) & p_{1}(y) & p_{2}(y) \\
           q_{0}(y) & q_{1}(y) & q_{2}(y) & 0  \\
           0  & q_{0}(y) & q_{1}(y) & q_{2}(y)
         \end{bmatrix}
       }_{\mathbf{B}(y)}
       \begin{bmatrix}
         1 \\ x \\ x^2 \\ x^3
       \end{bmatrix}
       =
       \begin{bmatrix}
         0 \\ 0 \\ 0 \\ 0
       \end{bmatrix}


We observe that the vector :math:`[1, x, x^2, x^3]^T` is never zero for any
real value :math:`x`. Thus necessarily the nullspace
:math:`\text{Null}(\mathbf{B}(y)` is always nontrivial and that means the
determinant of :math:`\mathbf{B}(y)` has to be zero.

.. important::
   Let the polynomial :math:`R` be defined as

   .. math::

      R \overset{\textrm{def}}{=}
      \left( p_{0}q_{2} - p_{2}q_{0} \right)^2 -
      \left( p_{0}q_{1} - p_{1}q_{0} \right)
      \left( p_{1}q_{2} - p_{2}q_{1} \right),

   Equation :eq:`eq-system` is equivalent to the following quartic equation in
   :math:`y`.

   .. math::
      :label: eq-detBy

      \det(\mathbf{B}(y)) = R(y) = 0,

Using any polynomial solver, we get the :math:`4` roots :math:`(y_i)_{1\leq
i\leq 4}` of the quartic polynomial :math:`R` and only keep those that are real.
Finally :math:`(x_i)_{1\leq i \leq 4}` are deduced from
Equation :eq:`eq:xinter`.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

In *Sara*, we can use several solvers to retrieve the roots of polynomial
:math:`R`.

1. **Companion matrix** approach: since *Sara* depends on *Eigen*, *Eigen* has
   an unsupported Polynomial solver using this simple approach.
2. **Jenkins-Traub** iterative but very accurate approach also available in
   *Sara*.
3. **Ferrari**’s method available in *Sara*.

The implementation in *Sara* uses Ferrari's method. While more tedious to
implement, the method has the advantage of being direct. Also, we experimentally
observe Ferrari’s method can sometimes be numerically inaccurate in particular
situations where for example one of the ellipse is quasi-degenerate.

In the future, depending on the use case, we can polish the roots to refine the
root values.


Intersection Area of Two Ellipses
---------------------------------

Our presentation complements :cite:`Eberly`. In the rest of the section, we
consider two ellipses :math:`(\mathcal{E}_i)_{1 \leq i \leq 2}` and we
respectively denote

-  the axes of ellipse :math:`\mathcal{E}_i` by :math:`(a_i, b_i)`, the
   ellipse center by :math:`\mathbf{x}_i`, the orientation by
   :math:`\theta_i`, and the direction vectors of axis :math:`a_i` and
   :math:`b_i` by

   .. math::

      \begin{aligned}
          \mathbf{u}_i &\overset{\textrm{def}}{=}\begin{bmatrix}  \cos(\theta_i) \\ \sin(\theta_i) \end{bmatrix} &
        \mathbf{v}_i &\overset{\textrm{def}}{=}\begin{bmatrix} -\sin(\theta_i) \\ \cos(\theta_i) \end{bmatrix}\end{aligned}

-  the area of the elliptic portion bounded a line segment and an arc
   for ellipse :math:`\mathcal{E}_i` by :math:`B_i`,

-  the number of intersection points by :math:`L`,

-  the intersection points by :math:`\mathbf{p}_i` for
   :math:`i \in \llbracket 1, L \rrbracket`, sorted in a
   counter-clockwise order, *i.e.*,

   .. math::
       :label: eq:counterclockwise

       \forall i \in \llbracket 1, L-1\rrbracket,\quad \angle\left([1,0]^T,
       \mathbf{p}_i\right) \ < \ \angle\left([1,0]^T, \mathbf{p}_{i+1}\right)

   where :math:`\angle(.,.)` denotes the angle between two vectors in
   the plane :math:`\mathbb{R}^2`.

-  the polar angles of points :math:`(\mathbf{p}_i)_{1\leq i \leq L}`
   with respect to ellipses :math:`\mathcal{E}_1` and
   :math:`\mathcal{E}_2` by :math:`(\phi_i)_{1\leq i \leq 2}` and
   :math:`(\psi_i)_{1\leq i \leq 2}`, *i.e.*,

   .. math::

      \begin{gathered}
        \forall i \in \llbracket 1, L\rrbracket,
          \phi_i \overset{\textrm{def}}{=}\angle\left(\mathbf{u}_1, \mathbf{p}_i - \mathbf{x}_1\right) \\
          \forall i \in \llbracket 1, L\rrbracket,
          \psi_i \overset{\textrm{def}}{=}\angle\left(\mathbf{u}_2, \mathbf{p}_i - \mathbf{x}_2\right)\end{gathered}

Retrieving the polar angles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To retrieve the polar angles, we need to place ourselves in the coordinate
system :math:`(\mathbf{x}_i, \mathbf{u}_i, \mathbf{v}_i)`. Using the convenient
function :math:`\mathrm{atan2}` giving values in :math:`]-\pi,\pi]`, we
have

.. math::

   \begin{aligned}
     \phi_i &= \mathrm{atan2}
     \left(
       \langle \mathbf{p}_i-\mathbf{x}_1, \mathbf{v}_1 \rangle,
       \langle \mathbf{p}_i-\mathbf{x}_1, \mathbf{u}_1 \rangle
     \right)\\
     \psi_i &= \mathrm{atan2}
     \left(
       \langle \mathbf{p}_i-\mathbf{x}_2, \mathbf{v}_2 \rangle,
       \langle \mathbf{p}_i-\mathbf{x}_2, \mathbf{u}_2 \rangle
     \right)\end{aligned}

0 or 1 intersection point
~~~~~~~~~~~~~~~~~~~~~~~~~

Either one ellipse is contained in the other or there are separated as
illustrated in Figure [:ref:`figinter01`].

.. _figinter01:
.. table:: Cases where there is zero or one intersection point.

   +----------------------------------+----------------------------------+
   | .. image:: ../figures/test0a.png | .. image:: ../figures/test0b.png |
   +----------------------------------+----------------------------------+
   | .. image:: ../figures/test1a.png | .. image:: ../figures/test1b.png |
   +----------------------------------+----------------------------------+

An ellipse, say :math:`\mathcal{E}_1`, is contained in the other
:math:`\mathcal{E}_2` if and only if its center satisfies
:math:`E_2(\mathbf{x}_1) < 0`. In that case, the area of the intersection is
just the area of ellipse :math:`\mathcal{E}_1`.  Otherwise, if there is no
containment, the intersection area is zero. In summary,

.. math::
    :label: eq-area01

    \mathop{\mathrm{area}}(\mathcal{E}_1 \cap \mathcal{E}_2) = \left\{
    \begin{array}{ll}
    \pi a_1 b_1 & \textrm{if}\ E_2(\mathbf{x}_1) < 0\\
    \pi a_2 b_2 & \textrm{if}\ E_1(\mathbf{x}_2) < 0\\
    0 & \textrm{otherwise}
    \end{array}
    \right.

2 intersection points
~~~~~~~~~~~~~~~~~~~~~

We will not detail the case when Polynomial :eq:`eq-detBy` have :math:`2` roots
with multiplicity :math:`2`. This still corresponds to the case where there are
two intersection points. But because of the root multiplicities, one ellipse is
contained in the other one and then Equation eq:`eq-area01` gives the correct
intersection area.

Otherwise, we have to consider two cases as illustrated in
Figure [:ref:`figinter2`], which :cite:`Eberly` apparently forgot to consider.
Namely, the cases correspond to whether the center of ellipses
:math:`\mathcal{E}_1` and :math:`\mathcal{E}_2` are on the same side or on
opposite side with respect to the line :math:`(\mathbf{p}_1, \mathbf{p}_2)`.

.. _figinter2:
.. table:: Cases where there are two intersection points.

    +-----------------------------------+-----------------------------------+
    | .. image:: ../figures/inter2a.png | .. image:: ../figures/inter2b.png |
    +-----------------------------------+-----------------------------------+

Denoting a unit normal of the line going across the intersection points
:math:`(\mathbf{p}_1, \mathbf{p}_2)` by :math:`\mathbf{n}` (cf.
Figure `1.9 <#fig:inter2>`__). If the ellipse centers
:math:`\mathbf{x}_1` and :math:`\mathbf{x}_2` are on opposite side with
respect to the line :math:`(\mathbf{p}_1, \mathbf{p}_2)`, *i.e.*,

.. math::

   \langle \mathbf{n}, \mathbf{x}_1 - \mathbf{p}_1 \rangle \langle \mathbf{n},
   \mathbf{x}_2 - \mathbf{p}_1 \rangle < 0,

then

.. math::

   \mathop{\mathrm{area}}(\mathcal{E}_1 \cap \mathcal{E}_2) =
       B_1(\phi_1, \phi_2) + B_2(\psi_1, \psi_2)

If they are on the same side with respect to the line
:math:`(\mathbf{p}_1, \mathbf{p}_2)`, *i.e.*,

.. math::

    \langle \mathbf{n}, \mathbf{x}_1 - \mathbf{p}_1 \rangle
    \langle \mathbf{n}, \mathbf{x}_2 - \mathbf{p}_1 \rangle > 0,

then

.. math::
    :label: eqinter2b

    \mathop{\mathrm{area}}(\mathcal{E}_1 \cap \mathcal{E}_2) =
    \left\{
    \begin{array}{ll}
      \displaystyle \left( \pi a_1 b_1 - B_1(\phi_1, \phi_2) \right) +
      B_2(\psi_1, \psi_2) &
      \textrm{if}
      |\langle\mathbf{n},\mathbf{x}_1-\mathbf{p}_1\rangle| \leq
      |\langle\mathbf{n},\mathbf{x}_2-\mathbf{p}_1\rangle| \\
      \\
      \displaystyle
      B_1(\phi_1, \phi_2) +
      \left( \pi a_2 b_2 - B_2(\psi_1, \psi_2) \right) &
      \textrm{otherwise}.
    \end{array}
    \right.


Note that the condition
:math:`|\langle\mathbf{n},\mathbf{x}_1-\mathbf{p}_1\rangle| \leq
|\langle\mathbf{n},\mathbf{x}_2-\mathbf{p}_1\rangle|` in
Equation :eq:`eqinter2b` just expresses the fact that the distance of ellipse
center :math:`\mathbf{x}_1` to the line :math:`(\mathbf{p}_1, \mathbf{p}_2)` is
smaller than the distance of ellipse center :math:`\mathbf{x}_2` to the line
:math:`(\mathbf{p}_1, \mathbf{p}_2)`.


3 and 4 intersection points
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _fig-inter34:
.. table:: Cases where there are three of four intersection points.

    +-----------------------------------+-----------------------------------+
    | .. image:: ../figures/inter3.png  | .. image:: ../figures/inter4.png  |
    +-----------------------------------+-----------------------------------+

These cases are rather easy to handle. Indeed, we see geometrically from
Figure [:ref:`fig-inter34`],

.. math::

   \mathop{\mathrm{area}}(\mathcal{E}_1 \cap \mathcal{E}_2) =
       \sum_{i=1}^{L}
         \underbrace{\min \left(
           B_1(\phi_i, \phi_{i+1}),
           B_2(\psi_i, \psi_{i+1})
         \right)}_{\textrm{smallest of elliptic portion area}} +
       \underbrace{\frac{1}{2} \sum_{i=1}^{L} \left|
           \det\left(\mathbf{p}_i, \mathbf{p}_{i+1}\right)
       \right|}_{\textrm{area of polygon}\ (\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_L)}

with :math:`\phi_{L+1} = \phi_1`, :math:`\psi_{L+1} = \psi_1` and
:math:`\mathbf{p}_{L+1} = \mathbf{p}_1`.

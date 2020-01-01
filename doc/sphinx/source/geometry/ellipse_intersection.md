Ellipse Intersections
=====================


:cite:`Eberly` provides a comprehensive study on the computation of ellipses
intersection, namely the computation of its area and its intersection points.
This is a non-trivial geometric problem. We complement :cite:`Eberly`'s study
with some additional technical details about the area computation of two
intersecting ellipses.


Origin-Centered Axis-Aligned Ellipses
*************************************

Let :math:`\mathcal{E}` be an ellipse with semi-major axis :math:`a` and
semi-minor axis :math:`b`. Let us first suppose that :math:`\mathcal{E}` is centered
at the origin and is axis-aligned oriented, \ie, such that the axis :math:`a` is along
the :math:`x`-axis and the axis :math:`b` along the :math:`y`-axis. Then the equation of ellipse
:math:`\mathcal{E}` is

.. math::
    :label: eq:elleq

    \frac{x^2}{a^2} + \frac{y^2}{b^2} = 1


Ellipse Area
************

.. _figriemannsum:
.. figure:: figures/ellipsearea.png
    :align: center
    :figclass: align-center
    :scale: 90 %

    Riemann sum approximating the upper quadrant area of the ellipse.

By symmetry the area of ellipse :math:`\mathcal{E}` is :math:`4` times the upper
quadrant area of the ellipse, *i.e.*,

.. math::

    \text{area}(\mathcal{E}) = 4 \int_{0}^{a} y(x) \operatorname{d} x \\

    = 4 b \int_{0}^{a} \sqrt{1 - \frac{x^2}{a^2}} \operatorname{d} x \\

    = \pi a b

The integral is the limit of the Riemann sum as illustrated in
Figure :numref:`figriemannsum`.

Let us detail the computation. We use the :math:`\mathcal{C}^1`-diffeomorphism
change of variable :math:`\frac{x}{a} = \sin \theta` which is valid for
:math:`[0, a] \rightarrow [0, \pi/2]`.

Recall that a :math:`\mathcal{C}^1`-diffeomorphism is an invertible
differentiable function with continuous derivative.

Differentiating, :math:`\operatorname{d}x = a \cos(\theta) \operatorname{d}
\theta`, and hence,

.. math::

    \text{area}(\mathcal{E})
    = 4ab \int_{0}^{\pi/2} \cos^2(\theta) \operatorname{d} \theta \\
    = 4ab \int_{0}^{\pi/2} \frac{1 + \cos(2\theta)}{2} \operatorname{d} \theta \\
    = 4ab \left[ \frac{\theta}{2} + \frac{\sin(2\theta)}{4} \right]_{0}^{\pi/2} \\
    = \pi a b.


Area of an Elliptical Sector
****************************

In this part, we review the computation of the area of an ellipse sector. It has
already been covered in :cite:`Eberly` but computation details are not expanded
in :cite:`Eberly`.

.. _figellsector:

.. figure:: figures/ellipticalsector.png
    :align: center
    :figclass: align-center
    :scale: 90 %

    The ellipse sector delimited by the polar angles :math:`(\theta_1,
    \theta_2)` is colored in blue.

The elliptic sector area is delimited in polar coordinates by :math:`[\theta_1,
\theta_2]` with :math:`\theta_1 < \theta_2` as illustrated in
Figure :numref:`figellsector`. Using polar coordinates, it equals to the following
nonnegative integral

.. math::

    A(\theta_1, \theta_2) = \frac{1}{2} \int_{\theta_1}^{\theta_2} r^2 \operatorname{d}\theta.

The change of variable in polar coordinates is :math:`x = r \cos\theta` and
:math:`y = r \sin\theta` and, thus with Equation :eq:`eq:elleq`,

.. math::

  \displaystyle\frac{r^2 \cos^2(\theta)}{a^2} + \frac{r^2 \sin^2(\theta)}{b^2} = 1,

therefore

.. math::

    \displaystyle r^2 = \frac{a^2 b^2}{b^2 \cos^2(\theta) + a^2 \sin^2(\theta)}.

Plugging the formula of :math:`r` in the integral,

.. math::

    A(\theta_1, \theta_2) = \frac{a^2 b^2}{2} \int_{\theta_0}^{\theta_1}
    \frac{\operatorname{d} \theta }{b^2 \cos^2(\theta) + a^2 \sin^2(\theta)}


Now  the integrand :math:`\displaystyle \frac{\operatorname{d}\theta}{b^2
\cos^2(\theta) + a^2 \sin^2(\theta)}` is invariant by the transformation
:math:`\theta \mapsto \theta+\pi`, *i.e.*,

.. math::

    \frac{\operatorname{d} \theta}       {b^2 \cos^2(\theta) + a^2 \sin^2(\theta)} =
    \frac{\operatorname{d} (\theta+\pi)} {b^2 \cos^2(\theta+\pi) + a^2 \sin^2(\theta+\pi)}.

According to Bioche's rule, a relevant change of variable is the
:math:`\mathcal{C}^1`-diffeomorphism change of variable :math:`t = \tan(\theta)`
which is valid for :math:`]-\pi/2, \pi/2[ \rightarrow ]-\infty, \infty[`. Let us first
rewrite

.. math::

    A(\theta_1, \theta_2)
    = \frac{a^2b^2}{2} \int_{\theta_1}^{\theta_2}
      \frac{\operatorname{d} \theta}{b^2 \cos^2(\theta) + a^2 \sin^2(\theta)}\\
    = \frac{a^2b^2}{2} \int_{\theta_1}^{\theta_2}
      \frac{\frac{\operatorname{d} \theta}{\cos^2(\theta)}}{b^2  + a^2 \tan^2(\theta)}\\
    = \frac{\cancel{a^2}b^2}{2} \int_{\theta_1}^{\theta_2}
       \frac{\frac{\operatorname{d} \theta}{\cos^2(\theta)}}{\cancel{a^2} (b/a)^2  +
       \tan^2(\theta))}\\

Differentiating :math:`t=\tan\theta`, :math:`\operatorname{d}t =
\frac{\operatorname{d} \theta}{\cos^2(\theta)}`, thus

.. math::

    A(\theta_1, \theta_2)
    = \frac{b^2}{2} \int_{\tan\theta_1}^{\tan\theta_2}
      \frac{\operatorname{d}t}{(b/a)^2  + t^2}\\
    = \frac{b^{\cancel{2}}}{2} \left[ \frac{a}{\cancel{b}}
      \arctan\left(\frac{a}{b} t\right)
      \right]_{\tan\theta_1}^{\tan\theta_2}\\
    = \frac{ab}{2} \left[ \arctan\left(\frac{a}{b} t\right)
      \right]_{\tan\theta_1}^{\tan\theta_2} \\
    = \frac{ab}{2} \left( \arctan\left(\frac{a}{b} \tan\theta_2\right) -
      \arctan\left(\frac{a}{b} \tan\theta_1\right) \right)

Hence,

.. math::

    A(\theta_1, \theta_2)
    = \frac{ab}{2} \left( \arctan\left(\frac{a}{b} \tan\theta_2\right) -
      \arctan\left(\frac{a}{b} \tan\theta_1\right) \right)

.. warning::

    The integral is properly defined for :math:`(\theta_1, \theta_2) \in
    ]-\pi/2, \pi/2[`. However by symmetry of the ellipse, we can easily retrieve
    the elliptical sector for any :math:`(\theta_1, \theta_2) \in ]-\pi, \pi[`.

Alternatively, :cite:`Eberly` provides a more convenient antiderivative because
it is defined in :math:`]-\pi, \pi]` as follows

.. math::

    F(\theta) = \frac{ab}{2}
    \left[
        \theta
      - \arctan \left( \frac{(b-a) \sin 2\theta}{(b+a) + (b-a)\cos 2 \theta}
                \right)
    \right].

Hence, the elliptic sector area equals to the following *nonnegative* quantity

.. math::

	  \forall (\theta_1, \theta_2) \in ]-\pi, \pi],
	  A(\theta_1, \theta_2) = \left| F(\theta_2) - F(\theta_1) \right|.


Area Bounded by a Line Segment and an Elliptical Arc
****************************************************

\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/ellipticalsector2.pdf}
  \caption[Ellipse sector bounded by a line segment and the elliptical arc `(\theta_1, \theta_2)`.]{The ellipse sector bounded by a line segment and the elliptical arc `(\theta_1, \theta_2)` is colored in blue.}
  :label:{fig:ellsector2}
\end{figure}

We are interested in computing the elliptic portion by a line segment and the
elliptical arc :math:`(\theta_1, \theta_2)` such that

.. math::

	  |\theta_2 - \theta_1| \leq \pi

This condition *is* important as a such elliptic portion always corresponds to
the blue elliptic portion in Figure~\ref{fig:ellsector2}. Let us denote the area
of such portion by :math:`B(\theta_1, \theta_2)`. Geometrically, we see that, if
:math:`|\theta_2 - \theta_1| \leq \pi`, then

.. math::

    B(\theta_1, \theta_2)
    = \text{area}(\mathrm{sector(\theta_1, \theta_2)})
    - \text{area}(\mathrm{triangle(\theta_1, \theta_2)})\\
    = A(\theta_1, \theta_2) - \frac{1}{2} |x_2 y_1 - x_1 y_2|

where :math:`(x_i,y_i) = (r_i\cos\theta_i, r_i\sin\theta_i)` and
:math:`\displaystyle r_i = \frac{ab}{\sqrt{b^2 \cos^2(\theta_i) + a^2
\sin^2(\theta_i)}}` for :math:`i = \{1,2\}`.

Note that the other portion corresponding to the red one in
Figure~\ref{fig:ellsector2} has an area which equals to
:math:`\pi a b - B(\theta_1, \theta_2) \geq B(\theta_1, \theta_2)`
if :math:`|\theta_2 - \theta_1| \leq \pi`.

To summarize, our portion of interest, illustrated by the blue elliptic portion
in Figure~\ref{fig:ellsector2}, has an area which equals to

.. math::

    \forall (\theta_1, \theta_2) \in ]-\pi, \pi], \ B(\theta_1, \theta_2) =
    \left\{
    \begin{array}{cl}
    	\displaystyle A(\theta_1, \theta_2) - \frac{1}{2} |x_2y_1 - x_1y_2| &
    	\textrm{if} \  |\theta_2 - \theta_1| \leq \pi \\
    	\displaystyle \pi a b - A(\theta_1, \theta_2)  + \frac{1}{2} |x_2y_1 - x_1y_2| &
    	\textrm{otherwise}
    \end{array}
    \right. .


General Ellipse Parameterization
********************************

The previous sections has provided the basis for area of intersecting ellipses.
However, ellipses are neither centered at the origin nor aligned with the axes
of the reference frame in general. Therefore, an ellipse `\mathcal{E}` is
entirely defined by the following geometric information

- a center :math:`\mathbf{x}_{\mathcal{E}}`,
- axis radii :math:`(a_{\mathcal{E}}, b_{\mathcal{E}})`,
- an orientation :math:`\theta_{\mathcal{E}}`, \ie, the oriented angle between the
  :math:`x`-axis and the axis of radius :math:`a_{\mathcal{E}}`.

or more concisely by the pair :math:`(\mathbf{x}_{\mathcal{E}},
\mathbf{\Sigma}_{\mathcal{E}})` where the positive definite matrix
:math:`\mathbf{\Sigma}_{\mathcal{E}} \in \mathcal{S}^{++}_2` is such that

.. math::
    :label: eq:sigma_eps

    \mathbf{\Sigma}_{\mathcal{E}} = \mathbf{R}_{\mathcal{E}} \mathbf{D}_{\mathcal{E}} \mathbf{R}_{\mathcal{E}}^T

where :math:`\mathbf{R}_{\mathcal{E}}` is a rotation matrix defined as

.. math::

    \mathbf{R}_{\mathcal{E}} =
    \begin{bmatrix}
      \cos\theta_{\mathcal{E}} & -\sin\theta_{\mathcal{E}}\\
      \sin\theta_{\mathcal{E}} &  \cos\theta_{\mathcal{E}}
    \end{bmatrix}

and :math:`\mathbf{D}_{\mathcal{E}}` is the diagonal matrix defined as

.. math::

    \mathbf{D}_{\mathcal{E}} =
    \begin{bmatrix}
      1/a_{\mathcal{E}}^2 & 0\\
      0 & 1/b_{\mathcal{E}}^2 & \\
    \end{bmatrix}

Note that Equation :eq:`eq:sigma_eps` is the singular value decomposition of
:math:`\mathbf{\Sigma}_{\mathcal{E}}` if the axis radii satisfy :math:`a_{\mathcal{E}}
< b_{\mathcal{E}}`.

Using these information, ellipse :math:`\mathcal{E}` can be parameterized by the following equation:

.. math::

    (\mathbf{x}-\mathbf{x}_{\mathcal{E}})^T \mathbf{\Sigma}_{\mathcal{E}} (\mathbf{x} - \mathbf{x}_{\mathcal{E}}) = 1

Or

.. math::

    \mathbf{x}^T \mathbf{A}_{\mathcal{E}} \mathbf{x} + \mathbf{b}_{\mathcal{E}}^T \mathbf{x} + c_{\mathcal{E}} = 0

with
:math:`\mathbf{A}_{\mathcal{E}} = \mathbf{\Sigma}_{\mathcal{E}}`,
:math:`\mathbf{b}_{\mathcal{E}} = 2 \mathbf{\Sigma}_{\mathcal{E}} \mathbf{x}_{\mathcal{E}}` and
:math:`c_{\mathcal{E}} = \mathbf{x}_{\mathcal{E}}^T \mathbf{\Sigma}_{\mathcal{E}} \mathbf{x}_{\mathcal{E}} -1`.

Denoting :math:`\mathbf{x}^T = [x, y]`, ellipse :math:`\mathcal{E}` can be
defined algebraically as

.. math::
    E(x,y) = e_1 x^2 + e_2xy + e_3y^2 + e_4x + e_5y + e_6 = 0,

where :math:`\mathbf{A}_{\mathcal{E}} = \begin{bmatrix} e_1 & e_2/2 \\ e_2/2 & e_3 \end{bmatrix}`, :math:`\mathbf{b}_{\mathcal{E}}^T = [e_4, e_5]` and :math:`c_{\mathcal{E}} = e_6`.
This algebraic form is the convenient one that we will use in order to compute the intersection points of two intersecting ellipses.


Intersection Points of Two Ellipses
***********************************

In this section, we sketch the computation of the intersection points. Our
presentation slightly differs from :cite:`Eberly`. First, let
:math:`(\mathcal{E}_i)_{1 \leq i \leq 2}` be two ellipses defined as

.. math::
    :label: eq:twoellipses

    (x,y) \in \mathcal{E}_i \iff
    E_i(x,y) = e_{i1} x^2 + e_{i2} xy + e_{i3} y^2 + e_{i4} x + e_{i5} y + e_{i6} = 0

The intersection points of ellipses :math:`(\mathcal{E}_i)_{1 \leq i \leq 2}`
satisfy Equation~(\ref{eq:twoellipses}) for `i \in \{1, 2\}`, *i.e.*, the
following equation system holds for intersection points

.. math::
    :label: eq:system

    \left\{ \begin{matrix} E_1(x,y) = 0 \\ E_2(x,y) = 0 \end{matrix} \right.

Now, let us rewrite :math:`E_i(x,y)` as a quadratic polynomial in :math:`x`,
*i.e*,

.. math::

    E_i(x,y) = e_{i1} x^2
             + (e_{i2} y + e_{i4}) x
             + (e_{i3} y^2 + e_{i5} y + e_{i6}) = 0

For convenience, we define

.. math::

    p_0(y) = e_{13} y^2 + e_{15} y + e_{16} \\
    q_0(y) = e_{23} y^2 + e_{25} y + e_{26} \\
    p_1(y) = e_{12} y + e_{14} \\
    q_1(y) = e_{22} y + e_{24} \\
    p_2(y) = e_{11} \\
    q_2(y) = e_{21}

Using the notations above, we observe that `x` can be computed as follows

.. math::

    (:eq:`eq:system`) \iff &
    \left\{
      \begin{matrix}
        p_2(y) x^2 + p_1(y) x + p_0(y) = 0 \\
        q_2(y) x^2 + q_1(y) x + q_0(y) = 0
      \end{matrix}
    \right. \\
    \Longrightarrow &
    \left\{
      \begin{matrix}
        q_2(y) \times \left( p_2(y) x^2 + p_1(y) x + p_0(y) \right)= 0\times q_2(y)\\
        p_2(y) \times \left( q_2(y) x^2 + q_1(y) x + q_0(y) \right)= 0\times p_2(y)
      \end{matrix}
    \right.

Then subtracting the first equation from the second equation, we get
\begin{empheq}[box=\myeqbox]{equation} :label:{eq:xinter}
x = \frac{p_0(y)q_2(y) - p_2(y)q_0(y)}{p_1(y)q_2(y) - p_2(y)q_1(y)}.
\end{empheq}
%
Furthermore, Equation System~(\ref{eq:system}) is equivalent to the following augmented equation system
\begin{equation}
\left\{
  \begin{array}{rl}
            E_1(x,y) &= 0 \\
    x\times E_1(x,y) &= 0 \\
            E_2(x,y) &= 0 \\
    x\times E_2(x,y) &= 0 \\
  \end{array}
\right.,
\end{equation}
which is equivalent to
\begin{equation}:label:{eq:system2}
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
\end{equation}
%
We recognize a linear system in the vector `[1, x, x^2, x^3]^T`. More particularly, `[1, x, x^2, x^3]^T` is in the nullspace of `\mathbf{B}(y)`, which then must have a zero determinant. Note that all the equations systems are \emph{equivalent}, so Equation System~(\ref{eq:system}) holds if and only if the determinant of `\mathbf{B}(y)` is zero. Letting the resultant be
\begin{empheq}[box=\myeqbox]{equation}
  R = \left( p_{0}q_{2} - p_{2}q_{0} \right)^2 -
      \left( p_{0}q_{1} - p_{1}q_{0} \right)
      \left( p_{1}q_{2} - p_{2}q_{1} \right),
\end{empheq}
Equation System~(\ref{eq:system}) is equivalent to the following quartic equation in `y`.
\begin{empheq}[box=\myeqbox]{equation}:label:{eq:detBy}
  \det(\mathbf{B}(y)) = R(y) = 0,
\end{empheq}
%
This quartic equation can be solved either by SVD from the characteristic polynomial of the companion matrix. The SVD is computed either from a direct method or from Jacobi's iterative and numerically stable method. Instead we compute the roots with Ferrari's method. While it is a tedious method, it has the advantage of being direct.
Also, we experimentally observe Ferrari's method can sometimes be numerically inaccurate in particular situations, \eg, one of the ellipse is quasi-degenerate. Therefore, some tuning may be required for numerical accuracy.

Using any polynomial solver, we get the `4` roots `(y_i)_{1\leq i\leq 4}` of the quartic polynomial `R` and only keep those that are real. Finally `(x_i)_{1\leq i \leq 4}` are deduced from Equation~(\ref{eq:xinter}).








\section{Intersection Area of Two Ellipses}

Our presentation is different from \textcite{Eberly} and details are added.
%
In the rest of the section, we consider two ellipses `(\mathcal{E}_i)_{1 \leq i \leq 2}` and we respectively denote
\begin{itemize}
\item the axes of ellipse `\mathcal{E}_i` by `(a_i, b_i)`, the ellipse center by `\mathbf{x}_i`, the orientation by `\theta_i`, and the direction vectors of axis `a_i` and `b_i` by
\begin{align}
	\mathbf{u}_i &\eqdef \begin{bmatrix}  \cos(\theta_i) \\ \sin(\theta_i) \end{bmatrix} &
  \mathbf{v}_i &\eqdef \begin{bmatrix} -\sin(\theta_i) \\ \cos(\theta_i) \end{bmatrix}
\end{align}
\item the area of the elliptic portion bounded a line segment and an arc for ellipse `\mathcal{E}_i` by `B_i`,
\item the number of intersection points by `L`,
\item the intersection points by `\mathbf{p}_i` for `i \in \llbracket 1, L \rrbracket`, sorted in a counter-clockwise order, \ie
\begin{equation}:label:{eq:counterclockwise}
	\forall i \in \llbracket 1, L-1\rrbracket,\quad
		\angle\left([1,0]^T, \bp_i\right) \ < \
		\angle\left([1,0]^T, \bp_{i+1}\right)
\end{equation}
where `\angle(.,.)` denotes the angle between two vectors in the plane `\mathbb{R}^2`.
\item the polar angles of points `(\bp_i)_{1\leq i \leq L}` with respect to ellipses `\mathcal{E}_1` and `\mathcal{E}_2` by `(\phi_i)_{1\leq i \leq 2}` and `(\psi_i)_{1\leq i \leq 2}`, \ie
\begin{gather}
  \forall i \in \llbracket 1, L\rrbracket,
 	\phi_i \eqdef \angle\left(\mathbf{u}_1, \bp_i - \mathbf{x}_1\right) \\
	\forall i \in \llbracket 1, L\rrbracket,
	\psi_i \eqdef \angle\left(\mathbf{u}_2, \bp_i - \mathbf{x}_2\right)
\end{gather}
\end{itemize}


\subsection{Retrieving the polar angles}

To retrieve the polar angles, we need to place ourselves in the reference frame `(\mathbf{x}_i, \mathbf{u}_i, \mathbf{v}_i)`, where `\mathbf{x}_i` is the origin of the reference frame and `\mathbf{u}_i` and `\mathbf{v}_i` are the direction vectors determining the ellipse orientation.
Using the convenient `\mathrm{atan2}` function giving values ranging in `]-\pi,\pi]`, we have
\begin{empheq}[box=\myeqbox]{align}
  \phi_i &= \mathrm{atan2}
  \left(
    \langle \bp_i-\mathbf{x}_1, \mathbf{v}_1 \rangle,
    \langle \bp_i-\mathbf{x}_1, \mathbf{u}_1 \rangle
  \right)\\
  \psi_i &= \mathrm{atan2}
  \left(
    \langle \bp_i-\mathbf{x}_2, \mathbf{v}_2 \rangle,
    \langle \bp_i-\mathbf{x}_2, \mathbf{u}_2 \rangle
  \right)
\end{empheq}






\subsection{0 or 1 intersection point}
Either one ellipse is contained in the other or there are separated as illustrated in Figure~\ref{fig:inter01}.
%
\begin{figure}[!t]
  \centering
  \begin{tabular}{cc}
    \includegraphics[width=0.48\textwidth]{figures/test0a.png}&
    \includegraphics[width=0.48\textwidth]{figures/test0b.png}\\
    \includegraphics[width=0.48\textwidth]{figures/test1a.png}&
    \includegraphics[width=0.48\textwidth]{figures/test1b.png}
  \end{tabular}
  \caption{Cases where there is zero or one intersection point.}
  :label:{fig:inter01}
\end{figure}
%
An ellipse, say `\mathcal{E}_1`, is contained in the other `\mathcal{E}_2` if and only if its center satisfies `E_2(\mathbf{x}_1) < 0`. In that case, the area of the intersection is just the area of ellipse `\mathcal{E}_1`.
Otherwise, if there is no containment, the intersection area is zero.
In summary,
\begin{equation} :label:{eq:area01}
\area(\mathcal{E}_1 \cap \mathcal{E}_2) = \left\{
\begin{array}{ll}
\pi a_1 b_1 & \textrm{if}\ E_2(\mathbf{x}_1) < 0\\
\pi a_2 b_2 & \textrm{if}\ E_1(\mathbf{x}_2) < 0\\
0 & \textrm{otherwise}
\end{array}
\right.
\end{equation}







\subsection{2 intersection points}
We will not detail the case when Polynomial~(\ref{eq:detBy}) have 2 roots with multiplicity 2. This still corresponds to the case where there are two intersection points. But because of the root multiplicities, one ellipse is contained in the other one and then Formula~(\ref{eq:area01}) gives the correct intersection area.

Otherwise, we have to consider two cases as illustrated in Figure~\ref{fig:inter2}, which \textcite{Eberly} apparently forgot to consider.
Namely, the cases correspond to whether the center of ellipses `\mathcal{E}_1` and `\mathcal{E}_2` are on the same side or on opposite side with respect to the line `(\bp_1, \bp_2)`.

Denoting a unit normal of the line going across the intersection points `(\bp_1, \bp_2)` by `\mathbf{n}` (cf. Figure~\ref{fig:inter2}).
If the ellipse centers `\mathbf{x}_1` and `\mathbf{x}_2` are on opposite side with respect to the line `(\bp_1, \bp_2)`, \ie,
`
  \langle\mathbf{n},\mathbf{x}_1-\bp_1\rangle \times
  \langle\mathbf{n},\mathbf{x}_2-\bp_1\rangle < 0,
`
then
\begin{equation}
  \area(\mathcal{E}_1 \cap \mathcal{E}_2) =
	B_1(\phi_1, \phi_2) + B_2(\psi_1, \psi_2)
\end{equation}
%
If they are on the same side with respect to the line `(\bp_1, \bp_2)`, \ie,
`
	\langle\mathbf{n},\mathbf{x}_1-\bp_1\rangle \times
  \langle\mathbf{n},\mathbf{x}_2-\bp_1\rangle > 0
`, then
\begin{equation}:label:{eq:inter2b}
  \area(\mathcal{E}_1 \cap \mathcal{E}_2) = \left\{
  \begin{array}{ll}
  	\displaystyle
  	\left( \pi a_1 b_1 - B_1(\phi_1, \phi_2) \right) +
  	B_2(\psi_1, \psi_2) &
  	\textrm{if}\
  	|\langle\mathbf{n},\mathbf{x}_1-\bp_1\rangle| \leq
  	|\langle\mathbf{n},\mathbf{x}_2-\bp_1\rangle| \\
  	%
  	\\
  	%
  	\displaystyle
  	B_1(\phi_1, \phi_2) +
  	\left( \pi a_2 b_2 - B_2(\psi_1, \psi_2) \right) &
  	\textrm{otherwise}.
  \end{array}
  \right.
\end{equation}
Note that the condition
`
  |\langle\mathbf{n},\mathbf{x}_1-\bp_1\rangle| \leq
  |\langle\mathbf{n},\mathbf{x}_2-\bp_1\rangle|
`
in Equation~(\ref{eq:inter2b}) just expresses the fact that the distance of ellipse center `\mathbf{x}_1` to the line `(\bp_1, \bp_2)` is smaller than the distance of ellipse center `\mathbf{x}_2` to the line `(\bp_1, \bp_2)`.

\begin{figure}[!t]
  \centering
  \begin{tabular}{cc}
    \includegraphics[width=0.48\textwidth]{figures/inter2a.pdf}&
    \includegraphics[width=0.48\textwidth]{figures/inter2b.pdf}
  \end{tabular}
  \caption{Cases where there are two intersection points.}
  :label:{fig:inter2}
\end{figure}






\subsection{3 and 4 intersection points}
\begin{figure}[!t]
  \centering
  \begin{tabular}{cc}
    \includegraphics[width=0.45\textwidth]{figures/inter3.pdf}&
    \includegraphics[width=0.45\textwidth]{figures/inter4.pdf}
  \end{tabular}
  \caption{Cases where there are three of four intersection points.}
  :label:{fig:inter34}
\end{figure}

These cases are rather easy to handle. Indeed, we see geometrically from Figure~\ref{fig:inter34},
\begin{equation}
	\area(\mathcal{E}_1 \cap \mathcal{E}_2) =
	\sum_{i=1}^{L}
	  \underbrace{\min \left(
	    B_1(\phi_i, \phi_{i+1}),
	    B_2(\psi_i, \psi_{i+1})
	  \right)}_{\textrm{smallest of elliptic portion area}} +
	\underbrace{\frac{1}{2} \sum_{i=1}^{L} \left|
		\det\left(\bp_i, \bp_{i+1}\right)
	\right|}_{\textrm{area of polygon}\ (\bp_1, \bp_2, \dots, \bp_L)}
\end{equation}
with `\phi_{L+1} = \phi_1`, `\psi_{L+1} = \psi_1` and `\bp_{L+1} = \bp_1`.


.. rubric:: References

.. bibliography:: phd.bib
    :encoding: latin
    :cited:

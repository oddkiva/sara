Improvements for the Match Propagation
======================================

Years later, it is always nice to see the match propagation method in action.
Still the method suffers from an algorithmic efficiency problem during the
region growing process because of the combinatorial search to estimate as
accurately the local affine transformation.

The match propagation can be seen as a kind exploration/exploitation tradeoff
policy to discover the space of inliers and outliers without using a *global*
geometric model which RANSAC and variants do to filter correspondences. Our goal
is to be able to multiple model fitting and hence this favors the region growing
approach because it seems more intuitively.


The match propagation can alleviate further the need to enumerate high-order
relationships used in hypergraph matching approaches.

Basically there is a combinatorial search for a good enough local affine
transformation that explains the neighborhood of a match. We can do much better
by using kernel regression to estimate **densely** the local affine
transformation.
The more matches there are in the vicinity of a match, the more reliable the
local affine estimation obtained by density estimation.

1. We also want to quantify uncertainty about the local affine estimation. In
   which case a Bayesian approach is well suited and we can use for example
   Gaussian Processes (GP). GP suffer from a scalability problem on big data. In
   our case, this is not a problem since we want to apply it **locally** thanks
   to the exploration schedule that the match propagation offers.
2. We still need to identify outlier and the intuition is that
   locally the number of outliers is actually really low if we are to propagate
   from a seed match. We can use L1-minimization to identify outliers.
3. We want to minimize the symmetric transfer error. The global optimization is
   not obvious and one way to do it is to do alternate minimization.

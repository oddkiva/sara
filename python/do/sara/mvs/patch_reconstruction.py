"""Patch-based reconstruction."""


# We assume we know the essential matrix, the internal camera matrix.
#
# Detect DoG and Harris response corners.
# In each 32x32 image block, find the local maxima.
#
# 1. Use a feature detectors to compute initial feature points.
# 2. Can use dense SIFT descriptor to prefilter all the possible matches.
#    Otherwise use a deeply learnt feature descriptor.
# 3. Match each feature (x, x') if their descriptor are close enough
# 4. Find correspondences where the essential matrix if |x'T E x| less than two
#    pixels.
# 5. By increasing distance, obtain c(p) and n(p) by triangulation using
#    Lindstrom approach
# 6. Find the list of images where p is visible.
# 7. Once a patch is reconstructed, don't reuse the features in the cell which
#    the patch belongs to.


# Expansion on the image grid. (Region growing).
#   Straightforward.

# Visibility score.


# Ingredients:
def patch(p):
    """3D x-axis-aligned rectangle.

    its projection on the image must be a mu x mu square.
    mu = 5 or 7.
    """
    c = np.zeros(3)
    n = np.zeros(3)
    pass

def photometric_discrepany(p, V, R):
    """
    h is the photometric discrepancy score.
    V set of images where p is visible.
    V_best is define by filtering by h(p, I, R(p)) <= alpha
    R reference image for p.
    """
    pass

def optimize_patch(p):
    """
    minimize g*(p) patch P=[c, n]
    c(p) is constrained on some ray.
    n(p) optimized by conjugate gradient.
    """
    c_best = None
    n_best = None

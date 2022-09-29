from glue.core.roi import Projected3dROI
from glue.utils.array import iterate_chunks

import numpy as np
from scipy.stats import gaussian_kde


class KDEROI(Projected3dROI):
    """
    A region of interest for our KDE.
    """

    def __init__(self, x, y, projection_matrix=None):
        super(KDEROI, self).__init__()
        self.x = x
        self.y = y
        self.projection_matrix = projection_matrix

    def defined(self):
        return True

    def contains3d(self, x, y, z):
        x = np.copy(np.asarray(x))
        y = np.copy(np.asarray(y))
        z = np.copy(np.asarray(z))

        # This seemed like a good idea
        # but results in a mask that isn't the same shape as the data
        # (which i very bad!)
        # indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        # x = x[indices]
        # y = y[indices]
        # z = z[indices]

        indices = np.logical_not(np.isfinite(x) & np.isfinite(y) & np.isfinite(z))
        x[indices] = 0
        y[indices] = 0
        z[indices] = 0

        min_dist = np.inf
        closest_pt = None

        for slices in iterate_chunks(x.shape, n_max=10000):
            # Work in homogeneous coordinates so we can support perspective
            # projections as well
            x_sub, y_sub, z_sub = x[slices], y[slices], z[slices]
            vertices = np.array([x_sub, y_sub, z_sub, np.ones(x_sub.shape)])

            # The following returns homogeneous screen coordinates
            screen_h = np.tensordot(self.projection_matrix,
                                    vertices, axes=(1, 0))

            # Convert to screen coordinates, as we don't care about z
            screen_x, screen_y = screen_h[:2] / screen_h[3]

            d = np.hypot(self.x - screen_x, self.y - screen_y)
            idx = np.unravel_index(np.argmin(d), shape=d.shape)
            dist = d[idx]
            if dist < min_dist:
                min_dist = dist
                closest_pt = (x_sub[idx], y_sub[idx], z_sub[idx])

        kde = gaussian_kde([x, y, z])

        threshold = 0.0001
        overall = kde([x, y, z])
        at_pt = kde(closest_pt)
        diff = overall - at_pt
        mask = np.abs(diff) < threshold
        mask[indices] = False
        return mask





from glue.core.roi import Projected3dROI
from glue.utils.array import iterate_chunks

import numpy as np
from scipy.stats import gaussian_kde


class KDEROI(Projected3dROI):
    """
    A region of interest for our KDE.
    """

    def __init__(self, x, y, data, projection_matrix=None):
        super(KDEROI, self).__init__()
        self.x = x
        self.y = y
        self.projection_matrix = projection_matrix

        # Create the KDE
        x, y, z = data
        self.indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        xm, ym, zm = x[self.indices], y[self.indices], z[self.indices]
        self.kde = gaussian_kde([xm, ym, zm])
        self.overall = self.kde([xm, ym, zm])

    def defined(self):
        return True

    def set_xy(self, x, y):
        self.x = x
        self.y = y

    def contains3d(self, x, y, z):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        # This seemed like a good idea
        # but results in a mask that isn't the same shape as the data
        # (which is very bad!)
        # indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        # x = x[indices]
        # y = y[indices]
        # z = z[indices]

        mask = np.zeros(x.shape, dtype=bool)

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
            idx = np.unravel_index(np.nanargmin(d), shape=d.shape)
            dist = d[idx]
            if dist < min_dist:
                min_dist = dist
                closest_pt = (x_sub[idx], y_sub[idx], z_sub[idx])

        threshold = 0.000001
        # threshold = 0.01
        at_pt = self.kde(closest_pt)
        diff = self.overall - at_pt
        mask[self.indices] = np.abs(diff) < threshold
        return mask





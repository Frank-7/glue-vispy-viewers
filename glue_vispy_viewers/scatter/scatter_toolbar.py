"""
This is for 3D selection in Glue 3d scatter plot viewer.
"""
from os.path import dirname, join, realpath

import matplotlib.cm
import numpy as np
from pandas import DataFrame
from sklearn.cluster import DBSCAN, OPTICS

from glue.config import viewer_tool
from glue.core import Subset
from glue.core.roi import Roi, Projected3dROI
from glue.viewers.common.tool import Tool
from glue.core.util import colorize_subsets, facet_subsets

from .layer_artist import ScatterLayerArtist
from ..common.selection_tools import VispyMouseMode


class NearestNeighborROI(Roi):

    def __init__(self, x=None, y=None, max_radius=None):
        self.x = x
        self.y = y
        self.max_radius = max_radius

    def contains(self, x, y):
        mask = np.zeros(x.shape, bool)
        d = np.hypot(x - self.x, y - self.y)
        index = np.argmin(d)
        if d[index] < self.max_radius:
            mask[index] = True
        return mask

    def move_to(self, x, y):
        self.x = x
        self.y = y

    def defined(self):
        try:
            return np.isfinite([self.x, self.y]).all()
        except TypeError:
            return False

    def center(self):
        return self.x, self.y

    def reset(self):
        self.x = self.y = self.max_radius = None

    def __gluestate__(self, context):
        return dict(x=float(self.x), y=float(self.y),
                    max_radius=float(self.max_radius))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(rec['x'], rec['y'], rec['max_radius'])


@viewer_tool
class PointSelectionMode(VispyMouseMode):

    icon = 'glue_point'
    tool_id = 'scatter3d:point'
    action_text = 'Select points using a point selection'

    def press(self, event):
        if event.button == 1:
            roi = Projected3dROI(roi_2d=NearestNeighborROI(event.pos[0], event.pos[1],
                                                           max_radius=5),
                                 projection_matrix=self.projection_matrix)
            self.apply_roi(roi)

    def release(self, event):
        pass

    def move(self, event):
        pass


@viewer_tool
class AutoSegmentMode(Tool):
    icon = 'glue_rainbow'  # TODO: Figure out how to add an icon
    tool_id = 'scatter3d:autoseg'
    action_text = 'Automatically segment points'

    def __init__(self, viewer):
        super(AutoSegmentMode, self).__init__(viewer)

    def deactivate(self):
        pass

    def activate(self):

        # Get the values of the currently active layer artist - we
        # specifically pick the layer artist that is selected in the layer
        # artist view in the left since we have to pick one.
        layer_artist = self.viewer._view.layer_list.current_artist()

        # If the layer artist is for a Subset not Data, pick the first Data
        # one instead (where the layer artist is a 3d scatter artist)
        if isinstance(layer_artist.layer, Subset):
            for layer_artist in self.iter_data_layer_artists():
                if isinstance(layer_artist, ScatterLayerArtist):
                    break
            else:
                return

        data = layer_artist.layer
        viewer_state = self.viewer.state
        input_data = DataFrame({
            'x': data[viewer_state.x_att],
            'y': data[viewer_state.y_att],
            'z': data[viewer_state.z_att]
        })

        method = DBSCAN  # Later, make this an option
        params = dict(eps=2.5, min_samples=2)
        model = method(**params)
        model.fit(input_data)
        subset_count = np.max(model.labels_) + 1

        data.add_component(model.labels_, "_autoseg_labels")
        subsets = facet_subsets(self.viewer._data, cid=data.id["_autoseg_labels"], steps=subset_count)
        colorize_subsets(subsets, matplotlib.cm.get_cmap("plasma"))

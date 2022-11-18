"""
This is for 3D selection in Glue 3d scatter plot viewer.
"""
import matplotlib.cm
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS

from glue.config import viewer_tool
from glue.core import Data, Subset
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


class BaseAutoFacetTool(Tool):
    facet_component = "_facet_labels"

    def _input_data(self, data):
        raise NotImplementedError()

    def _facets(self, input_data, **kwargs):
        raise NotImplementedError()

    def activate(self):

        # Get the values of the currently active layer artist - we
        # specifically pick the layer artist that is selected in the layer
        # artist view in the left since we have to pick one.
        layer_artist = self.viewer._view.layer_list.current_artist()

        # If the layer artist is for a Subset not Data, pick the first Data
        # one instead (where the layer artist is a 3d scatter artist)
        if isinstance(layer_artist.layer, Subset):
            for layer_artist in self.viewer._layer_artist_container:
                if isinstance(layer_artist.layer, Data) and \
                   isinstance(layer_artist, ScatterLayerArtist):
                    break
            else:
                return

        data = layer_artist.layer
        labels = self._facets(data)
        subset_count = np.max(labels) + 1
        data.add_component(labels, self.facet_component)
        subsets = facet_subsets(self.viewer._data, cid=data.id[self.facet_component], steps=subset_count)
        colorize_subsets(subsets, matplotlib.cm.get_cmap("plasma"))


class SKLAutoFacetTool(BaseAutoFacetTool):

    def __init__(self, viewer, model_cls, dialog_cls):
        super(SKLAutoFacetTool, self).__init__(viewer)
        self._model_cls = model_cls
        self._dialog_cls = dialog_cls

    def _input_data(self, data):
        viewer_state = self.viewer.state
        input_data = np.array([
            data[viewer_state.x_att],
            data[viewer_state.y_att],
            data[viewer_state.z_att]]
        ).transpose()
        return input_data

    def _get_params(self):
        params = {}
        dialog = self._dialog_cls(params)
        dialog.exec_()
        return params

    def _facets(self, data, **kwargs):
        input_data = self._input_data(data)
        params = self._get_params()
        model = self._model_cls(**params)
        model.fit(input_data)
        return model.labels_


@viewer_tool
class DBSCANAutoFacetTool(SKLAutoFacetTool):
    icon = 'glue_rainbow'  # TODO: Figure out how to add an icon
    tool_id = 'scatter3d:facet_dbscan'
    action_text = 'Automatically facet a data layer'
    
    def __init__(self, viewer):
        super(DBSCANAutoFacetTool, self).__init__(viewer, DBSCAN, None)

    # Dummy for now
    def _get_params(self):
        return {
            "eps": 2.5,
            "min_samples": 2
        }

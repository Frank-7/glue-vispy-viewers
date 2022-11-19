from dataclasses import dataclass
import os

import matplotlib.cm
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS

from glue.config import viewer_tool
from glue.core.roi import Roi, Projected3dROI
from glue.viewers.common.tool import Tool
from glue.core.util import colorize_subsets, facet_subsets

from .segmentation_tool_dialog import SegmentationToolDialog
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
    facet_component = '_facet_labels'

    params = {}
    options = {}

    def _input_data(self, data):
        raise NotImplementedError()

    def _facets(self, data, params):
        raise NotImplementedError()

    def _get_info(self):
        raise NotImplementedError()

    def activate(self):

        result = self._get_info()
        if not result:
            return
        data = self.options["data"]
        parameter_values = {k: v.value for k, v in self.params.items()}
        labels = self._facets(data, parameter_values)
        subset_count = np.max(labels) + 1
        components = [x.label for x in data.components]
        if self.facet_component in components:
            data.update_components({data.id[self.facet_component]: labels})
        else:
            data.add_component(labels, self.facet_component)
        subsets = facet_subsets(self.viewer._data, cid=data.id[self.facet_component], steps=subset_count)
        colorize_subsets(subsets, self.options["cmap"])


@dataclass
class SegmentationParameterInfo:
    name: str
    type: type
    value: int | float | bool


class SKLAutoFacetTool(BaseAutoFacetTool):
    options = {'cmap': matplotlib.cm.get_cmap("gray")}

    def __init__(self, viewer, model_cls):
        super(SKLAutoFacetTool, self).__init__(viewer)
        self._model_cls = model_cls

    def _input_data(self, data):
        viewer_state = self.viewer.state
        input_data = np.array([
            data[viewer_state.x_att],
            data[viewer_state.y_att],
            data[viewer_state.z_att]]
        ).transpose()
        return input_data

    def _get_info(self):
        dialog = SegmentationToolDialog(self.params, self.options, self.viewer._data)
        return dialog.exec_()

    def _facets(self, data, params):
        input_data = self._input_data(data)
        model = self._model_cls(**params)
        model.fit(input_data)
        return model.labels_


@viewer_tool
class DBSCANAutoFacetTool(SKLAutoFacetTool):
    icon = os.path.abspath(os.path.join(os.path.dirname(__file__), 'auto_seg_cloud_points.png'))
    tool_id = 'scatter3d:facet_dbscan'
    action_text = 'Automatically facet a data layer'

    params = {
        'eps': SegmentationParameterInfo(name='Epsilon', type=float, value=2.5),
        'min_samples': SegmentationParameterInfo(name='Min Samples', type=int, value=2)
    }
    
    def __init__(self, viewer):
        super(DBSCANAutoFacetTool, self).__init__(viewer, DBSCAN)

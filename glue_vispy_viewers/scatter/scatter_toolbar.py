from dataclasses import dataclass
import os

import matplotlib.cm
import numpy as np
from qtpy.QtWidgets import QAction
from sklearn.cluster import DBSCAN, OPTICS

from glue.config import viewer_tool
from glue.core.roi import Roi, Projected3dROI
from glue.viewers.common.tool import DropdownTool
from glue.core.util import colorize_subsets, facet_subsets
from glue.utils import nonpartial

from .autofacet_tool_dialog import SegmentationToolDialog
from ..common.selection_tools import VispyMouseMode

AUTOFACET_ICON = os.path.abspath(os.path.join(os.path.dirname(__file__), 'auto_seg_cloud_points.png'))


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


@dataclass
class AutofacetParameter:
    name: str
    value: int | float | bool


class BaseAutoFaceter:

    def facets(self, data, viewer_state, params):
        raise NotImplementedError()


class SKLAutoFaceter(BaseAutoFaceter):

    def __init__(self, model_cls):
        super(SKLAutoFaceter, self).__init__()
        self._model_cls = model_cls

    def input_data(self, data, viewer_state):
        input_data = np.array([
            data[viewer_state.x_att],
            data[viewer_state.y_att],
            data[viewer_state.z_att]]
        ).transpose()
        return input_data

    def facets(self, data, viewer_state, params):
        input_data = self.input_data(data, viewer_state)
        model = self._model_cls(**params)
        model.fit(input_data)
        return model.labels_


class OPTICSAutoFaceter(SKLAutoFaceter):
    name = 'OPTICS'

    params = {
        'min_samples': AutofacetParameter(name='Min Samples', value=7)
    }

    def __init__(self):
        super(OPTICSAutoFaceter, self).__init__(OPTICS)


class DBSCANAutoFaceter(SKLAutoFaceter):
    name = 'DBSCAN'

    params = {
        'eps': AutofacetParameter(name='Epsilon', value=2.5),
        'min_samples': AutofacetParameter(name='Min Samples', value=2)
    }

    def __init__(self):
        super(DBSCANAutoFaceter, self).__init__(DBSCAN)


@viewer_tool
class AutoFacetTool(DropdownTool):
    icon = AUTOFACET_ICON
    tool_id = 'scatter3d:autofacet'

    options = {
        'cmap': matplotlib.cm.get_cmap("gray"),
        'component': '_facet_labels'
    }
    faceters = [DBSCANAutoFaceter(),OPTICSAutoFaceter()]

    def get_info(self, faceter):
        dialog = SegmentationToolDialog(faceter, self.options,
                                        data_collection=self.viewer._data)
        return dialog.exec_()

    def menu_actions(self):
        actions = []
        for faceter in self.faceters:
            action = QAction(faceter.name)
            action.triggered.connect(nonpartial(self.facet, faceter))
            actions.append(action)
        return actions

    def facet(self, faceter):

        result = self.get_info(faceter)
        if not result:
            return
        data = self.options["data"]
        parameter_values = {k: v.value for k, v in faceter.params.items()}
        labels = faceter.facets(data, self.viewer.state, parameter_values)
        subset_count = np.max(labels) + 1
        components = [x.label for x in data.components]
        component_name = self.options["component"]
        if component_name in components:
            data.update_components({data.id[component_name]: labels})
        else:
            data.add_component(labels, component_name)
        subsets = facet_subsets(self.viewer._data, cid=data.id[component_name], steps=subset_count)
        cmap = self.options["cmap"]
        if self.options["reverse_cmap"]:
            cmap = cmap.reversed()
        colorize_subsets(subsets, cmap)


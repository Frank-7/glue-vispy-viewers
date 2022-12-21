"""
This is for 3D selection in Glue 3d scatter plot viewer.
"""

import numpy as np

from echo import add_callback
from glue.config import viewer_tool
from glue.core.hub import HubListener
from glue.core.message import NumericalDataChangedMessage
from glue.core.roi import Roi, Projected3dROI
from .kde_roi import KDEROI

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
class PointSelectionMode(VispyMouseMode, HubListener):

    icon = 'auto_seg_cloud_points'
    tool_id = 'scatter3d:auto_seg_cloud_point'
    action_text = 'Auto select cloud points using segmentation'

    def __init__(self, viewer):
        super(PointSelectionMode, self).__init__(viewer)
        self.roi = None
        self.stale = True
        add_callback(self.viewer.state, 'x_att', self.mark_stale)
        add_callback(self.viewer.state, 'y_att', self.mark_stale)
        add_callback(self.viewer.state, 'z_att', self.mark_stale)

    def _ndc_message_filter(self, msg):
        data = msg.sender
        have_data = data in [layer.layer for layer in self.viewer.layers]
        if not have_data:
            return False

        viewer_state = self.viewer.state
        have_attribute = msg.attribute in [viewer_state.x_att, viewer_state.y_att, viewer_state.z_att]
        return have_attribute

    def register_to_hub(self, hub):
        self.hub = hub
        self.hub.subscribe(self, NumericalDataChangedMessage, filter=self._ndc_message_filter, handler=self.mark_stale)

    def unregister(self, hub):
        self.hub.unsubscribe(self, NumericalDataChangedMessage)
        self.hub = None

    def notify(self, message):
        self.hub.broadcast(message)

    def mark_stale(self, _arg=None):
        self.stale = True

    def press(self, event):
        if event.button == 1:
            if self.stale:
                # For now, we're only thinking about one layer
                # For the future - think about how to handle more
                layer = self.viewer.layers[0].layer
                data = [layer[self.viewer.state.x_att], layer[self.viewer.state.y_att], layer[self.viewer.state.z_att]]
                self.roi = KDEROI(event.pos[0], event.pos[1], data=data, projection_matrix=self.projection_matrix)
                self.stale = False
            else:
                self.roi.set_xy(event.pos[0], event.pos[1])
            self.apply_roi(self.roi)

    def release(self, event):
        pass

    def move(self, event):
        pass

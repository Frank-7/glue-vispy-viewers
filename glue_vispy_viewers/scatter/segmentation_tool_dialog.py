import os
from qtpy.QtWidgets import QDialog

from glue.core.state_objects import State
from glue.core.data_combo_helper import DataCollectionComboHelper, ComboHelper
from glue.external.echo import CallbackProperty, SelectionCallbackProperty
from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui


class SegmentationToolState(State):

    data = SelectionCallbackProperty()
    methods = SelectionCallbackProperty()
    cmap = CallbackProperty()

    def __init__(self, data_collection=None):
        super(SegmentationToolState, self).__init__()

        self.data_helper = DataCollectionComboHelper(self, 'data', data_collection)


class SegmentationToolDialog(QDialog):

    def __init__(self, tool, data_collection=None, parent=None):

        super(SegmentationToolDialog, self).__init__(parent=parent)

        self.state = SegmentationToolState(data_collection=data_collection)

        self.ui = load_ui('save_hover.ui', self,
                          directory=os.path.dirname(__file__))

        self._connections = autoconnect_callbacks_to_qt(self.state, self.ui)

        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_cancel.clicked.connect(self.reject)

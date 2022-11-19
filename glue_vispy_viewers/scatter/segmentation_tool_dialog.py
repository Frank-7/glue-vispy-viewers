import os
from qtpy.QtWidgets import QDialog
from matplotlib.cm import get_cmap

from glue.core.state_objects import State
from glue.core.data_combo_helper import DataCollectionComboHelper
from glue.external.echo import CallbackProperty, SelectionCallbackProperty
from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui

# https://stackoverflow.com/questions/15829782/how-to-restrict-user-input-in-qlineedit-in-pyqt
# https://www.pythonguis.com/faq/looking-for-app-recommendations/

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QLineEdit, QHBoxLayout, QLabel
from qtpy.QtGui import QIntValidator, QDoubleValidator, QPalette


class SegmentationDialogState(State):

    data = SelectionCallbackProperty()
    cmap = CallbackProperty(get_cmap("gray"))

    def __init__(self, data_collection=None):
        super(SegmentationDialogState, self).__init__()
        self.data_helper = DataCollectionComboHelper(self, 'data', data_collection)


class SegmentationToolDialog(QDialog):


    def __init__(self, params, options, data_collection=None, parent=None):

        super(SegmentationToolDialog, self).__init__(parent=parent)

        self.state = SegmentationDialogState(data_collection=data_collection)

        self.ui = load_ui('segmentation_tool.ui', self,
                          directory=os.path.dirname(__file__))

        self._connections = autoconnect_callbacks_to_qt(self.state, self.ui)
        if "cmap" in options:
            self.state.cmap = options["cmap"]

        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_cancel.clicked.connect(self.reject)
        palette = QPalette()
        palette.setColor(QPalette.WindowText, Qt.red)
        self.ui.error_label.setPalette(palette)
        self.options = options
        self.params = params
        self.widgets = self._populate_form(params)

    @staticmethod
    def validator(t):
        if t == int:
            return QIntValidator()
        elif t == float:
            return QDoubleValidator()

    def _populate_form(self, params):

        widgets = {}

        for key, info in params.items():
            t = info.type
            v = info.value
            if t == bool:
                widget = QCheckBox()
                widget.setChecked(v)
            else:
                widget = QLineEdit()
                widget.setValidator(self.validator(t))
                widget.setText(str(v))
            widgets[key] = widget

            title = info.name
            row = QHBoxLayout()
            label = QLabel("{0}:".format(title))
            row.addWidget(label)
            row.addWidget(widget)
            self.ui.form_layout.addRow(row)

        return widgets

    def set_error_message(self, text):
        self.ui.error_label.setText(text)

    def accept(self):
        for key, widget in self.widgets.items():
            try:
                if isinstance(widget, QLineEdit):
                    validator = widget.validator()
                    text = widget.text()
                    if isinstance(validator, QIntValidator):
                        self.params[key].value = int(text)
                    else:
                        self.params[key].value = float(text)
                elif isinstance(widget, QCheckBox):
                    self.params[key].value = widget.isChecked()
            except ValueError:
                error_message = "Your value for the parameter {0} is not valid".format(self.schema[key].name)
                self.set_error_message(error_message)
                return

        cmap = self.state.cmap
        if cmap is None:
            self.set_error_message("You must select a colormap")
            return

        self.options.update(self.state.as_dict())

        super(SegmentationToolDialog, self).accept()
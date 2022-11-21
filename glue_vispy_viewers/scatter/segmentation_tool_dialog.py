import os
from qtpy.QtWidgets import QDialog
from matplotlib.cm import get_cmap

from echo import add_callback
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
    component = CallbackProperty()

    def __init__(self, data_collection=None):
        super(SegmentationDialogState, self).__init__()
        self.data_helper = DataCollectionComboHelper(self, 'data', data_collection)


class SegmentationToolDialog(QDialog):

    def __init__(self, faceter, options, data_collection=None, parent=None):

        super(SegmentationToolDialog, self).__init__(parent=parent)

        self.state = SegmentationDialogState(data_collection=data_collection)

        self.ui = load_ui('segmentation_tool.ui', self,
                          directory=os.path.dirname(__file__))
        add_callback(self.state, 'component', self._component_warn)
        self.ui.component_warning_label.hide()
        self._component_warn(self.state.component)

        self._connections = autoconnect_callbacks_to_qt(self.state, self.ui)
        self.state.update_from_dict(options)

        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_cancel.clicked.connect(self.reject)

        palette = QPalette()
        palette.setColor(QPalette.WindowText, Qt.red)
        self.ui.error_label.setPalette(palette)
        self.ui.component_warning_label.setPalette(palette)

        self.options = options
        self.params = faceter.params
        self.widgets = self._populate_form(self.params)
        self.ui.setWindowTitle(f"{faceter.name} Facet")
        self.ui.adjustSize()

    @staticmethod
    def validator(t):
        if t is int:
            return QIntValidator()
        elif t is float:
            return QDoubleValidator()

    def _component_warn(self, component):
        components = [c.label for c in self.state.data.components]
        visible = self.ui.component_warning_label.isVisible()
        if (component in components) ^ visible:
            if visible:
                self.ui.component_warning_label.hide()
            else:
                self.ui.component_warning_label.show()
            self.ui.adjustSize()

    def _populate_form(self, params):

        widgets = {}

        for key, info in params.items():
            t = type(info.value)
            if t is bool:
                widget = QCheckBox()
                widget.setChecked(info.value)
            else:
                widget = QLineEdit()
                widget.setValidator(self.validator(t))
                widget.setText(str(info.value))
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

        if self.state.cmap is None:
            self.set_error_message("You must select a colormap")
            return

        if not self.state.component:
            self.set_error_message("You must enter a name for the facet labels component")
            return

        self.options.update(self.state.as_dict())

        super(SegmentationToolDialog, self).accept()
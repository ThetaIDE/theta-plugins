"""Slurm's optional Qt settings contribution."""
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QMessageBox, QPushButton, QWidget
from src.app.api.workspace import WorkspacePaths
from src.app.api.desktop import label
from .commands import bootstrap_command, shortcuts, submission_command

class SiteSetup(QWidget):
    def __init__(self, parent=None, *, paths=None):
        super().__init__(parent)
        from src.app.api.desktop import ComboBox, label
        from src.app.api.sites import profiles
        self.paths = paths or WorkspacePaths.for_process()
        layout = QVBoxLayout(self)
        layout.addWidget(label('Cluster sites', 'cardTitle'))
        self.site = ComboBox()
        self.site.setEditable(False)
        self.site.addItems(list(profiles(self.paths)))
        layout.addWidget(self.site)
        layout.addWidget(label('Experiment to upload first', 'muted'))
        self.experiment = ComboBox()
        self.experiment.setEditable(False)
        layout.addWidget(self.experiment)
        for title, action in [('Copy first-run upload and package setup', self.copy_setup),
                              ('Create project SSH / push / sync shortcuts', self.make_shortcuts),
                              ('Open editable site configuration', self.open_config)]:
            button = QPushButton(title); button.clicked.connect(action); layout.addWidget(button)
        self.status = label('Address, SSH and transfer settings belong to the selected site component in Components.', 'muted')
        self.status.setWordWrap(True); layout.addWidget(self.status)

    def showEvent(self, event):
        from src.app.api.sites import profiles
        selected = self.site.currentText()
        self.site.blockSignals(True)
        self.site.clear()
        self.site.addItems(list(profiles(self.paths)))
        index = self.site.findText(selected)
        if index >= 0: self.site.setCurrentIndex(index)
        self.site.blockSignals(False)
        chosen = self.experiment.currentText()
        self.experiment.clear()
        self.experiment.addItems(experiments(self.paths))
        index = self.experiment.findText(chosen)
        if index >= 0: self.experiment.setCurrentIndex(index)
        super().showEvent(event)

    def copy_setup(self):
        try: command = bootstrap_command(self.paths.root, self.site.currentText(), self.experiment.currentText())
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, 'Site setup', str(exc)); return
        QApplication.clipboard().setText(command)
        self.status.setText('Setup script copied. Run it in a local terminal; review cluster module and package requirements first.')

    def make_shortcuts(self):
        try: paths = shortcuts(self.paths.root, self.site.currentText())
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, 'Site shortcuts', str(exc)); return
        self.status.setText('\n'.join(paths))

    def open_config(self):
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        from src.app.api.config import ConfigurationFiles
        component = self.site.currentText().split('/')[0]
        path = ConfigurationFiles(self.paths).create_override(f'site/{component}/_defaults.yaml')
        if path.is_file(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def experiments(paths):
    """Experiment selectors of the workspace; ``_``-prefixed files are shared bases."""
    from pathlib import PurePosixPath
    names = paths.config_resolver().selectors('experiment')
    return [str(PurePosixPath(name).relative_to('experiment').with_suffix('')) for name in names
            if not PurePosixPath(name).name.startswith('_')]


def create_settings(context):
    return SiteSetup(context.get('parent'), paths=context['workspace'])

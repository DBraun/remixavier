# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

"""
GUI App for running remixavier
"""

# <codecell>

import sys
import os

import numpy as np
import librosa
import soundfile as sf

from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QProgressBar,
    QPushButton,
    QFileDialog,
    QAction,
    QStatusBar,
)
from PyQt5.QtWidgets import QLabel

import estimate

# <codecell>


class Remixavier(QThread):
    """
    Class which runs the subtraction algorithm, in a thread.
    """

    percentDoneSignal = pyqtSignal(int)
    doneSignal = pyqtSignal(bool)
    statusSignal = pyqtSignal(str)

    def __init__(self):
        """
        Get ready to run - initialize thread and GUI elements
        """
        super(Remixavier, self).__init__()
        self.mix_file = None
        self.source_file = None
        self.wiener_threshold = None
        self.max_skew = .02

    def loadNewValues(self, mix_file, source_file, wiener_threshold, max_skew):
        """
        Load in a new parameter setting from the GUI

        Input:
            mix_file - path to mixure .wav file
            source_file - path to source .wav file
            wiener_threshold - threshold for wiener filtering, in dB
            max_skew - maximum allowed skew in alignment
        """
        self.mix_file = mix_file
        self.source_file = source_file
        self.wiener_threshold = wiener_threshold
        self.max_skew = max_skew

    @staticmethod
    def make_2d(audio: np.ndarray):
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        return audio

    def run(self):

        # Initialize signals
        self.percentDoneSignal.emit(0)
        percent_scale = 1000.0 / 5
        self.doneSignal.emit(0)
        self.statusSignal.emit("")
        # Load in audio data
        self.statusSignal.emit("Loading {}".format(os.path.split(self.mix_file)[1]))
        mix, self.fs = librosa.load(self.mix_file, mono=False, sr=None)
        self.percentDoneSignal.emit(1 * percent_scale)
        self.statusSignal.emit("Loading {}".format(os.path.split(self.source_file)[1]))
        source, self.fs = librosa.load(self.source_file, mono=False, sr=self.fs)
        mix = self.make_2d(mix)
        source = self.make_2d(source)
        self.percentDoneSignal.emit(2 * percent_scale)
        # Fix any gross timing offset
        self.statusSignal.emit("Aligning...")
        mix, source = estimate.align(mix, source, self.fs, max_skew=self.max_skew)
        self.percentDoneSignal.emit(3 * percent_scale)
        self.statusSignal.emit("Subtracting...")
        source = estimate.reverse_channel(mix, source)
        mix, source = estimate.pad(mix, source)
        self.percentDoneSignal.emit(4 * percent_scale)
        self.statusSignal.emit("Enhancing...")
        self.subtracted = estimate.wiener_enhance(
            mix - source, source, self.wiener_threshold
        )
        self.subtracted = mix - source
        self.percentDoneSignal.emit(5 * percent_scale)
        self.doneSignal.emit(1)


# <codecell>


# The GUI app
class AppForm(QMainWindow):

    # Initialize
    def __init__(self, parent=None):
        # Initialize window
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Remixavier")

        # Create GUI elements
        self.createMenu()
        self.createMainFrame()
        self.createStatusBar()

        # Instance of remixaiver
        self.remixavierInstance = Remixavier()
        self.remixavierInstance.percentDoneSignal.connect(self.updateProgressBar)
        self.remixavierInstance.doneSignal.connect(self.remixavierFinished)
        self.remixavierInstance.statusSignal.connect(self.updateStatusLabel)

    def updateProgressBar(self, percent):
        self.progressBar.setValue(percent)

    def remixavierFinished(self, finished):
        if finished:
            sf.write(
                self.output_file,
                self.remixavierInstance.subtracted.T,
                self.remixavierInstance.fs,
            )
            self.statusBar().showMessage("Done.", 0)
            self.progressBar.setValue(1000)
            self.startStopButton.setEnabled(1)

    def updateStatusLabel(self, text):
        self.statusBar().showMessage(text, 0)

    # Convenience function for creating sliders
    def createSlider(self, name, minimum, maximum, default, scale):
        slider = QSlider(Qt.Horizontal, self)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setMinimum(int(minimum * scale))
        slider.setMaximum(int(maximum * scale))
        slider.setValue(int(default * scale))
        slider.valueChanged.connect(self.updateLabels)
        return QLabel(name), slider

    # Create main GUI window
    def createMainFrame(self):
        # Initialize main widget
        self.mainFrame = QWidget()
        # Make widgets less squished
        self.mainFrame.setMinimumWidth(400)

        # Parameter sliders
        self.wiener_thresholdLabel, self.wiener_thresholdSlider = self.createSlider(
            "Wiener Threshold", -10, 10, 0, 1
        )
        self.max_skewLabel, self.max_skewSlider = self.createSlider("Max Skew", 0, 0.02, 0.02, 1000)

        # Update the slider labels with their default values
        self.updateLabels()

        # Progress bar for displaying progress
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(1)
        self.progressBar.setMaximum(1000)

        # Open button for starting analysis
        self.startStopButton = QPushButton("&Start")
        self.startStopButton.clicked.connect(self.startStopButtonClicked)

        # VBox for snippet length controls
        parametersVBox = QVBoxLayout()
        parametersVBox.addWidget(self.wiener_thresholdLabel)
        parametersVBox.addWidget(self.wiener_thresholdSlider)

        # Slider for max skew
        parametersVBox.addWidget(self.max_skewLabel)
        parametersVBox.addWidget(self.max_skewSlider)

        # Box for status bar and open button
        statusHBox = QHBoxLayout()
        statusHBox.addWidget(self.progressBar)
        statusHBox.addWidget(self.startStopButton)

        # Add all to main layout
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(parametersVBox)
        mainLayout.addLayout(statusHBox)

        # Add the main layout to the frame
        self.mainFrame.setLayout(mainLayout)
        # Main widget is the frame
        self.setCentralWidget(self.mainFrame)
        self.setFixedSize(self.mainFrame.minimumSize())

    # Updates all slider labels with their values
    def updateLabels(self):
        # Convert lengths to float (in seconds)
        self.wiener_thresholdLabel.setText(
            "Wiener Threshold: {}".format(self.wiener_thresholdSlider.value())
        )
        self.max_skewLabel.setText(f"Max Skew: {self.max_skewSlider.value() / 1000}")

    # Create menus
    def createMenu(self):
        # Open item
        openFile = QAction("Start Analysis", self)
        # Shortcut
        openFile.setShortcut("Ctrl+O")
        # Connect open action to show dialog
        openFile.triggered.connect(self.showDialog)
        # Exit (like quit)
        exitAction = QAction("Exit - like quitting", self)
        # Shortcut
        exitAction.setShortcut("Ctrl+E")
        exitAction.triggered.connect(self.exit)

        # Create menubar
        menubar = self.menuBar()
        # Add "file" menu
        fileMenu = menubar.addMenu("&File")
        # Add open and exit
        fileMenu.addAction(openFile)
        fileMenu.addAction(exitAction)

    def exit(self):
        if self.remixavierInstance.isRunning():
            self.remixavierInstance.quit()
        sys.exit()

    # Callback for when the start/stop button (aka "open" or "cancel") is clicked
    def startStopButtonClicked(self):
        if self.remixavierInstance.isRunning():
            # I don't know how to stop the thread... this doesn't work.
            self.remixavierInstance.quit()
            self.startStopButton.setEnabled(1)
        else:
            self.showDialog()

    # Show a dialog box for a set of audio files and start running remixavier
    def showDialog(self):
        # Get mixture file name from dialog box
        mix_file = QFileDialog.getOpenFileName(
            self, "Select a mixture file", ".", "Audio Files (*.mp3 *.wav)"
        )[0]
        # If the user didn't hit cancel
        if mix_file != "":
            # ... source file
            source_file = QFileDialog.getOpenFileName(
                self, "Select a source file", ".", "Audio Files (*.mp3 *.wav)"
            )[0]
            if source_file != "":
                # Where to save the resulting file
                self.output_file = QFileDialog.getSaveFileName(
                        self, "Save file", ".", "Audio Files (*.mp3 *.wav)"
                )[0]
                if self.output_file != "":
                    self.progressBar.reset()
                    self.remixavierInstance.loadNewValues(
                        mix_file, source_file, self.wiener_thresholdSlider.value(), self.max_skewSlider.value() / 1000
                    )
                    self.remixavierInstance.start()
                    self.startStopButton.setEnabled(0)

    def createStatusBar(self):
        self.statusText = QLabel('Click "Start" to begin.')
        self.statusBar().addWidget(self.statusText, 1)


def main():
    app = QApplication(sys.argv)
    form = AppForm()
    form.show()
    form.raise_()
    app.exec()


if __name__ == "__main__":
    main()

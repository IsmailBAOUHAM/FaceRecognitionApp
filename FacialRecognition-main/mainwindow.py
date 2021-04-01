import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot
from out_window import Ui_OutputDialog




class WelcomeWindow(QMainWindow):
    def __init__(self):
        super(WelcomeWindow, self).__init__()
        loadUi("./mainwindow.ui", self)


        self.setWindowIcon(QIcon('icon.png'))
        
        self.runButton.clicked.connect(self.runSlot)

        # def refreshAll(self):
        #     """
        #     Set the text of lineEdit once it's valid
        #     """
        #     self.Videocapture_ = "0"

    @pyqtSlot()
    def runSlot(self):
            """
            Called when the user presses the Run button
            """
            print("Clicked Run")

            ui.hide()  # hide the main window
            self.outputWindow_()  # Create and open new output window

    def outputWindow_(self):
            """
            Created new window for visual output of the video in GUI
            """
            self._new_window = Ui_OutputDialog()
            self._new_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = WelcomeWindow()
    ui.show()
    sys.exit(app.exec_())
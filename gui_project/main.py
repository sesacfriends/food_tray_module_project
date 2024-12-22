import sys
import os
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader

# IMPORT MODULES
from modules import *

# SET DPI
os.environ["QT_FONT_DPI"] = "96"

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        # LOAD UI FILE
        ui_file = QFile("main.ui")
        if not ui_file.open(QFile.ReadOnly):
            print(f"Cannot open UI file: {ui_file.errorString()}")
            sys.exit(-1)
            
        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()
        
        if not self.ui:
            print(loader.errorString())
            sys.exit(-1)

        # SET WINDOW TITLE
        self.ui.setWindowTitle("Camera Analysis App")

        # APPLY CUSTOM THEME
        self.apply_style()

        # BUTTON CONNECTIONS
        self.ui.btnToCamera.clicked.connect(self.show_camera)
        self.ui.btnComplete.clicked.connect(self.show_analysis)
        self.ui.btnToHome.clicked.connect(self.show_home)
        
        # SHOW INITIAL PAGE
        self.ui.stackedWidget.setCurrentWidget(self.ui.homePage)
        
        # SHOW WINDOW
        self.ui.show()

    def apply_style(self):
        # APPLY STYLESHEET
        style_file = QFile("themes/py_dracula_light.qss")
        if style_file.open(QFile.ReadOnly | QFile.Text):
            style = str(style_file.readAll(), 'utf-8')
            self.ui.setStyleSheet(style)

    def show_home(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.homePage)
        print("Moved to Home page")

    def show_camera(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.cameraPage)
        print("Moved to Camera page")

    def show_analysis(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.analysisPage)
        print("Moved to Analysis page")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
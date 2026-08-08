"""BioWave wireless-capable EMG mouse controller.

This is the supported entry point for EMG-to-mouse control with the BioWave
wired and Wi-Fi device protocols used by ``main3.py``.  The complete
implementation lives in :mod:`mouse2`, which includes serial/TCP input,
wireless discovery and authenticated UDP streaming, REST/FLEX calibration,
Random-Forest inference, gesture mapping, and safe pyautogui mouse control.

Keeping this as a thin entry point avoids maintaining two diverging copies of
the safety-critical mouse-control implementation.
"""

import sys

from PyQt5.QtWidgets import QApplication

from mouse2 import MouseControllerApp


def main():
    """Launch the BioWave wired/wireless mouse controller."""
    app = QApplication(sys.argv)
    window = MouseControllerApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

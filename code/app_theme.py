import os


THEME_COLORS = {
    "special": "#BF092F",
    "bg": "#132440",
    "title_bar": "#0D1B33",
    "panel": "#16476A",
    "accent": "#3B9797",
    "success": "#3B9797",
    "graph_bg": "#132440",
    "text": "#E8EEF0",
    "muted": "#A9C2CF",
    "disabled": "#6F8A99",
}


def apply_dark_title_bar(window):
    """Request a dark native title bar on Windows where supported."""
    try:
        import ctypes
        import sys
        from ctypes import wintypes
    except Exception:
        return False

    if sys.platform != "win32":
        return False

    try:
        hwnd = int(window.winId())
    except Exception:
        return False

    def _hex_to_colorref(hex_color):
        val = (hex_color or "").strip().lstrip("#")
        if len(val) != 6:
            return None
        try:
            r = int(val[0:2], 16)
            g = int(val[2:4], 16)
            b = int(val[4:6], 16)
        except ValueError:
            return None
        return (b << 16) | (g << 8) | r

    use_dark = ctypes.c_int(1)
    use_dark_size = ctypes.sizeof(use_dark)
    attrs = (20, 19)  # Win10 20H1+, then legacy fallback
    enabled_dark = False
    for attr in attrs:
        try:
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(attr),
                ctypes.byref(use_dark),
                wintypes.DWORD(use_dark_size),
            )
            if result == 0:
                enabled_dark = True
                break
        except Exception:
            continue

    caption_color = _hex_to_colorref(THEME_COLORS.get("title_bar", THEME_COLORS["bg"]))
    text_color = _hex_to_colorref(THEME_COLORS["text"])
    if caption_color is not None:
        try:
            caption_val = ctypes.c_int(caption_color)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(35),  # DWMWA_CAPTION_COLOR
                ctypes.byref(caption_val),
                wintypes.DWORD(ctypes.sizeof(caption_val)),
            )
        except Exception:
            pass
    if text_color is not None:
        try:
            text_val = ctypes.c_int(text_color)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(36),  # DWMWA_TEXT_COLOR
                ctypes.byref(text_val),
                wintypes.DWORD(ctypes.sizeof(text_val)),
            )
        except Exception:
            pass
    return enabled_dark


def app_stylesheet(font_size=16):
    c = THEME_COLORS
    default_font = "Bahnschrift" if os.name == "nt" else "Sans Serif"
    return f"""
QWidget {{
    background-color: {c['bg']};
    color: {c['text']};
    font-family: "{default_font}";
    font-size: {int(font_size)}px;
}}
QMainWindow, QDialog {{
    background-color: {c['bg']};
}}
QLabel {{
    color: {c['text']};
}}
QPushButton {{
    background-color: {c['accent']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {c['panel']};
    color: {c['text']};
}}
QPushButton:pressed {{
    background-color: {c['special']};
    border-color: {c['special']};
}}
QPushButton:disabled {{
    background-color: {c['panel']};
    color: {c['disabled']};
    border-color: {c['accent']};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    border-radius: 6px;
    padding: 4px 6px;
    selection-background-color: {c['accent']};
}}
QComboBox QAbstractItemView {{
    background-color: {c['panel']};
    color: {c['text']};
    selection-background-color: {c['accent']};
    border: 1px solid {c['accent']};
}}
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}
QTabWidget::pane {{
    border: 1px solid {c['accent']};
    background: {c['panel']};
}}
QTabBar::tab {{
    background: {c['panel']};
    color: {c['muted']};
    border: 1px solid {c['accent']};
    padding: 6px 12px;
}}
QTabBar::tab:selected {{
    background: {c['accent']};
    color: {c['text']};
}}
QHeaderView::section {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    padding: 4px;
}}
QTableWidget {{
    background-color: {c['panel']};
    color: {c['text']};
    gridline-color: {c['accent']};
    border: 1px solid {c['accent']};
}}
QProgressBar {{
    border: 1px solid {c['accent']};
    border-radius: 5px;
    text-align: center;
    background-color: {c['panel']};
    color: {c['text']};
}}
QProgressBar::chunk {{
    background-color: {c['success']};
}}
QScrollArea {{
    border: 1px solid {c['accent']};
    background-color: {c['panel']};
}}
QScrollBar:vertical {{
    border: none;
    background: {c['bg']};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c['accent']};
    min-height: 24px;
    border-radius: 5px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: transparent;
    border: none;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: {c['bg']};
}}
QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{
    background: transparent;
    width: 0px;
    height: 0px;
}}
QScrollBar:horizontal {{
    border: none;
    background: {c['bg']};
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {c['accent']};
    min-width: 24px;
    border-radius: 5px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    background: transparent;
    border: none;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: {c['bg']};
}}
QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {{
    background: transparent;
    width: 0px;
    height: 0px;
}}
"""


def configure_high_dpi():
    """Call once, before constructing QApplication.

    Without this, PyQt5 can report window/widget sizes in physical pixels on
    a Retina/HiDPI display (every MacBook since ~2012, plus most Windows
    laptops today) instead of logical points, which is what makes a window
    sized for a "normal" screen come out oversized and clip its own buttons.
    """
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication

        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass


def fit_window_to_screen(window, desired_width, desired_height, margin_ratio=0.92):
    """Resize+center ``window`` to fit the available screen.

    A window is only ever made smaller than requested, never larger, and it
    stays user-resizable afterwards. This is the actual fix for "buttons
    aren't visible on my MacBook" bugs: those happen when a window is coded
    for a bigger screen than the one it's opened on (e.g. a 1380x920 window
    on a 13" MacBook Air's 1280x800 logical desktop) and macOS clips it
    instead of shrinking it.
    """
    try:
        from PyQt5.QtWidgets import QApplication

        screen = window.screen() if hasattr(window, "screen") else None
        screen = screen or QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
    except Exception:
        avail = None

    width, height = int(desired_width), int(desired_height)
    if avail is not None:
        width = min(width, int(avail.width() * margin_ratio))
        height = min(height, int(avail.height() * margin_ratio))
    window.resize(width, height)
    if avail is not None:
        window.move(max(avail.x(), avail.center().x() - width // 2),
                    max(avail.y(), avail.center().y() - height // 2))


def wrap_in_scroll_area(window, central_widget):
    """Put ``central_widget`` inside a borderless, resizable QScrollArea and
    make it ``window``'s central widget.

    A window built this way never permanently hides a control when shown
    smaller than its ideal size (a MacBook's logical desktop, a split
    screen, an external-monitor scale change) - content scrolls instead of
    clipping. Returns the QScrollArea in case a caller needs it.
    """
    from PyQt5.QtWidgets import QFrame, QScrollArea

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(central_widget)
    window.setCentralWidget(scroll)
    return scroll


def apply_dark_theme(app, font_size=16):
    font = app.font()
    if os.name == "nt":
        from PyQt5.QtGui import QFontInfo

        font.setFamily("Bahnschrift")
        # Fallback for Windows systems where Bahnschrift is unavailable.
        if QFontInfo(font).family().lower() != "bahnschrift":
            font.setFamily("Segoe UI")
    else:
        font.setFamily("Sans Serif")
    app.setFont(font)
    app.setStyleSheet(app_stylesheet(font_size=font_size))


def themed_button_style(kind="accent"):
    c = THEME_COLORS
    if kind == "danger":
        bg = c["special"]
        text = c["text"]
        border = c["special"]
        hover_bg = c["panel"]
        hover_border = c["panel"]
    elif kind == "success":
        bg = c["success"]
        text = c["text"]
        border = c["accent"]
        hover_bg = c["panel"]
        hover_border = c["accent"]
    elif kind == "muted":
        bg = c["panel"]
        text = c["text"]
        border = c["accent"]
        hover_bg = c["bg"]
        hover_border = c["accent"]
    else:
        bg = c["accent"]
        text = c["text"]
        border = c["accent"]
        hover_bg = c["panel"]
        hover_border = c["accent"]
    return (
        f"QPushButton {{ "
        f"background-color: {bg}; color: {text}; font-weight: bold; "
        f"border: 1px solid {border}; border-radius: 6px; padding: 5px 12px; "
        f"}} "
        f"QPushButton:hover:!disabled {{ "
        f"background-color: {hover_bg}; border-color: {hover_border}; "
        f"}} "
        f"QPushButton:pressed:!disabled {{ "
        f"background-color: {c['special']}; border-color: {c['special']}; color: {c['text']}; "
        f"}} "
        f"QPushButton:disabled {{ "
        f"background-color: {c['panel']}; color: {c['disabled']}; border-color: {c['panel']}; "
        f"}}"
    )


def themed_label_style(kind="muted"):
    c = THEME_COLORS
    if kind == "danger":
        color = c["special"]
    elif kind == "success":
        color = c["success"]
    else:
        color = c["muted"]
    return f"color: {color}; font-weight: bold;"


def themed_status_color(color_hint=None):
    hint = (color_hint or "").strip().lower()
    success_hints = {"#2e7d32", "#00c853", "#4caf50", "#78a083", "#3b9797"}
    warning_hints = {"#ff9800", "#f57c00"}
    error_hints = {"#f44336", "#c62828", "#bf092f"}
    if hint in error_hints:
        return THEME_COLORS["special"]
    if hint in warning_hints:
        return THEME_COLORS["special"]
    if hint in success_hints:
        return THEME_COLORS["success"]
    return THEME_COLORS["muted"]

"""Tema escuro estilo DAW (QSS) e cores por instrumento."""

ACCENT = "#e8804a"
BG = "#14161b"
PANEL = "#1d2129"
PANEL_2 = "#252a34"
TEXT = "#d8dce4"
TEXT_DIM = "#8a93a3"

STEM_COLORS = {
    "vocals": "#e85d75",
    "drums": "#f2a541",
    "bass": "#4f9cf0",
    "guitar": "#50c878",
    "piano": "#b48ce0",
    "other": "#8a93a3",
    "click": "#ffd166",
}


def stem_color(name: str) -> str:
    if name in STEM_COLORS:
        return STEM_COLORS[name]
    if name.lower().startswith("guitarra"):
        shades = ["#50c878", "#37a76a", "#7fdca4", "#2c8a57"]
        try:
            idx = int("".join(ch for ch in name if ch.isdigit()) or 1) - 1
        except ValueError:
            idx = 0
        return shades[idx % len(shades)]
    return "#8a93a3"


QSS = f"""
* {{ font-family: 'Segoe UI', 'Ubuntu', sans-serif; font-size: 13px; }}
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}

QGroupBox {{
    background: {PANEL};
    border: 1px solid #2c313c;
    border-radius: 10px;
    margin-top: 16px;
    padding: 18px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px; top: 2px;
    color: {TEXT_DIM};
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 1px;
}}

QPushButton {{
    background: {PANEL_2};
    border: 1px solid #343a46;
    border-radius: 7px;
    padding: 7px 14px;
    color: {TEXT};
}}
QPushButton:hover {{ background: #2e3440; border-color: #424a59; }}
QPushButton:pressed {{ background: #20242d; }}
QPushButton:disabled {{ color: #565d6b; background: #1b1f26; border-color: #262b34; }}

QPushButton[kind="accent"] {{
    background: {ACCENT};
    border: none;
    color: #16110d;
    font-weight: 700;
}}
QPushButton[kind="accent"]:hover {{ background: #f09060; }}
QPushButton[kind="accent"]:disabled {{ background: #5a4334; color: #8a8077; }}

QPushButton[kind="tiny"] {{
    padding: 3px 6px;
    min-width: 24px;
    font-weight: 700;
    border-radius: 6px;
}}
QPushButton[kind="solo"], QPushButton[kind="mute"] {{
    padding: 3px 2px;
    min-width: 24px;
    font-weight: 700;
    border-radius: 6px;
}}
QPushButton[kind="solo"]:checked {{ background: #f2a541; color: #16110d; border-color: #f2a541; }}
QPushButton[kind="mute"]:checked {{ background: #e85d75; color: #16110d; border-color: #e85d75; }}
QPushButton[kind="loop"]:checked {{ background: {ACCENT}; color: #16110d; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{
    height: 6px; border-radius: 3px;
    background: #2b303b;
}}
QSlider::sub-page:horizontal {{
    height: 6px; border-radius: 3px;
    background: {ACCENT};
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: {TEXT};
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider:disabled {{ background: transparent; }}
QSlider::sub-page:horizontal:disabled {{ background: #3a3f4a; }}

QDoubleSpinBox, QSpinBox {{
    background: {PANEL_2};
    border: 1px solid #343a46;
    border-radius: 7px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
}}

QProgressBar {{
    background: {PANEL_2};
    border: none; border-radius: 6px;
    height: 10px; text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}

QStatusBar {{ background: {PANEL}; color: {TEXT_DIM}; border-top: 1px solid #2c313c; }}
QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid #424a59; padding: 6px; border-radius: 6px; }}
QLabel[kind="dim"] {{ color: {TEXT_DIM}; }}
QLabel[kind="time"] {{ font-family: 'Consolas', 'DejaVu Sans Mono', monospace; font-size: 14px; color: {TEXT}; }}
"""

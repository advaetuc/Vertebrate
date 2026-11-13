"""StatTile class."""
from .common import *  # shared Qt imports, constants, and helpers

class StatTile(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("tile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)
        self.value = make_label("—", 22, QFont.Weight.Bold, mono=True)
        layout.addWidget(self.value)
        layout.addWidget(make_label(title, 12, obj="muted"))

    def set(self, text: str, color: str | None = None) -> None:
        self.value.setText(text)
        self.value.setStyleSheet(f"color: {color};" if color else "")


"""FallLogDialog class."""
from .common import *  # shared Qt imports, constants, and helpers
from .falllog import FallLog

class FallLogDialog(QDialog):
    """Window listing this device's fall events; an ongoing fall is shown live on top."""
    COLUMNS = ("Fall detected at", "Ended at", "Duration", "Alarm", "Location", "Location source", "Ended by")
    LOCATION_COLUMN = 4

    def __init__(self, log: FallLog, parent=None):
        super().__init__(parent)
        self.log = log
        self.setWindowTitle(f"Fall log · {log.device}")
        self.resize(980, 520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        title = QLabel(f"Fall log — {log.device}")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)
        self.summary = QLabel()
        self.summary.setObjectName("subtitle")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.LOCATION_COLUMN, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        self.empty = QLabel("No falls recorded on this device yet.")
        self.empty.setObjectName("subtitle")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty)
        buttons = QHBoxLayout()
        folder = QPushButton("Open log file location")
        folder.clicked.connect(self.open_folder)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(folder)
        buttons.addStretch()
        buttons.addWidget(close)
        layout.addLayout(buttons)
        # Keeps the duration of an ongoing fall ticking while the window is open.
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
        self.timer.start()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    @Slot()
    def _tick(self):
        if self.log.current is not None:
            self.refresh()

    def refresh(self):
        rows = self.log.rows()
        entries = []
        if self.log.current is not None:
            current = self.log.current
            stamp = lambda moment: moment.strftime("%Y-%m-%d %H:%M:%S") if moment else ""
            entries.append(({
                "fall_detected_at": stamp(current["start"]),
                "ended_at": "ongoing",
                "duration": format_duration(self.log.ongoing_seconds()),
                **self._location_fields(current["location"]),
                "ended_by": "—",
                "alarm_rang_at": stamp(current["alarm_rang"]),
                "alarm_silenced_at": stamp(current["alarm_silenced"]),
            }, True))
        entries += [(row, False) for row in rows]
        self.table.setRowCount(len(entries))
        alert_bg = QColor("#631e2a")
        for r, (row, ongoing) in enumerate(entries):
            location = "—"
            if row.get("latitude") and row.get("longitude"):
                location = f"{float(row['latitude']):.5f}, {float(row['longitude']):.5f}"
                if row.get("accuracy_m"):
                    location += f"  (±{row['accuracy_m']} m)"
            alarm = "—"
            if row.get("alarm_rang_at"):
                alarm = f"Rang {row['alarm_rang_at'][-8:]}"  # time part; date is in the first column
                if row.get("alarm_silenced_at"):
                    alarm += f", silenced {row['alarm_silenced_at'][-8:]}"
            values = (row.get("fall_detected_at", ""), row.get("ended_at", ""), row.get("duration", ""),
                      alarm, location, row.get("location_source", ""), row.get("ended_by", ""))
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                if ongoing:
                    item.setBackground(alert_bg)
                self.table.setItem(r, c, item)
        self.table.setVisible(bool(entries))
        self.empty.setVisible(not entries)
        total = sum(float(row.get("duration_seconds") or 0) for row in rows)
        alarms = sum(1 for row in rows if row.get("alarm_rang_at"))
        text = f"{len(rows)} recorded fall{'s' if len(rows) != 1 else ''}"
        if rows:
            text += f" · alarm rang {alarms}× · total time in alert {format_duration(total)}"
        if self.log.current is not None:
            text += " · a fall is being detected now"
        text += f"\nFile: {self.log.path}"
        if self.log.last_error:
            text += f"\n{self.log.last_error}"
        self.summary.setText(text)

    @staticmethod
    def _location_fields(location):
        if location is None:
            return {"latitude": "", "longitude": "", "accuracy_m": "", "location_source": "unavailable"}
        lat, lon, accuracy, source = location
        return {"latitude": f"{lat}", "longitude": f"{lon}",
                "accuracy_m": f"{accuracy:.0f}" if accuracy else "", "location_source": source}

    @Slot()
    def open_folder(self):
        FALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(FALL_LOG_DIR)))


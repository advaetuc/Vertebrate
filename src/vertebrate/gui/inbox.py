"""Local inbox: worker-owned disk reads, atomic acknowledgement, timer polling."""
from datetime import datetime, timezone
from threading import Event

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                              QPushButton, QTextEdit, QVBoxLayout, QWidget)

from ..incidents import incident_to_json
from ..storage import StorageWorker
from .video import assert_gui_thread
from .workers import QtWorkerThread


class InboxWidget(QWidget):
    def __init__(self, outbox, parent=None):
        super().__init__(parent)
        self.outbox = outbox
        self.writer = None
        self.owns_writer = False
        self.fault = Event()
        self.pending = None
        self._signature = None
        self.opened_incident = None
        self.items = QListWidget()
        self.items.setAccessibleName('Local incidents')
        self.metadata = QTextEdit()
        self.metadata.setReadOnly(True)
        self.snapshot = QLabel('No snapshot selected')
        self.snapshot.setMinimumWidth(220)
        self.snapshot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.open_button = QPushButton('Open incident')
        self.ack_button = QPushButton('Acknowledge')
        self.retry_button = QPushButton('Retry save')
        self.message = QLabel('Local inbox')
        self.message.setWordWrap(True)
        buttons = QHBoxLayout()
        for button in (self.open_button, self.ack_button, self.retry_button): buttons.addWidget(button)
        content = QHBoxLayout()
        content.addWidget(self.items, 1)
        content.addWidget(self.metadata, 2)
        content.addWidget(self.snapshot, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.message)
        layout.addLayout(buttons)
        layout.addLayout(content)
        self.open_button.clicked.connect(self.open_selected)
        self.ack_button.clicked.connect(self.acknowledge_selected)
        self.retry_button.clicked.connect(self.retry_selected)

    def ensure_writer(self, retained=None):
        assert_gui_thread()
        if self.writer is not None:
            return
        self.writer = retained or StorageWorker(self.outbox, pause_acquisition=self.fault.set)
        if retained:
            self.writer.reopen(thread_factory=QtWorkerThread, wait_ready=False)
        else:
            self.writer.start(thread_factory=QtWorkerThread, wait_ready=False)
        self.owns_writer = True

    def bind(self, writer):
        if self.owns_writer:
            raise RuntimeError('Release the inbox writer before starting capture')
        self.writer = writer
        self._signature = None

    def release(self):
        """Nonblocking handoff; poll until True before opening another writer."""
        if self.writer is None:
            return True
        if self.owns_writer:
            self.writer.request_stop()
            if self.writer._thread.is_alive():
                return False
            self.writer._thread.dispose()
        self.writer = None
        self.owns_writer = False
        self.pending = None
        return True

    def _selected(self):
        item = self.items.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _request(self, operation):
        key = self._selected()
        if not key or self.pending or not self.writer or not self.writer.ready:
            return
        try:
            if operation == 'open':
                future = self.writer.read_incident(key)
            elif operation == 'ack':
                future = self.writer.acknowledge(key, datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
            else:
                future = self.writer.retry(key)
            self.pending = (operation, future)
            self.message.setText('Working…')
        except Exception as exc:
            self.message.setText(str(exc))

    def open_selected(self): self._request('open')
    def acknowledge_selected(self): self._request('ack')
    def retry_selected(self): self._request('retry')

    def poll(self):
        assert_gui_thread()
        writer = self.writer
        ready = writer is not None and writer.ready and writer.startup_error is None and writer._thread.is_alive()
        for button in (self.open_button, self.ack_button, self.retry_button):
            button.setEnabled(ready and self.pending is None)
        if writer is None:
            return
        if writer.startup_error:
            self.message.setText(f'Inbox unavailable: {writer.startup_error}')
        elif writer.startup_issues and self.message.text() == 'Local inbox':
            self.message.setText('Incomplete records ignored: ' + '; '.join(writer.startup_issues))
        records = writer.records
        signature = tuple((r.incident.incident_id, r.status, r.incident.acknowledgement.status, r.error) for r in records)
        if signature != self._signature:
            selected = self._selected()
            self.items.clear()
            for record in reversed(records):
                incident = record.incident
                item = QListWidgetItem(f'{incident.timing.confirmed_at_utc} · Person {incident.person.tracker_id}\n'
                    f'{record.status} · {incident.acknowledgement.status}')
                item.setData(Qt.ItemDataRole.UserRole, incident.incident_id)
                item.setToolTip(record.error or incident.incident_id)
                self.items.addItem(item)
                if incident.incident_id == selected: self.items.setCurrentItem(item)
            if self.items.count() and self.items.currentRow() < 0: self.items.setCurrentRow(0)
            self._signature = signature
        if self.pending and self.pending[1].done():
            operation, future = self.pending
            self.pending = None
            try:
                result = future.result()
                if operation == 'open':
                    incident, media = result
                    self.opened_incident = incident
                    self.metadata.setPlainText(incident_to_json(incident))
                    if media:
                        pixmap = QPixmap.fromImage(QImage.fromData(media))
                        self.snapshot.setPixmap(pixmap.scaled(320, 200, Qt.AspectRatioMode.KeepAspectRatio,
                                                             Qt.TransformationMode.SmoothTransformation))
                    else:
                        self.snapshot.setText(incident.media.snapshot_error or 'Snapshot unavailable')
                elif self.opened_incident and self.opened_incident.incident_id == result.incident_id:
                    self.opened_incident = result
                    self.metadata.setPlainText(incident_to_json(result))
                self.message.setText({'open': 'Incident opened', 'ack': 'Acknowledgement saved', 'retry': 'Incident saved'}[operation])
            except Exception as exc:
                self.message.setText(f'Not saved/opened: {exc}')
            for button in (self.open_button, self.ack_button, self.retry_button):
                button.setEnabled(ready)

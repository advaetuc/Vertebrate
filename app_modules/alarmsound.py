"""AlarmSound class."""
from .common import *  # shared Qt imports, constants, and helpers

class AlarmSound(QObject):
    """Loops the siren on this device's speakers until stopped. GUI thread only.
    Falls back to repeated system beeps if Qt Multimedia or the WAV is unavailable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ringing = False
        self._effect = None
        self._beeper = QTimer(self)
        self._beeper.setInterval(600)
        self._beeper.timeout.connect(QApplication.beep)
        if QSoundEffect is None:
            return
        try:
            self._effect = QSoundEffect(self)
            self._effect.setSource(QUrl.fromLocalFile(str(ensure_alarm_sound())))
            self._effect.setLoopCount(QSoundEffect.Loop.Infinite.value)
            self._effect.setVolume(1.0)  # App volume; the device's system volume still applies.
        except Exception as exc:
            print(f"Alarm sound unavailable, using system beeps: {exc}", file=sys.stderr)
            self._effect = None

    def start(self) -> None:
        if self.ringing:
            return
        self.ringing = True
        if self._effect is not None and self._effect.status() != QSoundEffect.Status.Error:
            self._effect.play()
        else:
            QApplication.beep()
            self._beeper.start()

    def stop(self) -> None:
        self.ringing = False
        self._beeper.stop()
        if self._effect is not None:
            self._effect.stop()


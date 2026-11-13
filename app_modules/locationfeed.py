"""LocationFeed class."""
from .common import *  # shared Qt imports, constants, and helpers

class LocationFeed(QObject):
    """Live device location for the watermark. GUI thread only.

    Two Qt Positioning sources may run side by side:
      "gps"    - NMEA from a serial GPS receiver, if one is detected;
      "system" - the OS location service (Windows Location / macOS CoreLocation),
                 which uses the device's own GPS chip when it has one and
                 otherwise Wi-Fi/network positioning.
    A fresh GPS-receiver fix wins; otherwise the system fix is shown. Only the
    latest fix per source is kept in memory; nothing is saved or sent by the app.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sources = {}
        self._fixes = {}     # kind -> (lat, lon, accuracy m or None, taken QDateTime, received QDateTime)
        self._problems = {}  # kind -> readable reason when that source has no usable fix
        self._running = False
        self._permission_asked = False
        self.gps_port = find_gps_port()
        if self.gps_port:
            source = QGeoPositionInfoSource.createSource(
                "nmea", {"nmea.source": f"serial:{self.gps_port}", "nmea.baudrate": GPS_BAUD}, self)
            if source is not None:
                self._add("gps", source)
        if SYSTEM_SOURCE and SYSTEM_SOURCE in QGeoPositionInfoSource.availableSources():
            source = QGeoPositionInfoSource.createSource(SYSTEM_SOURCE, self)
        else:
            source = QGeoPositionInfoSource.createDefaultSource(self)
        if source is not None:
            # All methods: lets the OS use a built-in GPS chip when present.
            source.setPreferredPositioningMethods(QGeoPositionInfoSource.PositioningMethod.AllPositioningMethods)
            self._add("system", source)
        elif "gps" not in self._sources:
            self._problems["system"] = "Location unavailable on this device"

    def _add(self, kind: str, source: QGeoPositionInfoSource) -> None:
        source.setUpdateInterval(LOCATION_INTERVAL_MS)
        source.positionUpdated.connect(lambda info, k=kind: self._on_position(k, info))
        source.errorOccurred.connect(lambda error, k=kind: self._on_error(k, error))
        self._sources[kind] = source
        self._problems[kind] = "Locating…"

    def start(self) -> None:
        self._running = True
        if "gps" in self._sources:
            self._sources["gps"].startUpdates()
        if "system" in self._sources:
            self._start_system_with_permission()

    def stop(self) -> None:
        self._running = False
        for source in self._sources.values():
            source.stopUpdates()

    def _start_system_with_permission(self) -> None:
        """Ask the OS for location access once, then start the system source.
        The source is started whatever the answer: if access is really blocked,
        it reports AccessError and the watermark shows how to allow it."""
        app = QCoreApplication.instance()
        permission = QLocationPermission()
        permission.setAccuracy(QLocationPermission.Accuracy.Precise)
        permission.setAvailability(QLocationPermission.Availability.WhenInUse)
        try:
            status = app.checkPermission(permission)
        except Exception:
            status = None
        if status == Qt.PermissionStatus.Undetermined and not self._permission_asked:
            self._permission_asked = True
            try:
                app.requestPermission(permission, self, lambda _result: self._start_system())
                return
            except Exception:
                pass
        self._start_system()

    def _start_system(self) -> None:
        if self._running and "system" in self._sources:
            self._sources["system"].startUpdates()

    def _on_position(self, kind: str, info: QGeoPositionInfo) -> None:
        coordinate = info.coordinate()
        if not info.isValid() or not coordinate.isValid():
            return
        accuracy = None
        attribute = QGeoPositionInfo.Attribute.HorizontalAccuracy
        if info.hasAttribute(attribute):
            accuracy = info.attribute(attribute)
        received = QDateTime.currentDateTime()
        taken = info.timestamp() if info.timestamp().isValid() else received
        self._fixes[kind] = (coordinate.latitude(), coordinate.longitude(), accuracy, taken, received)
        self._problems.pop(kind, None)

    def _on_error(self, kind: str, error) -> None:
        errors = QGeoPositionInfoSource.Error
        if kind == "gps":
            if error == errors.UpdateTimeoutError:
                self._problems[kind] = "GPS receiver: no satellite fix yet"
            else:
                self._problems[kind] = f"GPS receiver on {self.gps_port} could not be read"
            return
        if error == errors.AccessError:
            where = ("Settings > Privacy & security > Location" if IS_WINDOWS
                     else "System Settings > Privacy & Security > Location Services")
            self._problems[kind] = f"Location blocked: allow it in {where}"
        elif error == errors.ClosedError:
            self._problems[kind] = "Location services are turned off"
        elif error == errors.UpdateTimeoutError:
            if kind not in self._fixes:
                self._problems[kind] = "Waiting for a location fix…"
        else:
            self._problems[kind] = "Location unavailable"

    def _best(self, now: QDateTime):
        gps = self._fixes.get("gps")
        system = self._fixes.get("system")
        if gps is not None and gps[4].secsTo(now) <= GPS_FRESH_S:
            return "GPS receiver", gps
        if system is not None:
            label = "system location"
            if "gps" in self._sources:
                label += " (GPS receiver has no fix)"
            return label, system
        if gps is not None:
            return "GPS receiver", gps
        return None, None

    def current_fix(self):
        """(latitude, longitude, accuracy m or None, source label) or None."""
        label, fix = self._best(QDateTime.currentDateTime())
        if fix is None:
            return None
        lat, lon, accuracy, _taken, _received = fix
        return lat, lon, accuracy, label

    def problem_text(self) -> str:
        """Why there is no usable fix yet (for the sidebar)."""
        return self._problems.get("system") or self._problems.get("gps") or "Locating…"

    def watermark_lines(self) -> list[str]:
        now = QDateTime.currentDateTime()
        stamp = now.toString("dd MMM yyyy  HH:mm:ss")
        label, fix = self._best(now)
        if fix is None:
            problem = (self._problems.get("system") or self._problems.get("gps") or "Locating…")
            return [problem, stamp]
        lat, lon, accuracy, taken, _received = fix
        text = f"{abs(lat):.5f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.5f}° {'E' if lon >= 0 else 'W'}"
        if accuracy is not None and accuracy > 0:
            text += f"  ±{accuracy:.0f} m"
        age = taken.secsTo(now)
        if age > STALE_AFTER_S:
            text += f"  (fix {age // 60} min old)"
        return [text, f"{stamp}  ·  via {label}"]


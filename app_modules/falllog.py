"""FallLog class."""
from .common import *  # shared Qt imports, constants, and helpers

class FallLog:
    """Per-device CSV of confirmed fall alerts. GUI thread only.

    An event starts when the debounced alert turns on and ends when the alert
    has stayed off for END_GRACE_S (end time = when it first went off), or when
    the camera stops. One row is appended per finished event. The file lives in
    runtime/logs/falls/<device>.csv and opens directly in Excel or Numbers.
    """

    def __init__(self, device: str):
        self.device = device
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", device).strip("._") or "this-device"
        self.path = FALL_LOG_DIR / f"{safe}.csv"
        self.current = None       # ongoing event: {"start": datetime, "location": tuple | None}
        self._clear_since = None  # when the alert of the ongoing event last went off
        self.last_error = None

    def update(self, is_fallen: bool, location) -> bool:
        """Feed the alert state of each displayed frame. True if the log changed."""
        now = datetime.now().astimezone()
        if is_fallen:
            self._clear_since = None
            if self.current is None:
                self.current = {"start": now, "location": location,
                                "alarm_after": None, "alarm_rang": None, "alarm_silenced": None}
                return True
            if self.current["location"] is None and location is not None:
                self.current["location"] = location  # first fix arrived mid-event
            return False
        if self.current is None:
            return False
        if self._clear_since is None:
            self._clear_since = now
        elif (now - self._clear_since).total_seconds() >= END_GRACE_S:
            self._finish(self._clear_since, "Posture cleared")
            return True
        return False

    def close_event(self, reason: str) -> bool:
        """End an ongoing event (camera stopped, error, app closing)."""
        if self.current is None:
            return False
        self._finish(self._clear_since or datetime.now().astimezone(), reason)
        return True

    def mark_alarm(self, after_seconds: int) -> None:
        if self.current is not None and self.current["alarm_rang"] is None:
            self.current["alarm_after"] = after_seconds
            self.current["alarm_rang"] = datetime.now().astimezone()

    def mark_silenced(self) -> None:
        if self.current is not None and self.current["alarm_rang"] is not None:
            self.current["alarm_silenced"] = datetime.now().astimezone()

    def ongoing_seconds(self) -> float:
        if self.current is None:
            return 0.0
        return (datetime.now().astimezone() - self.current["start"]).total_seconds()

    def _finish(self, end: datetime, reason: str) -> None:
        event = self.current
        start, location = event["start"], event["location"]
        self.current, self._clear_since = None, None
        stamp = lambda moment: moment.strftime("%Y-%m-%d %H:%M:%S") if moment else ""
        seconds = (end - start).total_seconds()
        offset = start.strftime("%z")
        lat, lon, accuracy, source = location if location is not None else ("", "", None, "unavailable")
        row = {
            "device": self.device,
            "fall_detected_at": start.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": end.strftime("%Y-%m-%d %H:%M:%S"),
            "utc_offset": f"{offset[:3]}:{offset[3:]}" if offset else "",
            "duration": format_duration(seconds),
            "duration_seconds": f"{seconds:.1f}",
            "latitude": f"{lat:.6f}" if lat != "" else "",
            "longitude": f"{lon:.6f}" if lon != "" else "",
            "accuracy_m": f"{accuracy:.0f}" if accuracy else "",
            "location_source": source,
            "ended_by": reason,
            "alarm_after_seconds": event["alarm_after"] if event["alarm_after"] is not None else "",
            "alarm_rang_at": stamp(event["alarm_rang"]),
            "alarm_silenced_at": stamp(event["alarm_silenced"]),
        }
        try:
            FALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
            self._upgrade_header()
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            # utf-8-sig so Excel shows the ° and ± correctly; newline="" per csv docs.
            with self.path.open("a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            self.last_error = None
        except OSError as exc:
            self.last_error = f"Could not write the fall log: {exc}"
            print(self.last_error, file=sys.stderr)

    def _upgrade_header(self) -> None:
        """Rewrite a log made by an older version so its columns match LOG_FIELDS."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) == LOG_FIELDS:
                return
            old_rows = list(reader)
        with self.path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(old_rows)

    def rows(self) -> list[dict]:
        """Finished events for this device, newest first."""
        if not self.path.exists():
            return []
        try:
            with self.path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error) as exc:
            self.last_error = f"Could not read the fall log: {exc}"
            return []
        return list(reversed(rows))


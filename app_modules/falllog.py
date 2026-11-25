"""FallLog class."""
from .common import *  # shared Qt imports, constants, and helpers

class FallLog:
    """Per-device CSV of confirmed fall alerts. GUI thread only.

    An event starts when any person's fall is confirmed (five consecutive fallen
    frames of that one person) and ends when no one has been confirmed for
    END_GRACE_S (end time = when it first went off), or when the camera stops.
    person_ids lists who was confirmed; IDs restart with each camera session.
    What the person did afterwards (moved, lay still, got up) is recorded from
    the camera thread's post-fall reports (note_aftermath). One row is appended
    per finished event. The file lives in runtime/logs/falls/<device>.csv and
    opens directly in Excel or Numbers.
    """

    def __init__(self, device: str):
        self.device = device
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", device).strip("._") or "this-device"
        self.path = FALL_LOG_DIR / f"{safe}.csv"
        self.current = None       # ongoing event: {"start": datetime, "location": tuple | None}
        self._clear_since = None  # when the alert of the ongoing event last went off
        self.last_error = None
        self.last_finished = None  # the event _finish() just wrote (for caregiver alerts)

    def update(self, is_fallen: bool, location, person_ids=()) -> bool:
        """Feed the alert state of each displayed frame, with the IDs of the people
        whose fall is confirmed. True if the log changed."""
        now = datetime.now().astimezone()
        if is_fallen:
            self._clear_since = None
            if self.current is None:
                self.current = {"start": now, "location": location, "people": set(person_ids),
                                "alarm_after": None, "alarm_rang": None, "alarm_silenced": None,
                                "moved": False, "longest_still": 0.0, "critical_at": None,
                                "critical_ids": set(), "moved_again": False, "recovered": {},
                                "still_now": {}}
                return True
            self.current["people"].update(person_ids)  # someone else may fall during the event
            if self.current["location"] is None and location is not None:
                self.current["location"] = location  # first fix arrived mid-event
            return False
        if self.current is None:
            return False
        if self._clear_since is None:
            self._clear_since = now
        elif (now - self._clear_since).total_seconds() >= END_GRACE_S:
            got_up = bool(self.current["recovered"]) and not any(self.current["still_now"].values())
            self._finish(self._clear_since, "Got up" if got_up else "Posture cleared")
            return True
        return False

    def note_aftermath(self, aftermath: dict) -> list[str]:
        """Record the camera thread's post-fall reports ({person ID: report}) for
        the people in the ongoing event. Returns what changed for alerts:
        'critical' the first time someone lies still for STILL_CRITICAL_S, and
        'moving_again' the first time a still person moves afterwards."""
        event = self.current
        if event is None:
            return []
        changes = []
        for pid, report in aftermath.items():
            if pid not in event["people"]:
                continue
            event["moved"] |= bool(report["moved"])
            event["longest_still"] = max(event["longest_still"], float(report["longest_still_s"]))
            event["still_now"][pid] = report["status"] == "still"
            if report["status"] == "recovered":
                event["recovered"][pid] = float(report["down_s"])
            if report["critical"] and pid not in event["critical_ids"]:
                event["critical_ids"].add(pid)
                if event["critical_at"] is None:
                    event["critical_at"] = datetime.now().astimezone()
                    changes.append("critical")
            if report["moved_after_critical"] and not event["moved_again"]:
                event["moved_again"] = True
                changes.append("moving_again")
        return changes

    def still_people(self) -> list[int]:
        """People in the ongoing event lying still past the critical time."""
        return sorted(pid for pid, still in self.current["still_now"].items() if still) if self.current else []

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
            "person_ids": ";".join(str(i) for i in sorted(event["people"])),
            "after_fall": self.after_fall_text(event),
            "longest_still_seconds": f"{event['longest_still']:.0f}",
            "no_movement_alert_at": stamp(event["critical_at"]),
        }
        self.last_finished = {**row, "location": location, "seconds": seconds, "people": sorted(event["people"])}
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

    @staticmethod
    def after_fall_text(event: dict) -> str:
        """'Got up after 25 s', 'No movement for 18 s', 'Moved', or 'Not judged'."""
        if event["recovered"]:
            return f"Got up after {format_duration(min(event['recovered'].values()))}"
        if any(event["still_now"].values()):
            return f"No movement for {format_duration(event['longest_still'])}"
        if event["critical_at"] is not None:
            return f"Still for {format_duration(event['longest_still'])}, then moved"
        if event["moved"]:
            return "Moved"
        return "Not judged"

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


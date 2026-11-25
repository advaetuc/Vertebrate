"""Caregiver alerts via Discord: notifier, message builder and setup dialog."""
from .common import *  # shared Qt imports, constants, and helpers

# ---------------------------------------------------------------------------
# Caregiver alerts via Discord
# ---------------------------------------------------------------------------
# Posts a message to a Discord channel when a fall is confirmed, when the alarm
# rings and when the fall ends. Two ways to connect, both plain HTTPS with the
# standard library (no Discord package), identical on Windows and macOS:
#   webhook - a channel webhook URL (Channel settings > Integrations > Webhooks)
#   bot     - a bot token + channel ID (the bot needs "Send Messages" there)
# Messages carry this device's name, times, duration and location; never
# images or video. Settings live in QSettings (per user, never in the repo).
DISCORD_API = "https://discord.com/api/v10"
DISCORD_USER_AGENT = "DiscordBot (https://discord.com, 1.0) Vertebrate"  # Discord rejects requests without one
WEBHOOK_PREFIXES = ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/",
                    "https://canary.discord.com/api/webhooks/", "https://ptb.discord.com/api/webhooks/")
NOTIFY_ATTEMPTS = 3
# Fixed caregiver channel: when this file holds a Discord webhook URL, alerts are
# always on and go to that channel; the in-app setup is not shown. runtime/ is
# Git-ignored, so the URL (a secret: anyone with it can post) never reaches GitHub.
FIXED_WEBHOOK_FILE = APP_DIR / "runtime" / "config" / "discord_webhook.txt"
COLOR_FALL, COLOR_ALARM, COLOR_ENDED, COLOR_TEST = 0xFF8A4C, 0xF04452, 0x34D399, 0x2DD4BF
COLOR_CRITICAL = 0xB0102A  # no movement after a fall: deeper red than the alarm


def _https_context() -> ssl.SSLContext:
    """certifi's CA bundle when available: python.org Python on macOS ships no
    system certificates for urllib until 'Install Certificates' is run."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _settings_bool(settings: QSettings, key: str, default: bool) -> bool:
    value = settings.value(key, default)
    if isinstance(value, str):  # the Windows registry hands booleans back as text
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def fixed_webhook() -> str | None:
    """The webhook URL from FIXED_WEBHOOK_FILE (first non-empty line), if valid."""
    try:
        lines = FIXED_WEBHOOK_FILE.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    url = next((line.strip() for line in lines if line.strip()), "")
    if url and not url.startswith(WEBHOOK_PREFIXES):
        print(f"[caregivers] Ignoring {FIXED_WEBHOOK_FILE}: not a Discord webhook URL", file=sys.stderr, flush=True)
        return None
    return url or None


def load_caregiver_config(settings: QSettings) -> dict:
    url = fixed_webhook()
    if url:  # fixed channel: always on, every event, no ping
        return {"enabled": True, "mode": "webhook", "webhook": url, "token": "", "channel": "", "mention": "",
                "on_fall": True, "on_alarm": True, "on_end": True, "fixed": True}
    settings.beginGroup("caregivers")
    try:
        return {
            "enabled": _settings_bool(settings, "enabled", False),
            "mode": str(settings.value("mode", "webhook")),
            "webhook": str(settings.value("webhook", "")),
            "token": str(settings.value("token", "")),
            "channel": str(settings.value("channel", "")),
            "mention": str(settings.value("mention", "")),
            "on_fall": _settings_bool(settings, "on_fall", True),
            "on_alarm": _settings_bool(settings, "on_alarm", True),
            "on_end": _settings_bool(settings, "on_end", True),
            "fixed": False,
        }
    finally:
        settings.endGroup()


def save_caregiver_config(settings: QSettings, config: dict) -> None:
    settings.beginGroup("caregivers")
    for key, value in config.items():
        if key != "fixed":
            settings.setValue(key, value)
    settings.endGroup()
    settings.sync()


def caregiver_config_problem(config: dict) -> str | None:
    """Why this configuration cannot send, or None."""
    if config["mode"] == "bot":
        if not config["token"].strip():
            return "Enter the bot token."
        if not config["channel"].strip().isdigit():
            return "The channel ID is a long number (Discord: right-click the channel > Copy Channel ID)."
        return None
    if not config["webhook"].strip().startswith(WEBHOOK_PREFIXES):
        return "Paste a Discord webhook URL (it starts with https://discord.com/api/webhooks/)."
    return None


class DiscordNotifier(QObject):
    """Sends caregiver messages on a background thread so the window never waits
    on the network. Retries network errors and Discord rate limits; reports each
    outcome through `result` (delivered to the GUI thread)."""

    result = Signal(str, bool, str)  # kind, delivered, detail

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.config = load_caregiver_config(settings)
        self._jobs: queue.Queue = queue.Queue()
        self._ssl = _https_context()
        self._thread = threading.Thread(target=self._run, name="discord-notifier", daemon=True)
        self._thread.start()

    @property
    def active(self) -> bool:
        return self.config["enabled"] and caregiver_config_problem(self.config) is None

    def reload(self) -> None:
        self.config = load_caregiver_config(self.settings)

    def send(self, kind: str, embed: dict, mention: bool, config: dict | None = None) -> bool:
        """Queue one message. `config` overrides the saved one (used by Send test)."""
        if config is None:  # a real alert: obey the saved switches
            config = self.config
            if not self.active or not config.get(f"on_{kind}", True):
                reason = ("caregiver alerts are switched off — tick 'Send caregiver alerts' and press Save"
                          if not config["enabled"] else
                          caregiver_config_problem(config) or f"'{kind}' messages are switched off")
                print(f"[caregivers] {kind} alert not sent: {reason}", file=sys.stderr, flush=True)
                return False
        elif caregiver_config_problem(config) is not None:
            return False
        config = dict(config)  # snapshot: later setting changes don't affect a queued message
        content = config["mention"].strip() if mention else ""
        body = {
            "content": content,
            "embeds": [embed],
            # Only ping who the caregiver asked for; never stray @mentions from text.
            "allowed_mentions": {"parse": ["everyone", "roles", "users"]} if content else {"parse": []},
        }
        if config["mode"] == "webhook":
            body["username"] = "Vertebrate"
        self._jobs.put((kind, config, body))
        return True

    def flush(self, timeout: float) -> None:
        """Give queued messages a moment to go out (used when the app closes)."""
        deadline = time.monotonic() + timeout
        while self._jobs.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.05)

    # -- worker thread -------------------------------------------------------
    def _run(self) -> None:
        while True:
            kind, config, body = self._jobs.get()
            try:
                delivered, detail = self._deliver(config, body)
            except Exception as exc:  # never let the sender thread die
                delivered, detail = False, f"{type(exc).__name__}: {exc}"
            finally:
                self._jobs.task_done()
            print(f"[caregivers] {kind} alert " + ("delivered" if delivered else f"NOT delivered: {detail}"),
                  file=sys.stderr, flush=True)
            self.result.emit(kind, delivered, detail)

    def _deliver(self, config: dict, body: dict) -> tuple[bool, str]:
        headers = {"Content-Type": "application/json", "User-Agent": DISCORD_USER_AGENT}
        if config["mode"] == "bot":
            url = f"{DISCORD_API}/channels/{config['channel'].strip()}/messages"
            headers["Authorization"] = f"Bot {config['token'].strip()}"
        else:
            url = config["webhook"].strip()
            url += ("&" if "?" in url else "?") + "wait=true"
        data = json.dumps(body).encode("utf-8")
        last = "Unknown error"
        for attempt in range(NOTIFY_ATTEMPTS):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=10, context=self._ssl) as response:
                    if 200 <= response.status < 300:
                        return True, "Delivered"
                    last = f"HTTP {response.status}"
            except urllib.error.HTTPError as exc:
                text = exc.read().decode("utf-8", "replace")
                if exc.code == 429:  # rate limited: wait as long as Discord asks (bounded)
                    try:
                        wait = float(json.loads(text).get("retry_after", 1.0))
                    except (ValueError, AttributeError):
                        wait = 1.0
                    time.sleep(min(max(wait, 0.5), 10.0))
                    last = "Rate limited by Discord"
                    continue
                reasons = {
                    401: "Discord rejected the bot token.",
                    403: "The bot is not allowed to post in that channel (check its permissions).",
                    404: "Webhook or channel not found. Was it deleted?",
                }
                if exc.code in reasons:
                    return False, reasons[exc.code]  # retrying will not help
                last = f"Discord error HTTP {exc.code}"
            except urllib.error.URLError as exc:
                last = f"No connection to Discord ({exc.reason})"
            except TimeoutError:
                last = "Discord did not answer in time"
            if attempt < NOTIFY_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))  # 2 s, then 4 s, between attempts
        return False, last


def caregiver_embed(kind: str, device: str, title: str, description: str, location=None,
                    extra_fields: list | None = None, color: int | None = None) -> dict:
    """A Discord embed: coloured card with title, text, location and a map link."""
    fields = list(extra_fields or [])
    if location is not None:
        lat, lon, accuracy, source = location
        where = f"{lat:.5f}, {lon:.5f}" + (f" (±{accuracy:.0f} m)" if accuracy else "")
        fields.append({"name": "Location",
                       "value": f"{where}\n[Open in Google Maps](https://www.google.com/maps?q={lat:.6f},{lon:.6f})"
                                f"\nvia {source}", "inline": False})
    elif kind != "test":
        fields.append({"name": "Location", "value": "Not available on this device", "inline": False})
    if color is None:  # otherwise the fall's priority colour (fallanalysis.PRIORITY_STYLE)
        color = {"fall": COLOR_FALL, "alarm": COLOR_ALARM, "critical": COLOR_CRITICAL,
                 "end": COLOR_ENDED}.get(kind, COLOR_TEST)
    return {
        "title": title[:256],
        "description": description[:4000],
        "color": color,
        "fields": fields[:10],
        "footer": {"text": f"Vertebrate · {device}"},
        "timestamp": datetime.now().astimezone().isoformat(),
    }


class CaregiverDialog(QDialog):
    """Set up Discord caregiver alerts: connection, who to ping, which events."""

    def __init__(self, notifier: DiscordNotifier, device: str, parent=None):
        super().__init__(parent)
        self.notifier = notifier
        self.device = device
        self.setWindowTitle("Caregiver alerts · Discord")
        self.setMinimumWidth(620)
        config = load_caregiver_config(notifier.settings)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(12)
        layout.addWidget(make_label("Caregiver alerts", 22, QFont.Weight.Bold))
        layout.addWidget(make_label(
            "Post a message to a Discord channel when a fall is detected, when the alarm rings and when "
            "the fall ends. Messages include this device’s name, times and location — "
            "never images or video. Needs an internet connection; the alarm on this device works without one.",
            13, obj="muted", wrap=True))

        self.enabled = QCheckBox("Send caregiver alerts")
        self.enabled.setChecked(config["enabled"])
        layout.addWidget(self.enabled)

        layout.addWidget(caption("Connect with"))
        self.mode = QComboBox()
        self.mode.addItem("Webhook URL (simplest)", "webhook")
        self.mode.addItem("Bot token + channel ID", "bot")
        self.mode.setCurrentIndex(1 if config["mode"] == "bot" else 0)
        layout.addWidget(self.mode)

        self.webhook_box = QWidget()
        webhook = QVBoxLayout(self.webhook_box)
        webhook.setContentsMargins(0, 0, 0, 0)
        webhook.setSpacing(6)
        self.webhook = QLineEdit(config["webhook"])
        self.webhook.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)  # the URL is a secret
        self.webhook.setPlaceholderText("https://discord.com/api/webhooks/…")
        webhook.addWidget(self.webhook)
        webhook.addWidget(make_label("Discord: channel ⚙ Edit Channel › Integrations › Webhooks › New Webhook › "
                                     "Copy Webhook URL.", 12, obj="muted", wrap=True))
        layout.addWidget(self.webhook_box)

        self.bot_box = QWidget()
        bot = QVBoxLayout(self.bot_box)
        bot.setContentsMargins(0, 0, 0, 0)
        bot.setSpacing(6)
        self.token = QLineEdit(config["token"])
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("Bot token")
        self.channel = QLineEdit(config["channel"])
        self.channel.setPlaceholderText("Channel ID, e.g. 112233445566778899")
        bot.addWidget(self.token)
        bot.addWidget(self.channel)
        bot.addWidget(make_label("Developer Portal › your app › Bot › Reset Token. Invite the bot with "
                                 "“Send Messages” and “Embed Links”. Channel ID: enable Developer Mode, "
                                 "right-click the channel › Copy Channel ID.", 12, obj="muted", wrap=True))
        layout.addWidget(self.bot_box)

        layout.addWidget(caption("Ping"))
        self.mention = QLineEdit(config["mention"])
        self.mention.setPlaceholderText("Optional: @here, @everyone, <@&role-id> or <@user-id>")
        layout.addWidget(self.mention)

        layout.addWidget(caption("Send a message when"))
        self.on_fall = QCheckBox("A fall is detected")
        self.on_alarm = QCheckBox("The emergency alarm rings")
        self.on_end = QCheckBox("The fall ends (person up, silenced or camera stopped)")
        for box, key in ((self.on_fall, "on_fall"), (self.on_alarm, "on_alarm"), (self.on_end, "on_end")):
            box.setChecked(config[key])
            layout.addWidget(box)

        self.status = make_label("", 13, obj="muted", wrap=True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.test_button = QPushButton("Send test message")
        self.test_button.setObjectName("ghost")
        cancel = QPushButton("Cancel")
        cancel.setObjectName("ghost")
        save = QPushButton("Save")
        save.setObjectName("primary")
        buttons.addWidget(self.test_button)
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.test_button.clicked.connect(self.send_test)
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self.save)
        self.notifier.result.connect(self._on_result)
        self._mode_changed()

    def current_config(self) -> dict:
        return {
            "enabled": self.enabled.isChecked(),
            "mode": self.mode.currentData(),
            "webhook": self.webhook.text().strip(),
            "token": self.token.text().strip(),
            "channel": self.channel.text().strip(),
            "mention": self.mention.text().strip(),
            "on_fall": self.on_fall.isChecked(),
            "on_alarm": self.on_alarm.isChecked(),
            "on_end": self.on_end.isChecked(),
        }

    def _set_status(self, text: str, color: str = MUTED) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color};")

    @Slot()
    def _mode_changed(self, *_):
        bot = self.mode.currentData() == "bot"
        self.bot_box.setVisible(bot)
        self.webhook_box.setVisible(not bot)
        self.adjustSize()

    @Slot()
    def send_test(self):
        config = self.current_config()
        problem = caregiver_config_problem(config)
        if problem:
            self._set_status(problem, DANGER)
            return
        embed = caregiver_embed("test", self.device, "✅ Vertebrate test message",
                                f"Caregiver alerts from **{self.device}** are set up correctly. "
                                "Real alerts will look like this, with times and location.")
        self.notifier.send("test", embed, mention=False, config=config)
        self.test_button.setEnabled(False)
        self._set_status("Sending…")

    @Slot(str, bool, str)
    def _on_result(self, kind, delivered, detail):
        if kind != "test":
            return
        self.test_button.setEnabled(True)
        if delivered:
            self._set_status("Test message delivered. Check the Discord channel.", OK)
        else:
            self._set_status(f"Not delivered: {detail}", DANGER)

    @Slot()
    def save(self):
        config = self.current_config()
        problem = caregiver_config_problem(config)
        if config["enabled"] and problem:
            self._set_status(problem, DANGER)
            return
        save_caregiver_config(self.notifier.settings, config)
        self.notifier.reload()
        self.accept()

    def done(self, code):
        try:
            self.notifier.result.disconnect(self._on_result)
        except (RuntimeError, TypeError):
            pass
        super().done(code)

"""Network denial for the complete preflight, including dependency imports.

This guards Python network APIs used by the pinned stack, not arbitrary native
code. An OS firewall is needed for a security boundary against hostile code.
"""

from contextlib import contextmanager, ExitStack
import http.client
import os
from pathlib import Path
import platform
import socket
import subprocess
import urllib.request
from unittest.mock import patch

from .contracts import OfflineNetworkError


@contextmanager
def offline_guard(root: Path):
    attempts: list[str] = []
    # CPython 3.12.0 on Windows obtains OS version via the local `ver` command.
    # Cache it before subprocess denial; this performs no network operation.
    platform.uname()

    def deny(operation):
        def blocked(*args, **kwargs):
            attempts.append(operation)
            raise OfflineNetworkError(f'Offline mode blocked {operation}')
        return blocked

    # Retain these process-wide so later imports cannot silently re-enable them.
    os.environ.update(YOLO_OFFLINE='1', ULTRALYTICS_OFFLINE='1',
                      YOLO_AUTOINSTALL='false', YOLO_VERBOSE='false',
                      YOLO_CONFIG_DIR=str(root / 'runtime' / 'logs' / 'ultralytics'))
    with ExitStack() as stack:
        for obj, names in (
            (socket, ('create_connection', 'getaddrinfo', 'gethostbyname', 'gethostbyname_ex')),
            (socket.socket, ('connect', 'connect_ex', 'sendto')),
            (urllib.request, ('urlopen', 'urlretrieve')),
            (urllib.request.OpenerDirector, ('open',)),
            (http.client.HTTPConnection, ('connect', 'request')),
            (http.client.HTTPSConnection, ('connect',)),
            # Prevent shell-based installers/downloaders from bypassing guards.
            (subprocess, ('Popen',)),
            (os, ('system',)),
        ):
            for name in names:
                stack.enter_context(patch.object(obj, name, deny(f'{obj.__name__}.{name}')))
        yield attempts

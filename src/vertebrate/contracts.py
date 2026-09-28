"""Small immutable records shared by configuration and preflight consumers."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_IMPLEMENTED = "not_implemented"


class ConfigValidationError(ValueError):
    """A configuration violates the application contract."""


class AssetVerificationError(ValueError):
    """A local asset does not match its recorded manifest."""


class OfflineNetworkError(RuntimeError):
    """An operation tried to use the network during offline preflight."""


@dataclass(frozen=True)
class ModelAssetRecord:
    name: str
    path: Path
    task: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ConfigRecord:
    profile: str
    revision: int
    sha256: str


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class DoctorResult:
    status: Status
    offline: bool
    checks: tuple[CheckResult, ...]
    network_attempts: tuple[str, ...]
    config: ConfigRecord | None = None

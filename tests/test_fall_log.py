"""Clearing the fall log keeps a backup and never loses a fall in progress.

Run from the project folder (no camera needed):
    Windows  .venv\\Scripts\\python.exe -m unittest discover -s tests -v
    macOS    .venv/bin/python -m unittest discover -s tests -v
"""
import shutil
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_modules import falllog  # noqa: E402


class FallLogClearTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self._saved = falllog.FALL_LOG_DIR
        falllog.FALL_LOG_DIR = self.folder  # never touch the real log
        self.log = falllog.FallLog("test-device")

    def tearDown(self):
        falllog.FALL_LOG_DIR = self._saved
        shutil.rmtree(self.folder, ignore_errors=True)

    def _fall(self, person=1, end=True):
        self.log.update(True, None, [person])
        if end:
            self.log.update(False, None, [])
            self.log._clear_since -= timedelta(seconds=3)  # past the end-of-fall grace time
            self.log.update(False, None, [])

    def test_clear_keeps_a_backup_and_empties_the_list(self):
        for _ in range(3):
            self._fall()
        before = self.log.path.read_bytes()
        backup = self.log.clear()
        self.assertEqual(self.log.rows(), [])
        self.assertTrue(backup and backup.parent == self.folder / "cleared")
        self.assertEqual(backup.read_bytes(), before)

    def test_fall_in_progress_survives_clearing(self):
        self._fall()
        self._fall(person=2, end=False)  # still going on
        self.log.clear()
        self.assertIsNotNone(self.log.current)
        self.log.update(False, None, [])
        self.log._clear_since -= timedelta(seconds=3)
        self.log.update(False, None, [])
        rows = self.log.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["person_ids"], "2")

    def test_nothing_to_clear(self):
        self.assertIsNone(self.log.clear())


if __name__ == "__main__":
    unittest.main()

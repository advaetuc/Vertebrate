"""Debounce class."""
from .common import *  # shared Qt imports, constants, and helpers

class Debounce:
    def __init__(self):
        self.count = 0

    def update(self, any_fallen):
        self.count = min(self.count + 1, CONSECUTIVE_FRAMES) if any_fallen else 0
        return self.count >= CONSECUTIVE_FRAMES


"""Sorting: simulated bins, optionally mirrored to an Arduino servo diverter over USB serial.

Protocol (9600 baud, one line per glove):  'P\\n' -> PASS, 'R\\n' -> REWORK, 'X\\n' -> REJECT
The sketch in firmware/glove_sorter moves the servo to the bin angle and answers 'OK <c> <angle>'.
"""
from __future__ import annotations

import time

CODES = {"PASS": "P", "REWORK": "R", "REJECT": "X"}


def list_ports() -> list[str]:
    try:
        from serial.tools import list_ports as lp
        return [p.device for p in lp.comports()]
    except Exception:
        return []


class SimSorter:
    """Counts gloves per bin. Always available; the GUI draws these bins."""

    def __init__(self):
        self.bins = {k: 0 for k in CODES}
        self.history: list[tuple[float, int, str]] = []

    @property
    def connected(self) -> bool:
        return False

    def sort(self, glove_id: int, label: str) -> str | None:
        self.bins[label] += 1
        self.history.append((time.time(), glove_id, label))
        return None

    def close(self):
        pass


class SerialSorter(SimSorter):
    def __init__(self, port: str, baud: int = 9600, serial_factory=None):
        super().__init__()
        if serial_factory is None:
            import serial
            serial_factory = serial.Serial
        self.ser = serial_factory(port, baud, timeout=0.05)
        time.sleep(0.05)
        self.last_reply = ""

    @property
    def connected(self) -> bool:
        return self.ser is not None and getattr(self.ser, "is_open", True)

    def sort(self, glove_id: int, label: str) -> str | None:
        super().sort(glove_id, label)
        self.ser.write((CODES[label] + "\n").encode("ascii"))
        try:
            reply = self.ser.readline().decode("ascii", "ignore").strip()
        except Exception:
            reply = ""
        self.last_reply = reply
        return reply or None

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

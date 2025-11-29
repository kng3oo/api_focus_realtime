import csv
import os
import threading

class SafeCSVLogger:
    def __init__(self, csv_path: str, header):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        self._lock = threading.Lock()
        self._f = open(csv_path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if os.stat(csv_path).st_size == 0:
            self._w.writerow(header)
            self._f.flush()

    def write(self, row):
        with self._lock:
            self._w.writerow(row)
            self._f.flush()

    def close(self):
        try:
            self._f.close()
        except:
            pass

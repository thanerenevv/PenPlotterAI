import serial
import serial.tools.list_ports
import time
import re


class Plotter:
    GRBL_STARTUP = 2.0
    POLL_INTERVAL = 0.1
    ACK_TIMEOUT = 10.0

    def __init__(self, port=None, baud=115200):
        if port is None:
            port = self._autodetect_port()
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=2)
        time.sleep(self.GRBL_STARTUP)
        self.ser.flushInput()
        self._send_raw("\r\n\r\n")
        time.sleep(0.5)
        self.ser.flushInput()

    def _autodetect_port(self):
        candidates = [
            p.device
            for p in serial.tools.list_ports.comports()
            if "USB" in p.description or "ttyUSB" in p.device or "ttyACM" in p.device
        ]
        if not candidates:
            raise RuntimeError("No USB serial device found. Check connection.")
        return candidates[0]

    def _send_raw(self, text):
        self.ser.write(text.encode())

    def _send_command(self, cmd):
        line = cmd.strip() + "\n"
        self.ser.write(line.encode())
        deadline = time.time() + self.ACK_TIMEOUT
        while time.time() < deadline:
            resp = self.ser.readline().decode(errors="replace").strip()
            if resp.lower().startswith("ok") or resp.lower().startswith("error"):
                return resp
        raise TimeoutError(f"No ACK for command: {cmd}")

    def wait_idle(self):
        while True:
            self.ser.write(b"?")
            resp = self.ser.readline().decode(errors="replace").strip()
            if re.search(r"Idle", resp, re.IGNORECASE):
                return
            time.sleep(self.POLL_INTERVAL)

    def send_gcode(self, gcode, on_progress=None):
        lines = [
            l.strip()
            for l in gcode.splitlines()
            if l.strip() and not l.strip().startswith(";")
        ]
        total = len(lines)
        for i, line in enumerate(lines):
            self._send_command(line)
            if on_progress:
                on_progress(i + 1, total)
        self.wait_idle()

    def home(self):
        self._send_command("$H")
        self.wait_idle()

    def soft_reset(self):
        self.ser.write(b"\x18")
        time.sleep(1)
        self.ser.flushInput()

    def unlock(self):
        self._send_command("$X")

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

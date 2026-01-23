import threading
import time
from queue import Queue, Empty

import cv2
import serial


class FrameGrabber(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, output_queue: Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cap = cap
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            success, frame = self.cap.read()
            if not success:
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > 0:
                    pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if pos >= frame_count - 1:
                        self.stop_event.set()
                        if self.output_queue.empty():
                            self.output_queue.put(None)
                        break
                time.sleep(0.01)
                continue
            if not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except Empty:
                    pass
            self.output_queue.put(frame)


class SerialSender(threading.Thread):
    def __init__(self, port: str, baud: int, output_queue: Queue, stop_event: threading.Event, enabled: bool = True):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.ser = None
        self.enabled = enabled

    def open(self) -> None:
        self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
        time.sleep(2)

    def run(self) -> None:
        if not self.enabled:
            return
        try:
            self.open()
        except serial.SerialException:
            return
        while not self.stop_event.is_set():
            try:
                data = self.output_queue.get(timeout=0.1)
            except Empty:
                continue
            if self.ser is not None:
                self.ser.write(data.encode("utf-8"))

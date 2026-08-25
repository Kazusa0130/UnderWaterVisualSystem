"""Smoke test: run a few frames of the lateral_docking pipeline.

This script monkey-patches configuration to disable I/O and visualization,
then runs the main loop for a fixed number of frames.  It is used for quick
regression checks after algorithm changes.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config as _config

_config.DEBUG = False
_config.SAVE_OUTPUT = False
_config.SERIAL_ENABLED = False
_config.ENABLE_LIVE_VIZ = False
_config.TRACK_MODE = 0

import main

original_main = main.main


def limited_main(max_frames: int = 30):
    """Run the main loop for at most ``max_frames`` frames."""
    import cv2

    video_path = _config.VIDEO_PATH
    real_cap = cv2.VideoCapture(video_path)
    if not real_cap.isOpened():
        print(f"Smoke test skipped: cannot open {video_path}")
        return

    frame_counter = {"n": 0}

    class _CountedCapture:
        def __init__(self, cap):
            self._cap = cap

        def read(self):
            frame_counter["n"] += 1
            if frame_counter["n"] > max_frames:
                return False, None
            return self._cap.read()

        def isOpened(self):
            return self._cap.isOpened()

        def release(self):
            return self._cap.release()

        def get(self, prop):
            return self._cap.get(prop)

        def set(self, prop, value):
            return self._cap.set(prop, value)

    wrapped = _CountedCapture(real_cap)
    orig_video_capture = cv2.VideoCapture
    cv2.VideoCapture = lambda path: wrapped
    try:
        main.main()
    finally:
        cv2.VideoCapture = orig_video_capture
        real_cap.release()


if __name__ == "__main__":
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    _config.TRACK_MODE = mode
    import main

    # main.py binds config values at import time; patch the local copy for
    # this smoke test.
    main.TRACK_MODE = mode
    print(f"Smoke test: track_mode={mode}, 30 frames")
    limited_main(30)
    print("Smoke test passed.")

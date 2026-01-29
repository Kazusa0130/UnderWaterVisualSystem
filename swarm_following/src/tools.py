import os
from pathlib import Path
from typing import Tuple

import cv2


def ensure_save_dirs(base_path: str):
    raw_dir = Path(base_path) / "raw_data"
    out_dir = Path(base_path) / "output_data"
    serial_dir = Path(base_path) / "serial_log"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    serial_dir.mkdir(parents=True, exist_ok=True)


def count_files_in_directory(directory_path: str):
    raw_dir = Path(directory_path) / "raw_data"
    out_dir = Path(directory_path) / "output_data"
    raw_count = len([name for name in os.listdir(raw_dir) if (raw_dir / name).is_file()])
    out_count = len([name for name in os.listdir(out_dir) if (out_dir / name).is_file()])
    return raw_count, out_count


def open_video_writers(directory_path: str, fps: float, size: tuple):
    ensure_save_dirs(directory_path)
    raw_count, out_count = count_files_in_directory(directory_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_path = directory_path + f"raw_data/raw_{raw_count}.mp4"
    print(raw_path)
    out_path = directory_path + f"output_data/output_{out_count}.mp4"
    raw_writer = cv2.VideoWriter(raw_path, fourcc, fps, size)
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, size)
    return raw_writer, out_writer, raw_path, out_path


def open_serial_log(directory_path: str):
    ensure_save_dirs(directory_path)
    serial_dir = Path(directory_path) / "serial_log"
    log_count = len([name for name in os.listdir(serial_dir) if (serial_dir / name).is_file()])
    log_path = serial_dir / f"serial_{log_count}.txt"
    handle = open(log_path, "a", encoding="utf-8")
    return handle, str(log_path)

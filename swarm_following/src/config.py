from pathlib import Path

BASE_DIR = "./swarm_following/"

# WEIGHTS_PATH = str(BASE_DIR / "weights" / "best.pt")
WEIGHTS_PATH = str(BASE_DIR + "/models/best-1024ls.pt")
CAMERA_YAML = str(BASE_DIR + "config/stereo_camera_npu/camera_parameters.yaml")

CAMERA_INDEX = "D:/Documents/Project/BJTU_Under_Water_Visual_System/swarm_following/video/following_data1.mp4"
WIDTH = 640
HEIGHT = 480
FPS = 20
CONF = 0.7
MAX_DEPTH = 7000.0
MAX_AXIS = 5000.0
SERIAL_PORT = "/dev/ttyTHS0"
SERIAL_BAUD = 115200
SERIAL_ENABLED = False
SEND_ON_EMPTY = True
SHOW_FPS = True
SHOW_IMSHOW = True

BLACK_LOWER = (0, 0, 0)
BLACK_UPPER = (128, 128, 128)

YELLOW_LOWER = (20, 80, 80)
YELLOW_UPPER = (40, 255, 255)
RED_LOWER_1 = (0, 80, 80)
RED_UPPER_1 = (10, 255, 255)
RED_LOWER_2 = (170, 80, 80)
RED_UPPER_2 = (180, 255, 255)
MIN_ROI_SIZE = 2
MIN_COLOR_AREA = 200
SHOW_MASK = True
MASK_ALPHA = 0.1

SGBM_BLOCK_SIZE = 5
SGBM_PREFILTER_CAP = 15
SGBM_UNIQUENESS = 5
SGBM_SPECKLE_WINDOW = 100
SGBM_SPECKLE_RANGE = 2

WINDOW_NAME = "YOLOv8 Tracking"

SAVE_PATH = BASE_DIR + "outputs/"
SAVE_OUTPUT = True
SAVE_RAW_VIDEO = True
SAVE_OUTPUT_VIDEO = True
SAVE_SERIAL_LOG = False

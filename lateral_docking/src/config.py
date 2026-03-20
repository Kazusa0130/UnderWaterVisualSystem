MODEL_PATH = "./lateral_docking/models/best_0318.pt"
VIDEO_PATH = "./lateral_docking/videos/rawdata_14_cut1.avi"
# VIDEO_PATH = 0
CONFIG_PATH = "./lateral_docking/config/stereo_camera_npu/camera_parameters.yaml"
SAVE_PATH = "./lateral_docking/outputs/"

DEBUG = True
SAVE_OUTPUT = True

SERIAL_PORT = '/dev/ttyTHS1'
SERIAL_BAUD = 115200

OBJ_LENGTH = 0.60  # m
OBJ_WIDTH = 0.46   # m

FLIP = 1 # 传感器安装带来的倒置问题

POINT_MODULE = 0 # 0: 4 points, 1: 5 points
MIRROR_MODULE = 0 # 0: 不剔除镜像点, 1: 剔除镜像点
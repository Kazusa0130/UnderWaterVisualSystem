MODEL_PATH = "./lateral_docking/models/best_0513.pt"
# VIDEO_PATH = "./lateral_docking/videos/rawdata_14_cut1.avi"
VIDEO_PATH = "D:/Documents/Project/BJTU_Under_Water_Visual_System/Dev/five_point_detector_test/data/rawdata_35_cut1.avi"# VIDEO_PATH = 0
CONFIG_PATH = "./lateral_docking/config/stereo_camera_npu/camera_parameters.yaml"
SAVE_PATH = "./lateral_docking/outputs/"

DEBUG = True
SAVE_OUTPUT = True

SERIAL_PORT = '/dev/ttyTHS1'
SERIAL_BAUD = 115200

FLIP = 1 # 传感器安装带来的倒置问题

# POINT_MODULE 已由动态逻辑替代：当 red_led_detector 检测到红色光点时自动使用 5 点 PnP，
# 否则回退到 4 点 PnP。该参数不再需要在 main.py 中手动设置。
# POINT_MODULE = 0 # 0: 4 points, 1: 5 points

# 第五个光点（中心点）检测模式
# 0: 使用传统图像处理方法 (red_led_detector.py)
# 1: 使用 YOLO 目标检测输出
CENTER_POINT_MODE = 1

# YOLO 类别 ID 配置（仅在 CENTER_POINT_MODE = 1 时生效）
# 根据实际模型训练时的类别标签进行设置
YOLO_CORNER_CLASS_ID = 0   # 四个角点的类别 ID
YOLO_CENTER_CLASS_ID = 1   # 中心光点的类别 ID

MIRROR_MODULE = 0 # 0: 不剔除镜像点, 1: 剔除镜像点

# 轨迹可视化配置 (用于 visualize_traj.py 脚本)
# 实时3D可视化已移除，使用 visualize_traj.py 进行离线可视化
OBJ_LENGTH = 0.60  # m - 目标长度（用于可视化）
OBJ_WIDTH = 0.46   # m - 目标宽度（用于可视化）
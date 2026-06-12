MODEL_PATH = "./lateral_docking/models/best_0513.pt"
# VIDEO_PATH = "./lateral_docking/videos/rawdata_14_cut1.avi"
VIDEO_PATH = "./lateral_docking/videos/2026_0605/data7.mp4"# VIDEO_PATH = 0
CONFIG_PATH = "./lateral_docking/config/stereo_camera_npu/camera_parameters.yaml"
SAVE_PATH = "./lateral_docking/outputs/"

DEBUG = True
SAVE_OUTPUT = True


SERIAL_PORT = '/dev/ttyTHS1'
SERIAL_BAUD = 115200
SERIAL_ENABLED = False  # True: 启用串口输出；False: 禁用（开发环境建议关闭）

FLIP = 1 # 传感器安装带来的倒置问题

# YOLO 类别 ID 配置
YOLO_CORNER_CLASS_ID = 0   # 四个角点的类别 ID
YOLO_CENTER_CLASS_ID = 1   # 中心光点的类别 ID

MIRROR_MODULE = 0 # 0: 不剔除镜像点, 1: 剔除镜像点

# 4 点 PnP 模式下是否显示完整的位姿可视化（旋转轴 + Yaw/Pitch/Roll）
# True:  4 点模式与 5 点模式一样显示完整位姿（旋转轴、姿态角、平移量）
# False: 4 点模式仅显示 X/Y/Z 平移量（原有行为）
SHOW_ROTATION_FOR_4POINT = True

# 轨迹可视化配置 (用于 visualize_traj.py 脚本)
# 实时3D可视化已移除，使用 visualize_traj.py 进行离线可视化
OBJ_LENGTH = 0.60  # m - 目标长度（用于可视化）
OBJ_WIDTH = 0.46   # m - 目标宽度（用于可视化）

# 实时 matplotlib 位姿可视化窗口（独立窗口，类似 visualize_traj.py 风格）
ENABLE_LIVE_MATPLOTLIB_VIZ = True  # True: 启用独立 matplotlib 实时位姿窗口
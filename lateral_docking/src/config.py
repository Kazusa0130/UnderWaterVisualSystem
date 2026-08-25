MODEL_PATH = "./lateral_docking/models/best_0813.pt"
# VIDEO_PATH = "./lateral_docking/videos/rawdata_14_cut1.avi"
VIDEO_PATH = "./lateral_docking/videos/2026_0812/data8.mp4"# VIDEO_PATH = 0
CONFIG_PATH = "./lateral_docking/config/stereo_camera_npu_0813/camera_parameters.yaml"
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

# ---------------------------------------------------------------------------
# 角点几何校验（论文 4.3 节：SVM 替换为几何约束）
# ---------------------------------------------------------------------------
GEOMETRY_REJECT_INVALID = True       # 校验失败时是否拒绝该帧检测
GEOMETRY_MIN_AREA_PX2 = 100.0        # 最小四边形面积（px^2）
GEOMETRY_MAX_ASPECT_RATIO = 5.0      # 最大宽高比
GEOMETRY_MIN_ANGLE_DEG = 30.0        # 最小内角（度）

# ---------------------------------------------------------------------------
# PnP 物理门限
# ---------------------------------------------------------------------------
PNP_MIN_DEPTH_M = 0.4                # 最小有效深度（m）
PNP_MAX_RANGE_M = 50.0               # 最大有效距离（m）

# 轨迹可视化配置 (用于 visualize_traj.py 脚本)
# 实时3D可视化已移除，使用 visualize_traj.py 进行离线可视化
OBJ_LENGTH = 0.60  # m - 目标长度（用于可视化）
OBJ_WIDTH = 0.46   # m - 目标宽度（用于可视化）

# 实时位姿可视化总开关（仅 matplotlib 后端）
ENABLE_LIVE_VIZ = False  # True: 启用实时位姿窗口；False: 禁用（开发环境建议关闭）
LIVE_VIZ_BACKEND = "matplotlib"  # 已移除 Open3D，仅支持 "matplotlib"
LIVE_VIZ_FPS = 15  # 可视化窗口目标刷新率

# ---------------------------------------------------------------------------
# 多假设 PnP 参数（论文 5.2 节：多组 P3P 解过滤）
# ---------------------------------------------------------------------------
MULTIHYPOTHESIS_4P = True        # 4 点模式启用多假设 PnP（枚举 3 点子集 + 完整 4 点）
REPROJ_ERROR_THRESHOLD = 50.0    # 像素，重投影误差超过此阈值判为无效解（放宽以适应检测噪声）


# ---------------------------------------------------------------------------
# 传统光斑检测参数（track_mode=1）
# ---------------------------------------------------------------------------
THRESH_FALLBACK_PARAMS = {
    "brightness_thresh": 250,
    "halo_hue_range": (69, 89),
    "halo_sat_thresh": 60,
    "halo_val_thresh": 60,
    "min_area": 1122,
    "max_area": 11941,
    "min_circularity": 0.0,
    "max_aspect_ratio": 5.0,
    "morph_kernel_size": 3,
    "core_shrink_factor": 0.62,
    "core_dominant": True,
    "halo_overlap_ratio_thresh": 0.0,
}

# ---------------------------------------------------------------------------
# 阈值 fallback 位姿恢复参数
# ---------------------------------------------------------------------------
# 目标物理面积（m^2），用于从像素面积估算深度
THRESH_FALLBACK_TARGET_PHYSICAL_AREA_M2 = OBJ_WIDTH * OBJ_LENGTH
FALLBACK_ANGLE_FILL_VALUE = 0.0      # fallback 无法估计角度时填充的值

# ---------------------------------------------------------------------------
# 自动切换到传统跟踪的策略
# ---------------------------------------------------------------------------
# 当 YOLO+PnP 近距离解算成功后目标丢失，自动切到传统光斑跟踪并一直保持。
AUTO_SWITCH_TO_TRADITIONAL = True      # 是否启用自动切换
CLOSE_DISTANCE_THRESHOLD_M = 2.0       # 有效距离小于此值时标记条件1
LOST_FRAMES_BEFORE_TRADITIONAL = 5     # 条件1成立后连续丢失多少帧切换到传统跟踪

# ---------------------------------------------------------------------------
# 跟踪模式（与 CLAUDE.md 对齐）
# ---------------------------------------------------------------------------
TRACK_MODE = 0  # 0: YOLO + PnP; 1: 传统光斑特征提取（最大面积）
ENABLE_TRACK_MODE_COLUMN = False     # CSV 是否附加 track_mode 列（兼容旧格式）



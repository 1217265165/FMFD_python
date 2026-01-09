# 频段划分与容差配置
# 单频段模式：只有一个频段范围
# 注意：真实数据只有一个频段（10MHz~8.2GHz），不需要切换点检测
SINGLE_BAND_MODE = True  # 启用单频段模式

# 单频段范围（真实数据）
SINGLE_BAND_RANGE = (1e7, 8.2e9)  # 10MHz ~ 8.2GHz

# 多频段范围（仿真数据，向后兼容）
BAND_RANGES = [
    (9e3, 1.3e10),
    (1.3e10, 4.6e10),
    (4.6e10, 6.0e10),
]
K_LIST = [3, 4, 5]            # 每段包络系数
SWITCH_TOL = 0.2              # 切换点步进容差 (dB)
N_POINTS = 10000              # 基线频率网格点数（原值）
N_POINTS_REAL = 1000          # 真实数据基线点数（更少，因为原始数据只有~820点）

# 包络参数（单频段模式使用）
ENVELOPE_Q_LOW = 0.02         # 下包络分位数
ENVELOPE_Q_HIGH = 0.98        # 上包络分位数
ENVELOPE_SMOOTH_WINDOW = 21   # 平滑窗口大小

# 突变检测参数
ABRUPT_Q_DR = 0.995           # 一阶差分阈值分位数
ABRUPT_Q_D2R = 0.995          # 二阶差分阈值分位数

# 禁用的故障模块（单频段真实数据模式）
# 前置放大器在前放OFF模式下不是诊断对象
DISABLED_FAULT_MODULES = ['前置放大器']

# 字体配置（中文）
FONT_FAMILY = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]

# 默认文件路径（统一输出到 Output/ 下）
OUTPUT_DIR = "Output"
BASELINE_ARTIFACTS = f"{OUTPUT_DIR}/baseline_artifacts.npz"
BASELINE_META = f"{OUTPUT_DIR}/baseline_meta.json"
NORMAL_FEATURE_STATS = f"{OUTPUT_DIR}/normal_feature_stats.csv"
SWITCH_CSV = f"{OUTPUT_DIR}/switching_features.csv"
SWITCH_JSON = f"{OUTPUT_DIR}/switching_features.json"
PLOT_PATH = f"{OUTPUT_DIR}/baseline_switching.png"
DETECTION_RESULTS = f"{OUTPUT_DIR}/detection_results.csv"
SIM_FAULT_DATASET = f"{OUTPUT_DIR}/sim_fault_dataset.csv"
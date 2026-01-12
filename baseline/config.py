# Single-band mode configuration
# 单频段数据：freq=10MHz→8.2GHz，step=10MHz，N=820
SINGLE_BAND_MODE = True

# Single-band frequency parameters
FREQ_START_HZ = 1e7      # 10 MHz
FREQ_STEP_HZ = 1e7       # 10 MHz step
FREQ_N_POINTS = 820      # 820 points → 10MHz to 8.2GHz

# Legacy multi-band configuration (disabled in single-band mode)
BAND_RANGES = [
    (FREQ_START_HZ, FREQ_START_HZ + (FREQ_N_POINTS - 1) * FREQ_STEP_HZ),
] if SINGLE_BAND_MODE else [
    (9e3, 1.3e10),
    (1.3e10, 4.6e10),
    (4.6e10, 6.0e10),
]
K_LIST = [3.5] if SINGLE_BAND_MODE else [3, 4, 5]  # 单频段使用单一包络系数
SWITCH_TOL = 0.2              # 切换点步进容差 (dB)
N_POINTS = FREQ_N_POINTS if SINGLE_BAND_MODE else 10000

# Coverage validation thresholds
COVERAGE_MEAN_MIN = 0.97      # 平均覆盖率必须 >= 97%
COVERAGE_MIN_MIN = 0.93       # 最小覆盖率必须 >= 93%

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
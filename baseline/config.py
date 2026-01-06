# 频段划分与容差配置
BAND_RANGES = [
    (9e3, 1.3e10),
    (1.3e10, 4.6e10),
    (4.6e10, 6.0e10),
]
K_LIST = [3, 4, 5]            # 每段包络系数
SWITCH_TOL = 0.2              # 切换点步进容差 (dB)
N_POINTS = 10000              # 基线频率网格点数

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
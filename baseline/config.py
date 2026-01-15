# Single-band mode configuration
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
K_LIST = [3.5] if SINGLE_BAND_MODE else [3, 4, 5]  # backward compatibility
SWITCH_TOL = 0.2              # 切换点步进容差 (dB)
N_POINTS = FREQ_N_POINTS if SINGLE_BAND_MODE else 10000

# Vendor prior tolerance (dB) per frequency segment
# 10–100MHz: ±0.80 dB
# 100MHz–3.25GHz: ±0.40 dB
# 3.25–5.25GHz: ±0.60 dB
# 5.25–8.2GHz: ±0.80 dB
VENDOR_TOL_SEGMENTS = [
    ((1e7, 1e8), 0.80),
    ((1e8, 3.25e9), 0.40),
    ((3.25e9, 5.25e9), 0.60),
    ((5.25e9, 8.2e9), 0.80),
]

# Coverage validation thresholds
TARGET_COVERAGE = 0.97            # mean coverage target
TARGET_COVERAGE_MIN_SEG = 0.95    # per-segment minimum coverage
COVERAGE_MEAN_MIN = TARGET_COVERAGE  # retained for backward compatibility
COVERAGE_MIN_MIN = 0.93           # legacy fallback

# RRS smoothing controls
RRSM_SG_WINDOW = 41               # Savitzky-Golay window (must be odd)
RRSM_SG_POLY = 3                  # SG polynomial order
RRSM_SG_MAE_MAX = 0.03            # max MAE between smoothed and raw (dB)

# Outlier filtering
OUTLIER_VIOL_RATE_TH = 0.15       # >15% points exceed prior*1.2 → outlier
PRIOR_EXPAND_FOR_OUTLIER = 1.2

# Envelope computation controls
QUANTILE_TARGET = TARGET_COVERAGE
SEG_SMOOTH_WINDOW_RATIO = 0.05    # 3%~6% recommended; use 5% default
SEG_SMOOTH_WINDOW_MIN = 7         # minimum window length (odd enforced)
PRIOR_MAX_FACTOR = 1.1            # soft upper bound factor for width
TRANSITION_POINTS = 15            # points for segment boundary blending
W_FLOOR_MIN = 0.05                # minimum half-width (dB)

# Smoothness check default thresholds (used by quality checker)
SMOOTHNESS_STD_MAX = 0.01

# Font configuration (Chinese)
FONT_FAMILY = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]

# Default file paths (anchored under Output/)
OUTPUT_DIR = "Output"
BASELINE_ARTIFACTS = f"{OUTPUT_DIR}/baseline_artifacts.npz"
BASELINE_META = f"{OUTPUT_DIR}/baseline_meta.json"
NORMAL_FEATURE_STATS = f"{OUTPUT_DIR}/normal_feature_stats.csv"
SWITCH_CSV = f"{OUTPUT_DIR}/switching_features.csv"
SWITCH_JSON = f"{OUTPUT_DIR}/switching_features.json"
PLOT_PATH = f"{OUTPUT_DIR}/baseline_switching.png"
DETECTION_RESULTS = f"{OUTPUT_DIR}/detection_results.csv"
SIM_FAULT_DATASET = f"{OUTPUT_DIR}/sim_fault_dataset.csv"
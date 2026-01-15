# Baseline and RRS/Envelope exports
from .baseline import (
    load_and_align,
    align_to_frequency,
    compute_rrs_bounds,
    compute_coverage,
    detect_switch_steps,
    auto_expand_envelope,
)

from .rrs_envelope import (
    build_frequency_axis_hz,
    vendor_tolerance_db,
    compute_rrs,
    build_rrs_and_envelope,
    build_rrs_and_envelope_v4,
    build_rrs_and_envelope_v5,  # New v5 with stable band offset and vendor exceed outlier detection
    compute_rrs_bounds_v2,
    compute_envelope_width_v4,
    check_width_smoothness,
    # Global offset functions
    estimate_global_offset,
    remove_global_offsets,
    get_stable_band_mask,
    # Outlier detection (v5)
    detect_outliers_by_vendor_exceed,
    compute_segmented_extra_margin,
    # Configuration constants
    EXTRA_MAX_DEFAULT,
    EXTRA_MAX_LIMIT,
    SMOOTHNESS_THRESHOLD,
    WINDOW_VARIATION_MAX,
    COVERAGE_MEAN_TARGET,
    COVERAGE_MIN_TARGET,
    STABLE_BAND_START_HZ,
    STABLE_BAND_END_HZ,
)

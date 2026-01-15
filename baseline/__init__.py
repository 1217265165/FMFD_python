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
    compute_rrs_bounds_v2,
    compute_envelope_width_v4,
    check_width_smoothness,
    # Configuration constants
    EXTRA_MAX_DEFAULT,
    EXTRA_MAX_LIMIT,
    SMOOTHNESS_THRESHOLD,
    WINDOW_VARIATION_MAX,
    COVERAGE_MEAN_TARGET,
    COVERAGE_MIN_TARGET,
)

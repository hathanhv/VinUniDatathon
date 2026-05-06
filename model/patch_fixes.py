"""patch_fixes.py — applies 4 inference fixes to final_pipeline.py"""
import re

with open("final_pipeline.py", encoding="utf-8") as f:
    src = f.read()

original_len = len(src)
changes = []

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: Freeze _trend_days_from_peak, _year in _build_future_row
# ─────────────────────────────────────────────────────────────────────────────
OLD2 = (
    '        # \u2500\u2500 Trend \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '        "_trend_days_from_peak": float((fdate - TREND_PEAK_DATE).days),\n'
    '        "_annual_yoy_growth":    float(last_yoy_growth),\n'
)
NEW2 = (
    '        # \u2500\u2500 Trend (FROZEN at 2022-12-31 \u2014 prevents OOD extrapolation) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '        # Freeze trend/year: extrapolating the declining-trend past 2022\n'
    '        # was the root cause of ~2.4x Revenue underprediction in inference.\n'
    '        "_trend_days_from_peak": float(_FROZEN_TREND_DAYS),\n'
    '        "_annual_yoy_growth":    float(last_yoy_growth),\n'
    '        "_year":                 _FROZEN_YEAR,\n'
)
if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1)
    changes.append("FIX2: trend freeze applied")
else:
    changes.append("FIX2: NOT FOUND - check manually")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2b: Also freeze _trend_norm derivation
# ─────────────────────────────────────────────────────────────────────────────
OLD2b = '    # Derived from frozen trend\n    row["_trend_norm"] = _FROZEN_TREND_NORM\n'
OLD2b_alt = '    # Derived from trend\n    row["_trend_norm"] = row["_trend_days_from_peak"] / TREND_NORM_DAYS\n'
if OLD2b not in src and OLD2b_alt in src:
    src = src.replace(OLD2b_alt,
        '    # Derived from frozen trend (consistent with frozen _trend_days_from_peak)\n'
        '    row["_trend_norm"] = _FROZEN_TREND_NORM\n', 1)
    changes.append("FIX2b: trend_norm freeze applied")
elif OLD2b in src:
    changes.append("FIX2b: already frozen")
else:
    changes.append("FIX2b: NOT FOUND")

with open("final_pipeline.py", "w", encoding="utf-8") as f:
    f.write(src)

print("\n".join(changes))
print(f"File size: {original_len} -> {len(src)} bytes")

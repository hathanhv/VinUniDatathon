"""patch_trend.py - freezes trend features in _build_future_row"""
import re, sys
sys.stdout.reconfigure(encoding='utf-8')

src = open("final_pipeline.py", encoding="utf-8").read()

OLD = (
    '        "_trend_days_from_peak": float((fdate - TREND_PEAK_DATE).days),\n'
    '        "_annual_yoy_growth":    float(last_yoy_growth),\n'
)
NEW = (
    '        "_trend_days_from_peak": float(_FROZEN_TREND_DAYS),  # frozen @ 2022-12-31\n'
    '        "_annual_yoy_growth":    float(last_yoy_growth),\n'
    '        "_year":                 _FROZEN_YEAR,  # guard OOD: 2023/2024 outside training\n'
)

if OLD in src:
    src = src.replace(OLD, NEW, 1)
    open("final_pipeline.py", "w", encoding="utf-8").write(src)
    print("Trend freeze: APPLIED")
else:
    print("ERROR: pattern not found")
    idx = src.find("_trend_days_from_peak")
    print("Context:", repr(src[idx-20:idx+120]))

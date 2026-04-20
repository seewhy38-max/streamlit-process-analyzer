## 2026-02-06 - Vectorized Pivot for Staging Calculations
**Learning:** Manual lot-by-lot loops with internal DataFrame filtering are a massive bottleneck in this codebase. Replacing them with a vectorized pivot approach (pivoting once for all timestamps) yields ~75x speedup even on small datasets (500 lots).
**Action:** Always check for manual loops over unique identifiers (like 'Lot ID') and replace with `pivot_table` or group-by operations where possible. Use vectorized string formatting (`dt.strftime`) for report generation.

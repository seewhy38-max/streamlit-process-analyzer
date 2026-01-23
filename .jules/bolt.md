## 2024-05-20 - Inefficient looping in staging time calculation

**Learning:** The `calculate_staging_time_violations` function uses inefficient loops (`for lot in ...`, `for ... in df_check.iterrows()`) for data pivoting and processing. This is a common pandas anti-pattern and a clear performance bottleneck.

**Action:** I will replace these loops with vectorized pandas operations like `pivot_table` and direct DataFrame manipulations to improve performance.

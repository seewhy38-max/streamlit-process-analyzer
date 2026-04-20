## 2026-02-04 - Optimizing Staging Time Violations
**Learning:** Nested loops over unique lot/step IDs with internal DataFrame filtering is a major performance bottleneck ((N \times M)$ where $ is lots and $ is steps). Replacing this with a single `pivot_table` and vectorized operations provides a massive speedup (~70x). Vectorized string formatting with `.dt.strftime()` is also significantly faster than `.apply()` or `iterrows()` for report generation.
**Action:** Always prefer `pivot_table` or `groupby` over manual loops for reshaping data. Use vectorized string operations for report generation.

**Learning:** When using `pd.pivot_table`, the resulting columns Index often gets a name from the pivot column (e.g., 'SPECNAME'). This can cause issues with `pd.testing.assert_frame_equal` and might be unexpected.
**Action:** Explicitly set `df.columns.name = None` after pivoting to maintain a standard DataFrame structure.

**Learning:** Initializing a new DataFrame column with `np.nan` before assigning datetime values can lead to silent type conversion to float64, which breaks subsequent `.dt` accessor calls.
**Action:** Use `pd.NaT` to initialize datetime-bound columns.

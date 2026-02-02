# BOLT'S JOURNAL - CRITICAL LEARNINGS ONLY

## 2026-02-02 - [Optimizing Staging Time Calculation]
**Learning:** Manual iteration over lot IDs and steps for filtering and pivoting is a major bottleneck. Vectorizing this using `pivot_table` and DataFrame operations provided a 65x speedup for 500 lots.
**Action:** Always look for manual loops that perform filtering or lookups on DataFrames and replace them with vectorized Pandas operations like `pivot_table`, `merge`, or `groupby`.

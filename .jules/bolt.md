## 2026-02-01 - Vectorizing complex multi-step aggregations
**Learning:** Manual loops over lots to filter data for specific steps is an O(N*M*R) operation that kills performance in Streamlit. Using a helper column to map relevant timestamps followed by a single `pivot_table` reduces this to O(N).
**Action:** Always look for patterns where data is filtered repeatedly in a loop; replace with `pivot_table` or `groupby` + `unstack`.

## 2026-02-01 - Handling CRLF in automated diffs
**Learning:** `replace_with_git_merge_diff` can be unreliable with CRLF. Converting to Unix format with `dos2unix`, applying the diff, and then converting back with `unix2dos` is a reliable workflow to preserve line endings while ensuring patch compatibility.
**Action:** Use `dos2unix`/`unix2dos` when modifying legacy Windows-originated Python files.

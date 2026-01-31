## 2026-01-31 - [Pandas Vectorization and Environment Gotchas]
**Learning:**
1. The codebase uses CRLF line endings, which causes `replace_with_git_merge_diff` to fail if the search block doesn't account for them. Using `dos2unix` before applying changes and `unix2dos` before committing keeps diffs clean.
2. `pd.pivot_table` can introduce a name to the columns index (e.g., the name of the column being pivoted), which might cause `pd.testing.assert_frame_equal` to fail if the original dataframe didn't have it.
3. Replacing `iterrows()` with vectorized Pandas operations (like `.dt.strftime()`) significantly improves performance beyond just the data pivoting itself.

**Action:**
1. Always check line endings with `cat -A` if `replace_with_git_merge_diff` fails.
2. Explicitly clear `df.columns.name` after a pivot if consistency is required.
3. Prioritize vectorizing the final report generation/formatting step, not just the core data transformation.

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import StringIO
from datetime import timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- Global Configuration (Define your critical steps and staging limits here) ---
CRITICAL_STEPS = {
    '4282': 'TIM-Adhesive Dispensing', '4285': 'LID ATTACH',
    '4276': 'STIFFENER DISPENSE', '4277': 'STIFFENER ATTACH',
    '4271': 'AOI (TOP Inspection)',
    '4255': 'AOI (Bottom inspection)',
    '4269': 'PGA/LGA CHIP CAP REWORK (Top)',
    '4221': 'AOI RERUN', '4294': 'MPU AUTO FLIP',
    '4404': 'BGA BALL ATTACH', '4414': 'BGA OS', '4415': 'BGA ICOS',
    '4280': 'TIM & ADHESIVE', '4300': 'LID ATTACH (Old)',
    '4275': 'INDIUM TIM & ADHESIVE',
    '4272': 'CONFORMAL COATING',
    '4295': 'INDIUM REFLOW',
}

# FINAL REVISED STAGING LIMITS (TIM/ADHESIVE use Track In to Track Out, Indium uses Track Out to Track In):
STAGING_LIMITS = {
    ('4282', '4285'): {'limit_hours': 6, 'description': 'TIM-Adhesive Dispensing (Track In) to LID ATTACH (Track Out)', 'start_ts': 'TRACK_IN_TS', 'end_ts': 'TRACK_OUT_TS'},
    ('4280', '4300'): {'limit_hours': 6, 'description': 'TIM & ADHESIVE (Old) (Track In) to LID ATTACH (Track Out)', 'start_ts': 'TRACK_IN_TS', 'end_ts': 'TRACK_OUT_TS'},
    ('4276', '4277'): {'limit_hours': 12, 'description': 'STIFFENER DISPENSE (Track In) to STIFFENER ATTACH (Track Out)', 'start_ts': 'TRACK_IN_TS', 'end_ts': 'TRACK_OUT_TS'},

    ('4275', '4295'): {'limit_hours': 6, 'description': 'INDIUM TIM & ADHESIVE (Track Out) to INDIUM REFLOW (Track In)', 'start_ts': 'TRACK_OUT_TS', 'end_ts': 'TRACK_IN_TS'},
}

# ======================================================================
# CORE DATA PROCESSING AND STYLING FUNCTIONS
# ======================================================================

@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(uploaded_file):
    if uploaded_file is None:
        return None

    try:
        # 1. OPTIMIZATION: Define dtypes upfront to speed up reading and reduce memory
        optimized_dtypes = {
            'ASSEMBLY_LOT': 'str',
            'SPECNAME': 'str',
            'EQUIPMENT_LINE_NAME': 'str',
            'OPERATOR': 'str',
            'EDV': 'str',
        }

        # 2. OPTIMIZATION: Use specialized date parsing
        date_format = '%d/%m/%Y %H:%M:%S'
        date_cols = ['TRACK_IN_TS', 'TRACK_OUT_TS']

        # Required columns (VENDORNAME is handled conditionally later)
        required_cols_base = ['ASSEMBLY_LOT', 'SPECNAME', 'TRACK_IN_TS', 'TRACK_OUT_TS', 'EQUIPMENT_LINE_NAME', 'OPERATOR', 'TRACKOUT_QTY', 'EDV']

        file_name = uploaded_file.name

        # Define a lambda function for robust date parsing
        date_parser = lambda x: pd.to_datetime(x, format=date_format, errors='coerce')

        # --- Initial Data Load (Read all columns first to check for VENDORNAME variants) ---
        if file_name.endswith('.csv') or uploaded_file.type == "text/csv":
            file_string = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(
                file_string,
                dtype=optimized_dtypes,
                parse_dates=date_cols,
                date_parser=date_parser
            )
        elif file_name.endswith('.xlsx') or uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            # openpyxl must be installed for XLSX support
            df = pd.read_excel(
                uploaded_file,
                sheet_name=0,
                dtype=optimized_dtypes,
                parse_dates=date_cols,
                date_parser=date_parser
            )
        else:
            st.error("Unsupported file type. Please upload a CSV or XLSX file.")
            return None

        # --- 3. Robust VENDORNAME Column Check and Rename ---
        vendor_col_found = False
        possible_vendor_names = ['VENDORNAME', 'Vendorname', 'vendorname', 'Vendor_Name']

        # Check all column names for a match (case-insensitive search for robust data parsing)
        for col in df.columns:
            if col in possible_vendor_names:
                df.rename(columns={col: 'VENDORNAME'}, inplace=True)
                vendor_col_found = True
                break

        if not vendor_col_found:
            df['VENDORNAME'] = 'Unknown (Column Missing)'

        # --- 4. Ensure all base required columns are present ---
        missing_base_cols = [col for col in required_cols_base if col not in df.columns]
        if missing_base_cols:
             # This will catch missing base columns (like ASSEMBLY_LOT)
             raise ValueError(f"Required base columns not found: {missing_base_cols}")

        # --- 5. Final Cleaning and Type Setting ---
        # Select the columns needed for the analysis
        df = df[required_cols_base + ['VENDORNAME']].copy()

        df['EDV'] = df['EDV'].astype(str).str.strip().replace('nan', 'Unknown', regex=False)
        df['VENDORNAME'] = df['VENDORNAME'].astype(str).str.strip().replace('nan', 'Unknown', regex=False)

        # 6. OPTIMIZATION: Fast numeric conversion for TRACKOUT_QTY
        if 'TRACKOUT_QTY' in df.columns:
            # Convert to string first, then convert empty strings/nan strings to NaN, then to numeric
            temp_qty = pd.to_numeric(
                df['TRACKOUT_QTY'].astype(str).replace('', np.nan, regex=False),
                errors='coerce'
            )

            # Filter rows where quantity is invalid or <= 0
            df = df[temp_qty > 0].copy()

            # Add the cleaned numeric column back (Quantity defaults to 1 if NaN after filtering)
            df['Trackout Quantity'] = temp_qty.fillna(1).astype(int)
            df.drop(columns=['TRACKOUT_QTY'], errors='ignore', inplace=True)
        else:
            df['Trackout Quantity'] = 1

        return df
    except Exception as e:
        st.error(f"Error loading or parsing file: {e}")
        return None

# --- Cycle Time Calculation (NOW INCLUDES UCT) ---
@st.cache_data(show_spinner=False)
def calculate_cycle_time(df):
    df_ct = df.copy()

    # 1. Calculate Lot Cycle Time (Original: Time for the entire batch)
    # Columns are already datetime objects from load_data
    df_ct['Lot Cycle Time (min)'] = (df_ct['TRACK_OUT_TS'] - df_ct['TRACK_IN_TS']).dt.total_seconds() / 60

    # 2. Calculate Unit Cycle Time (NEW! Normalized: Time per Unit)
    # Quantity column is already cleaned and numeric
    df_ct['Unit Cycle Time (min/unit)'] = np.where(
        df_ct['Trackout Quantity'] > 1,
        df_ct['Lot Cycle Time (min)'] / df_ct['Trackout Quantity'],
        df_ct['Lot Cycle Time (min)']
    )

    # Filter out invalid or zero-duration cycles
    df_ct = df_ct[(df_ct['Lot Cycle Time (min)'].notna()) & (df_ct['Lot Cycle Time (min)'] > 0.1) ]

    df_ct.drop(columns=['TRACKOUT_QTY'], errors='ignore', inplace=True)

    return df_ct

# --- Staging Time Calculation ---
@st.cache_data(show_spinner=False)
def calculate_staging_time_violations(df_filtered, all_affected_lots_base):
    # Determine all unique steps involved in staging limits
    steps_to_check = set()
    for (start, end) in STAGING_LIMITS.keys():
        steps_to_check.add(start)
        steps_to_check.add(end)

    df_staging = df_filtered[
        (df_filtered['ASSEMBLY_LOT'].isin(all_affected_lots_base)) &
        (df_filtered['SPECNAME'].isin(steps_to_check))
    ].copy()

    # NOTE: Timestamps should already be datetime objects from load_data
    df_staging.dropna(subset=['TRACK_IN_TS', 'TRACK_OUT_TS'], how='any', inplace=True)

    # Pre-pivot: Get the required timestamp (IN or OUT) for each step in each lot
    required_pivot_cols = {}
    for (start_step, end_step), config in STAGING_LIMITS.items():
        # Using the start step's required timestamp
        required_pivot_cols[start_step] = config['start_ts']
        # Using the end step's required timestamp
        required_pivot_cols[end_step] = config['end_ts']

    staging_pivot_data = []
    for lot in df_staging['ASSEMBLY_LOT'].unique():
        lot_data = {'ASSEMBLY_LOT': lot}
        for step, ts_col in required_pivot_cols.items():
            # Filter the lot data for the specific step and get the required timestamp (min value)
            ts = df_staging[
                (df_staging['ASSEMBLY_LOT'] == lot) &
                (df_staging['SPECNAME'] == step)
            ][ts_col].min()

            # The column name in the pivot table is the step ID (e.g., '4275')
            lot_data[step] = ts
        staging_pivot_data.append(lot_data)

    staging_pivot = pd.DataFrame(staging_pivot_data)
    all_staging_records_dfs = [] # Collect DataFrames instead of dicts

    for (start_step, end_step), config in STAGING_LIMITS.items():
        limit_hours = config['limit_hours']
        description = config['description']
        start_ts_type = config['start_ts']
        end_ts_type = config['end_ts']

        start_col = start_step
        end_col = end_step

        if start_col in staging_pivot.columns and end_col in staging_pivot.columns:

            df_check = staging_pivot.dropna(subset=[start_col, end_col]).copy()

            # Calculate time difference using the pre-pivoted TS columns
            time_diff_hrs = (df_check[end_col] - df_check[start_col]).dt.total_seconds() / 3600

            # Filter out negative durations
            valid_indices = time_diff_hrs >= 0
            df_check = df_check[valid_indices].copy()

            if df_check.empty:
                continue

            df_check['Actual Time (hrs)'] = time_diff_hrs[valid_indices]

            # --- ⚡ Bolt Optimization: Vectorized Operations ---
            # Replaced iterrows() loop with vectorized pandas functions for a ~10-100x speedup
            # on large datasets. All operations are now performed on the entire DataFrame at once.

            df_check['Violation'] = df_check['Actual Time (hrs)'] > limit_hours

            # Vectorized timestamp string formatting
            start_ts_suffix = f" ({start_ts_type.replace('TRACK_', '').replace('_TS', '')})"
            end_ts_suffix = f" ({end_ts_type.replace('TRACK_', '').replace('_TS', '')})"

            df_check['Start Step TS'] = df_check[start_col].dt.strftime('%Y-%m-%d %H:%M') + start_ts_suffix
            df_check['End Step TS'] = df_check[end_col].dt.strftime('%Y-%m-%d %H:%M') + end_ts_suffix

            # Assign constant values
            df_check['Staging Check'] = description
            df_check['Limit (hrs)'] = limit_hours

            # Rename ASSEMBLY_LOT to Lot ID
            df_check.rename(columns={'ASSEMBLY_LOT': 'Lot ID'}, inplace=True)

            # Select and append the formatted DataFrame
            result_df = df_check[['Lot ID', 'Staging Check', 'Actual Time (hrs)', 'Limit (hrs)', 'Start Step TS', 'End Step TS', 'Violation']]
            all_staging_records_dfs.append(result_df)

    # If no staging checks could be performed, return an empty DataFrame
    if not all_staging_records_dfs:
        return pd.DataFrame()

    # Concatenate all results into a single DataFrame
    df_report = pd.concat(all_staging_records_dfs, ignore_index=True)

    if not df_report.empty:
        df_report = df_report.sort_values(['Staging Check', 'Actual Time (hrs)'], ascending=[True, False]).reset_index(drop=True)
        # Columns are already selected and ordered correctly from the loop

    return df_report

# --- Traceability Matrix Generation ---
@st.cache_data(show_spinner=False)
def generate_traceability_matrix(df, affected_lots_base):
    df_affected = df[df['ASSEMBLY_LOT'].isin(affected_lots_base)].copy()
    df_affected['SPECNAME'] = df_affected['SPECNAME'].astype(str)

    # Filter out staging-only steps from traceability if needed, but for simplicity we use CRITICAL_STEPS
    traceability_steps = {k: v for k, v in CRITICAL_STEPS.items() if k not in ['4280', '4300']}

    df_filtered = df_affected[df_affected['SPECNAME'].isin(traceability_steps.keys())].copy()

    traceability = df_filtered.pivot_table(
        index='ASSEMBLY_LOT',
        columns='SPECNAME',
        values='EQUIPMENT_LINE_NAME',
        aggfunc='first'
    ).reset_index()

    traceability = traceability.rename(columns={k: f"{traceability_steps.get(k, 'Unknown Step')} - {k}" for k in traceability.columns if k in traceability_steps.keys()})

    common_tools = {}
    for specname in traceability_steps.keys():
        col_name = f"{traceability_steps[specname]} - {specname}"
        if col_name in traceability.columns and not traceability[col_name].empty:
            mode = traceability[col_name].mode()
            if not mode.empty:
                common_tools[specname] = mode.iloc[0]

    # NEW: Get all unique machines for ALL critical steps used by the affected lots
    all_trace_machines = df_filtered.groupby(['SPECNAME', 'EQUIPMENT_LINE_NAME']).size().reset_index(name='count')
    all_trace_machines['Step_Machine'] = all_trace_machines['SPECNAME'].map(CRITICAL_STEPS) + " (" + all_trace_machines['SPECNAME'] + ") - Tool: " + all_trace_machines['EQUIPMENT_LINE_NAME']

    return traceability, common_tools, all_trace_machines

# --- Timeline Data Generation (FIXED FOR ROBUST FILTERING) ---
@st.cache_data(show_spinner=False)
def generate_timeline_data(df, affected_lots_base, step_id, selected_machine, affected_quantity_map, window=5):
    df_copy = df.copy() # Work on a copy

    # 1. Robust Type Casting and Cleaning (CRITICAL FIX FOR STRING MISMATCH)
    df_copy['SPECNAME'] = df_copy['SPECNAME'].astype(str).str.strip()
    df_copy['ASSEMBLY_LOT'] = df_copy['ASSEMBLY_LOT'].astype(str).str.strip()
    df_copy['EQUIPMENT_LINE_NAME'] = df_copy['EQUIPMENT_LINE_NAME'].astype(str).str.strip()

    step_id_clean = str(step_id).strip()
    selected_machine_clean = str(selected_machine).strip()

    # 2. Filtering by Machine and Step
    df_timeline = df_copy[
        (df_copy['EQUIPMENT_LINE_NAME'] == selected_machine_clean) &
        (df_copy['SPECNAME'] == step_id_clean)
    ].sort_values('TRACK_OUT_TS').reset_index(drop=True)

    if df_timeline.empty:
        # Fails if no lot ran on this machine at this step in the data
        return None, None

    # 3. Finding Affected Lots and Determining Timeline Window

    # Ensure affected_lots_base is also clean (lots in the map are already cleaned)
    affected_lots_base_clean = [lot.strip() for lot in affected_lots_base]

    affected_indices = df_timeline[
        df_timeline['ASSEMBLY_LOT'].isin(affected_lots_base_clean)
    ].index.tolist()

    if not affected_indices:
        # Fails if the affected lot list has no intersection with the machine data
        return None, None

    start_index = max(0, min(affected_indices) - window)
    end_index = min(len(df_timeline), max(affected_indices) + window + 1)

    timeline_window = df_timeline.iloc[start_index:end_index].copy()
    timeline_window['SEQUENCE_INDEX'] = timeline_window.index # Keep for charting Y-axis

    # 4. Quantity and Affected Status Mapping
    if 'Trackout Quantity' not in timeline_window.columns:
         timeline_window['Trackout Quantity'] = 1

    # Map the affected quantity (keys should be clean lot IDs)
    timeline_window['AFFECTED_QTY'] = timeline_window['ASSEMBLY_LOT'].map(affected_quantity_map).fillna(0).astype(int)

    # 5. Final Output Formatting
    timeline_window = timeline_window[[
        'ASSEMBLY_LOT', 'SEQUENCE_INDEX', 'Trackout Quantity',
        'TRACK_IN_TS', 'TRACK_OUT_TS', 'OPERATOR',
        'EQUIPMENT_LINE_NAME', 'SPECNAME', 'AFFECTED_QTY'
    ]].copy()

    timeline_window.rename(columns={
        'ASSEMBLY_LOT': 'Lot ID',
        'Trackout Quantity': 'Trackout Lot Quantity',
        'TRACK_IN_TS': 'Track In Timestamp',
        'TRACK_OUT_TS': 'Track Out Timestamp',
        'OPERATOR': 'Operator',
        'EQUIPMENT_LINE_NAME': 'Machine ID',
        'SPECNAME': 'Step ID',
        'AFFECTED_QTY': 'Affected Quantity'
    }, inplace=True)

    timeline_window['Affected'] = timeline_window['Affected Quantity'] > 0

    return timeline_window, affected_lots_base

# --- Cycle Time Comparison Generation (Uses UCT) ---
def generate_cycle_time_comparison_table(df_ct_base, df_ct_affected, selected_lots):

    # Calculate Baseline Avg UCT
    baseline_ct = df_ct_base.groupby('Process Step Name')['Unit Cycle Time (min/unit)'].mean().reset_index()
    baseline_ct.rename(columns={'Unit Cycle Time (min/unit)': 'Baseline Avg UCT (min/unit)'}, inplace=True)

    # Calculate Affected Lot Avg UCT
    affected_ct = df_ct_affected.groupby('Process Step Name')['Unit Cycle Time (min/unit)'].mean().reset_index()
    affected_ct.rename(columns={'Unit Cycle Time (min/unit)': 'Affected Lot Avg UCT (min/unit)'}, inplace=True)

    affected_count = df_ct_affected.groupby('Process Step Name')['ASSEMBLY_LOT'].nunique().reset_index()
    affected_count.rename(columns={'ASSEMBLY_LOT': 'Affected Lot Count'}, inplace=True)

    comparison_df = pd.merge(baseline_ct, affected_ct, on='Process Step Name', how='inner')
    comparison_df = pd.merge(comparison_df, affected_count, on='Process Step Name', how='inner')

    # Calculate UCT difference
    comparison_df['UCT Difference (min/unit)'] = comparison_df['Affected Lot Avg UCT (min/unit)'] - comparison_df['Baseline Avg UCT (min/unit)']
    comparison_df['UCT Difference (%)'] = (comparison_df['UCT Difference (min/unit)'] / comparison_df['Baseline Avg UCT (min/unit)']) * 100

    comparison_df.rename(columns={'Affected Lot Count': f'{len(selected_lots)} Selected Full Lot ID(s) Count'}, inplace=True)

    return comparison_df

# --- Lot-Level UCT Comparison Function ---
def generate_lot_uct_comparison(df_ct_base, df_ct_affected, baseline_comparison_df):
    """
    Compares the UCT of individual affected lots against the product-specific process step baseline UCT.
    """
    if baseline_comparison_df.empty:
        return pd.DataFrame()

    # 1. Get Baseline Averages (from the aggregated comparison table)
    baseline_map = baseline_comparison_df.set_index('Process Step Name')['Baseline Avg UCT (min/unit)'].to_dict()

    # Filter the affected lot data to the relevant steps
    df_lot_uct = df_ct_affected.copy()

    # 2. Group affected lots by Lot ID and Step to get the Lot's Avg UCT
    lot_avg_uct = df_lot_uct.groupby(['ASSEMBLY_LOT', 'Process Step Name'])['Unit Cycle Time (min/unit)'].mean().reset_index()
    lot_avg_uct.rename(columns={'Unit Cycle Time (min/unit)': 'Lot Avg UCT (min/unit)'}, inplace=True)

    # 3. Merge with Baseline Average using Process Step Name
    lot_avg_uct['Baseline Avg UCT (min/unit)'] = lot_avg_uct['Process Step Name'].map(baseline_map)

    # 4. Calculate Difference
    lot_avg_uct['UCT Difference (min/unit)'] = lot_avg_uct['Lot Avg UCT (min/unit)'] - lot_avg_uct['Baseline Avg UCT (min/unit)']
    lot_avg_uct['UCT Difference (%)'] = (lot_avg_uct['UCT Difference (min/unit)'] / lot_avg_uct['Baseline Avg UCT (min/unit)']) * 100

    # Pivot the table so steps are columns (easier to read)
    lot_comparison_pivot = lot_avg_uct.pivot_table(
        index='ASSEMBLY_LOT',
        columns='Process Step Name',
        values='UCT Difference (%)'
    ).reset_index()

    lot_comparison_pivot.rename(columns={'ASSEMBLY_LOT': 'Full Lot ID'}, inplace=True)

    return lot_comparison_pivot

# --- Pareto Analysis ---
@st.cache_data(show_spinner="Running Pareto Analysis...")
def generate_pareto_analysis(df_filtered, all_affected_lots_base, affected_lots_edv_df):

    # 1. Identify critical steps (excluding staging checks)
    critical_step_ids = [k for k in CRITICAL_STEPS.keys() if k not in ['4280', '4300']]

    # 2. Filter data to only include affected lots and critical steps
    df_pareto = df_filtered[
        (df_filtered['ASSEMBLY_LOT'].isin(all_affected_lots_base)) &
        (df_filtered['SPECNAME'].isin(critical_step_ids))
    ].copy()

    # 3. Determine the product name for each affected lot
    edv_map = affected_lots_edv_df.reset_index().set_index('Full Lot ID')['Product Name (EDV)'].to_dict()
    df_pareto['Product Name (EDV)'] = df_pareto['ASSEMBLY_LOT'].map(edv_map).fillna('Unknown')

    # 4. Find the unique machine used by each affected lot at each critical step
    # Group by Lot, Step, and Machine, then take the count (which will be 1 if rows are unique)
    df_analysis = df_pareto.groupby(['ASSEMBLY_LOT', 'SPECNAME', 'EQUIPMENT_LINE_NAME', 'Product Name (EDV)']).size().reset_index(name='Lot_Count')

    # 5. Aggregate: Group by the cause (Product, Step, Machine) and count affected lots
    df_pareto_summary = df_analysis.groupby(['Product Name (EDV)', 'SPECNAME', 'EQUIPMENT_LINE_NAME'])['Lot_Count'].sum().reset_index(name='Affected Lot Count')

    # 6. Add Process Step Name
    df_pareto_summary['Process Step Name'] = df_pareto_summary['SPECNAME'].map(CRITICAL_STEPS).fillna(df_pareto_summary['SPECNAME'])

    # 7. Create combined cause identifier (The Pareto Category)
    df_pareto_summary['Cause Category'] = (
        "[" + df_pareto_summary['Product Name (EDV)'] + "] " +
        df_pareto_summary['Process Step Name'] + " (" +
        df_pareto_summary['SPECNAME'] + ") @ " +
        df_pareto_summary['EQUIPMENT_LINE_NAME']
    )

    # 8. Calculate Pareto values
    df_pareto_summary = df_pareto_summary.sort_values(by='Affected Lot Count', ascending=False).reset_index(drop=True)
    total_affected_lots = df_pareto_summary['Affected Lot Count'].sum()

    df_pareto_summary['Percentage (%)'] = (df_pareto_summary['Affected Lot Count'] / total_affected_lots) * 100
    df_pareto_summary['Cumulative Percentage (%)'] = df_pareto_summary['Percentage (%)'].cumsum()

    return df_pareto_summary[['Cause Category', 'Affected Lot Count', 'Percentage (%)', 'Cumulative Percentage (%)']]

# --- Utility Function: Map Timestamp to Shift (UPDATED FOR 12HR SHIFT) ---
def map_timestamp_to_shift(ts):
    """Maps a datetime object to a standard 12-hour shift based on 6:30 boundaries (Day: 6:30 to 18:30)."""
    if pd.isna(ts):
        return 'Unknown'

    # Define boundary time (6:30 AM and 6:30 PM)
    boundary_am = ts.replace(hour=6, minute=30, second=0, microsecond=0)
    boundary_pm = ts.replace(hour=18, minute=30, second=0, microsecond=0)

    # Check if time is in the 6:30 AM to 6:30 PM range
    if ts >= boundary_am and ts < boundary_pm:
        # 06:30:00 to 18:29:59
        return 'Day Shift (06:30 to 18:30)'
    else:
        # 18:30:00 to 06:29:59 (Spans across midnight)
        return 'Night Shift (18:30 to 06:30)'

# --- Shift Handoff Check (Table 7) (UPDATED FOR 12HR SHIFT BOUNDARIES) ---
def check_shift_handoffs(df_ct_affected, selected_lots, step_id):
    """Checks if affected lots tracked out near 12-hour shift changes (6:30 AM/PM)."""

    df_handoff = df_ct_affected[
        df_ct_affected['SPECNAME'] == step_id
    ].copy()

    if df_handoff.empty:
        return pd.DataFrame()

    # Map track-out time to the shift it ended in
    df_handoff['Track Out Shift'] = df_handoff['TRACK_OUT_TS'].apply(map_timestamp_to_shift)

    # Define Handoff Windows (e.g., 30 minutes before/after boundary)

    def is_handoff_window(ts):
        if pd.isna(ts):
            return False

        # Shift boundaries are 6:30 AM (6, 30) and 6:30 PM (18, 30)
        boundaries = [(6, 30), (18, 30)]

        candidate_boundaries = []
        for hour, minute in boundaries:
            # Boundary on the current day
            candidate_boundaries.append(ts.replace(hour=hour, minute=minute, second=0, microsecond=0))
            # Boundary on the next day (e.g., if ts is 6:40 PM, check 6:30 AM tomorrow)
            candidate_boundaries.append((ts + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0))
            # Boundary on the previous day (e.g., if ts is 6:20 AM, check 6:30 PM yesterday)
            candidate_boundaries.append((ts - timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0))

        min_time_diff = min(abs(ts - ref_ts).total_seconds() / 60 for ref_ts in candidate_boundaries)

        return min_time_diff <= 30 # Within 30 minutes of 6:30 boundary

    df_handoff['Is Handoff Window'] = df_handoff['TRACK_OUT_TS'].apply(is_handoff_window)

    df_handoff = df_handoff[df_handoff['Is Handoff Window']].copy()

    report = df_handoff[['ASSEMBLY_LOT', 'TRACK_OUT_TS', 'Track Out Shift', 'OPERATOR', 'EQUIPMENT_LINE_NAME']].copy()
    report.rename(columns={
        'ASSEMBLY_LOT': 'Lot ID',
        'TRACK_OUT_TS': 'Track Out Timestamp',
        'OPERATOR': 'Operator ID',
        'EQUIPMENT_LINE_NAME': 'Machine ID'
    }, inplace=True)

    # Calculate time to nearest 6:30 boundary
    def calculate_min_time_diff(ts):
        # Time of day as a timedelta from midnight (0:00)
        time_of_day = timedelta(hours=ts.hour, minutes=ts.minute, seconds=ts.second)

        # Boundary times as timedelta from midnight (6:30 AM, 6:30 PM)
        boundaries_24h = [timedelta(hours=6, minutes=30), timedelta(hours=18, minutes=30)]

        min_diff_minutes = float('inf')

        for boundary in boundaries_24h:
            # Calculate difference on the current day
            diff_current = abs((time_of_day - boundary).total_seconds() / 60)
            min_diff_minutes = min(min_diff_minutes, diff_current)

            # Check for wrapping around midnight (next day's AM boundary, previous day's PM boundary)
            # 24 * 60 = 1440 minutes in a day
            diff_next_day = abs((time_of_day - (boundary - timedelta(days=1))).total_seconds() / 60)
            diff_prev_day = abs((time_of_day - (boundary + timedelta(days=1))).total_seconds() / 60)

            min_diff_minutes = min(min_diff_minutes, diff_next_day, diff_prev_day)

        return min_diff_minutes

    report['Time to Nearest 6:30 Boundary (min)'] = report['Track Out Timestamp'].apply(calculate_min_time_diff)

    return report.sort_values('Track Out Timestamp')

# --- Operator-Machine Co-Occurrence (Table 8) (NEW) ---
def generate_operator_machine_matrix(df_ct_affected, top_n=5):
    """Generates a matrix showing the number of affected lots per Operator/Machine combo."""

    # 1. Get Top N Suspicious Steps by Affected Lot Count
    step_counts = df_ct_affected['SPECNAME'].value_counts()
    top_steps = step_counts.head(top_n).index.tolist()

    df_matrix = df_ct_affected[
        df_ct_affected['SPECNAME'].isin(top_steps)
    ].copy()

    # 2. Get Top N Suspicious Operators (based on how many unique affected lots they handled)
    operator_counts = df_matrix.groupby('OPERATOR')['ASSEMBLY_LOT'].nunique().sort_values(ascending=False)
    top_operators = operator_counts.head(top_n).index.tolist()

    # 3. Filter for only top operators and relevant columns
    df_filtered = df_matrix[
        df_matrix['OPERATOR'].isin(top_operators)
    ]

    # 4. Create the pivot matrix: Rows=Operator, Columns=Machine ID (only for top steps)
    pivot_table_data = []

    for step in top_steps:
        step_df = df_filtered[df_filtered['SPECNAME'] == step].copy()

        # Get machine counts at this specific step
        machine_counts = step_df['EQUIPMENT_LINE_NAME'].value_counts()
        top_machines = machine_counts.head(top_n).index.tolist()

        # Pivot table for the step
        step_pivot = step_df[
            step_df['EQUIPMENT_LINE_NAME'].isin(top_machines)
        ].pivot_table(
            index='OPERATOR',
            columns='EQUIPMENT_LINE_NAME',
            values='ASSEMBLY_LOT',
            aggfunc='nunique', # Count unique lots
            fill_value=0
        )

        # Rename columns to include Step name
        step_pivot.columns = [f"{CRITICAL_STEPS.get(step, step)} @ {col}" for col in step_pivot.columns]

        pivot_table_data.append(step_pivot)

    if not pivot_table_data:
        return pd.DataFrame(), [], []

    # Concatenate all step pivot tables horizontally
    final_matrix = pd.concat(pivot_table_data, axis=1).fillna(0).astype(int)
    final_matrix.index.name = 'Operator ID'
    final_matrix = final_matrix.reset_index()

    return final_matrix, top_steps, top_operators

# --- Substrate Vendor Analysis (NEW) ---
@st.cache_data(show_spinner="Running Vendor Analysis...")
def generate_vendor_pareto(df_filtered, all_affected_lots_base):
    # Check if VENDORNAME column exists (it should, as the load_data guarantees it)
    if 'VENDORNAME' not in df_filtered.columns:
        return pd.DataFrame(), False

    # 1. Filter data to include only affected lots
    df_vendor = df_filtered[
        df_filtered['ASSEMBLY_LOT'].isin(all_affected_lots_base)
    ].copy()

    # We only need one row per lot to get the VENDORNAME
    df_vendor_unique = df_vendor.drop_duplicates(subset=['ASSEMBLY_LOT']).copy()

    # 2. Aggregate: Group by VENDORNAME and count unique affected lots
    df_pareto_summary = df_vendor_unique.groupby('VENDORNAME')['ASSEMBLY_LOT'].nunique().reset_index(name='Affected Lot Count')

    # 3. Calculate Pareto values
    df_pareto_summary = df_pareto_summary.sort_values(by='Affected Lot Count', ascending=False).reset_index(drop=True)
    total_affected_lots = df_pareto_summary['Affected Lot Count'].sum()

    df_pareto_summary['Percentage (%)'] = (df_pareto_summary['Affected Lot Count'] / total_affected_lots) * 100
    df_pareto_summary['Cumulative Percentage (%)'] = df_pareto_summary['Percentage (%)'].cumsum()

    df_pareto_summary.rename(columns={'VENDORNAME': 'Substrate Vendor Name'}, inplace=True)

    return df_pareto_summary, True

# --- Styling Functions ---
def style_staging_table(df_staging):
    def highlight_violation_row(row):
        is_violation = row['Violation']
        style = 'background-color: #ffcccc; font-weight: bold' if is_violation else ''
        return [style] * len(row)

    styled_df = df_staging.style.apply(highlight_violation_row, axis=1).format({
        'Actual Time (hrs)': '{:.1f}', 'Limit (hrs)': '{:.0f}',
    }).hide(subset=['Violation'], axis=1)

    return styled_df

def style_traceability_table(df, common_tools):
    def highlight_common(s):
        parts = s.name.split(' - ')
        if len(parts) > 1:
            step_id = parts[1]
            is_common = s == common_tools.get(step_id)
            return ['background-color: #38761d; color: white' if v and step_id in CRITICAL_STEPS else '' for v in is_common]
        return ['' for _ in s]
    return df.style.apply(highlight_common, axis=0)

def style_timeline_table(df_window):
    """
    Styles the timeline table, hides Sequence Index, and ensures full Track Out/In Timestamp is visible.
    """
    def highlight_affected_full_row(s):
        is_affected = s['Affected Quantity'] > 0
        style = 'font-weight: bold; background-color: #fce5cd;' if is_affected else ''
        # Cols to style: Lot ID, Timestamps, Operator, Quantities
        cols_to_style = ['Lot ID', 'Track In Timestamp', 'Track Out Timestamp', 'Operator', 'Affected Quantity', 'Trackout Lot Quantity']
        styles = [style if col in cols_to_style else '' for col in s.index]
        return styles

    # Format both timestamps to show full detail for sequencing
    styled_df = df_window.style.apply(highlight_affected_full_row, axis=1).format(
        {
            'Track In Timestamp': lambda t: t.strftime('%Y-%m-%d %H:%M:%S'),
            'Track Out Timestamp': lambda t: t.strftime('%Y-%m-%d %H:%M:%S'),
            'Affected Quantity': '{:,.0f}',
            'Trackout Lot Quantity': '{:,.0f}',
        }
    )

    # Columns to hide: Sequence Index, Affected flag, Machine ID, Step ID
    columns_to_hide = ['Sequence Index', 'Affected', 'Machine ID', 'Step ID']
    existing_cols_to_hide = [col for col in columns_to_hide if col in styled_df.columns]

    return styled_df.hide(subset=existing_cols_to_hide, axis=1)

def style_comparison_table(df, selected_lots):
    return df.style.format({
        'Baseline Avg UCT (min/unit)': '{:.3f}',
        'Affected Lot Avg UCT (min/unit)': '{:.3f}',
        'UCT Difference (min/unit)': '{:+.3f}',
        'UCT Difference (%)': '{:+.1f}%'
    }).apply(lambda x: ['background-color: #fce5cd; font-weight: bold;' if v > 0 else '' for v in x],
             subset=['UCT Difference (%)'], axis=0)

# --- Styling Function for the Pivot Table ---
def style_lot_comparison_table(df):
    def highlight_positive_difference(series):
        is_positive = series > 0
        return ['background-color: #fce5cd; font-weight: bold;' if v else '' for v in is_positive]

    # Format the entire table, excluding the index columns ('Full Lot ID', 'Product Name (EDV)')
    # Note: Index columns are excluded by default when applying `subset=df.columns`
    return df.style.format('{:+.1f}%').apply(highlight_positive_difference, axis=0)

# --- Styling Function for Pareto Table ---
def style_pareto_table(df):
    return df.style.format({
        'Affected Lot Count': '{:,.0f}',
        'Percentage (%)': '{:.1f}%',
        'Cumulative Percentage (%)': '{:.1f}%'
    })

# --- Styling Function for Vendor Table (Reuse Pareto style) ---
def style_vendor_table(df):
    return df.style.format({
        'Affected Lot Count': '{:,.0f}',
        'Percentage (%)': '{:.1f}%',
        'Cumulative Percentage (%)': '{:.1f}%'
    })

# ======================================================================
# STREAMLIT PAGES
# ======================================================================

def get_affected_lot_edv_info(df, affected_lots):
    """Retrieves EDV/Product Name and quantity for each affected lot (with robust KeyError fix)."""

    # 1. IMMEDIATE CHECK for empty data or empty affected list (ROBUST FIX)
    if df is None or df.empty or not affected_lots:
        # Return an empty, but correctly structured, DataFrame immediately
        return pd.DataFrame({
            'Full Lot ID': [],
            'Product Name (EDV)': [],
            'Affected Quantity': []
        }).set_index('Full Lot ID')

    # 2. Filter the main DataFrame for the affected lots
    df_affected = df[df['ASSEMBLY_LOT'].isin(affected_lots)].copy()

    data = []

    if df_affected.empty:
        # If the lots are not found in the data, list them with 'NOT FOUND' EDV
        for lot_id in affected_lots:
            data.append({
                'Full Lot ID': lot_id,
                'Product Name (EDV)': 'NOT FOUND IN DATA',
                'Affected Quantity': st.session_state.affected_quantity_map.get(lot_id, 0)
            })
    else:
        # Get the unique EDV for each lot found in the data
        edv_map = df_affected.groupby('ASSEMBLY_LOT')['EDV'].agg(lambda x: ', '.join(x.unique())).to_dict()

        for lot_id in affected_lots:
            data.append({
                'Full Lot ID': lot_id,
                # Use the map result if available, otherwise mark as 'NOT FOUND'
                'Product Name (EDV)': edv_map.get(lot_id, 'NOT FOUND IN DATA'),
                'Affected Quantity': st.session_state.affected_quantity_map.get(lot_id, 0)
            })

    # 3. Create the DataFrame and set index (now guaranteed to have data if 'affected_lots' wasn't empty)
    return pd.DataFrame(data).set_index('Full Lot ID')

# --- Existing render_upload_page ---
def render_upload_page(df, affected_quantity_map, all_affected_lots_base, affected_lots_edv_df):
    st.title("⬆️ Data Upload & Preview")
    st.markdown("---")

    if df is not None:
        st.subheader("Data Cleaning Summary")
        col_count, col_rows, col_lots = st.columns(3)
        col_count.metric("Total Records Loaded", f"{len(df):,}")
        col_rows.metric("Unique Process Steps", f"{df['SPECNAME'].nunique()}")
        col_lots.metric("Unique Lots in Data", f"{df['ASSEMBLY_LOT'].nunique()}")

        st.subheader("Affected Lots Entered")
        st.info(f"**Total Affected Lots:** {len(all_affected_lots_base)} | **Total Affected Units:** {sum(affected_quantity_map.values()):,}")

        if all_affected_lots_base:

            # Show the table with the new EDV column (Fixed height for preview)
            st.dataframe(
                affected_lots_edv_df,
                use_container_width=True,
                height=250, # Keep fixed height for sidebar-dependent preview table
                column_config={
                    "Affected Quantity": st.column_config.NumberColumn(
                        "Affected Quantity",
                        format="%d",
                        width="small"
                    )
                }
            )

        st.subheader("Raw Data Sample Preview (First 100 Rows)")
        with st.container(height=300):
            st.dataframe(df.head(100), use_container_width=True)
    else:
        st.info("Upload data in the sidebar to view the preview.")


def render_staging_page(df_filtered, all_affected_lots_base):
    st.header("1. ⏳ Staging Time Analysis (Risk Assessment)")
    st.divider()

    if df_filtered is None:
        st.error("Data filtering incomplete.")
        return

    with st.spinner("Calculating Staging Time Violations..."):
        staging_violations_df = calculate_staging_time_violations(df_filtered, all_affected_lots_base)

    st.subheader("Table 1: Material Staging Time Report")
    st.markdown(r"Checks elapsed time against limits. **Violations are highlighted in red.**")

    if not staging_violations_df.empty:
        num_violations = staging_violations_df['Violation'].sum()

        if num_violations > 0:
            st.error(f"🚨 **{num_violations} Staging Time Violation(s) Found!**")
        else:
            st.success("✅ All checked lots passed the staging time requirements.")

        # FULL EXPANDED TABLE
        st.dataframe(style_staging_table(staging_violations_df), use_container_width=True)

    else:
        st.warning("No data available to calculate staging times for the affected lots at the required steps.")


def render_traceability_page(df_filtered, all_affected_lots_base):
    st.header("2. 🔍 Traceability & Commonality Analysis (Machine Focus)")
    st.divider()

    if df_filtered is None:
        st.error("Data filtering incomplete.")
        return

    with st.spinner("Generating Traceability Matrix..."):
        traceability_df, common_tools, all_trace_machines = generate_traceability_matrix(df_filtered, all_affected_lots_base)

    st.subheader("Table 2: Tool Traceability Matrix")
    st.markdown("The most common tool for a critical step across all affected lots is highlighted in **green**, indicating a potential suspect machine.")

    if not traceability_df.empty:
        # FULL EXPANDED TABLE
        st.dataframe(style_traceability_table(traceability_df, common_tools), use_container_width=True)
    else:
        st.warning("No traceability data found for the input lots in the critical steps within the current product filter.")

    return traceability_df, common_tools, all_trace_machines


def render_timeline_page(df_filtered, affected_lots_base, affected_quantity_map, traceability_df, common_tools, all_trace_machines):
    st.header("3. ⏱️ Time-Series Proximity (Timeline) Analysis")
    st.divider()

    if df_filtered is None:
        st.error("Data filtering incomplete.")
        return

    if traceability_df.empty:
        st.warning("Timeline analysis requires traceability data. Please check the 'Traceability & Commonality Analysis' section above.")
        return

    # Prepare options to include ALL step/machine combinations found
    timeline_options = {}
    for index, row in all_trace_machines.iterrows():
        option_label = row['Step_Machine']
        step_id = row['SPECNAME']
        machine_id = row['EQUIPMENT_LINE_NAME']
        timeline_options[option_label] = (step_id, machine_id)

    if common_tools:
        # Sort options: put the common tool options first
        sorted_labels = []
        first_option = None
        for step_id in common_tools.keys():
            step_name = CRITICAL_STEPS.get(step_id, f"Unknown Step {step_id}")
            machine_id = common_tools[step_id]
            common_label = f"{step_name} ({step_id}) - Tool: {machine_id}"

            # Find the exact original label that matches the common tool
            exact_original_label = next((label for label, (s, m) in timeline_options.items() if s == step_id and m == machine_id), None)

            if exact_original_label:
                # Add the common label with the suffix to the start
                sorted_labels.append(exact_original_label + " (COMMON)")
                if first_option is None:
                    first_option = exact_original_label + " (COMMON)"

        # Add remaining unique options (without the common ones)
        existing_labels = {label.replace(" (COMMON)", "") for label in sorted_labels}
        for label in sorted(timeline_options.keys()):
            if label not in existing_labels:
                sorted_labels.append(label)

        # Re-map keys after adding '(COMMON)' suffix
        final_timeline_options = {}
        for label in sorted_labels:
            original_label = label.replace(" (COMMON)", "")
            # Get the step_id and machine_id from the original map
            try:
                # Robust parsing (since the original label format is guaranteed by all_trace_machines logic)
                step_id_match = original_label.split(" (")[1].split(")")[0]
                machine_id_match = original_label.split(" - Tool: ")[1]
            except IndexError:
                # Should not happen if all_trace_machines is correct, but handles unexpected formats
                continue

            final_timeline_options[label] = (step_id_match, machine_id_match)
            if first_option is None:
                first_option = label


        with st.container():
            selected_label = st.selectbox(
                "Select Process Step and Tool to Analyze:",
                options=list(final_timeline_options.keys()),
                index=sorted_labels.index(first_option) if first_option in sorted_labels else 0,
                key='timeline_select_key'
            )

        selected_step, selected_machine = final_timeline_options[selected_label]

        st.info(f"Focused Analysis on Tool: **{selected_machine}** (Sequence: 5 lots before/after first/last affected lot).")

        with st.spinner(f"Generating timeline for {selected_machine}..."):
            # Uses the newly robustified generate_timeline_data
            timeline_data, _ = generate_timeline_data(
                df_filtered, affected_lots_base, selected_step, selected_machine, affected_quantity_map
            )

        if timeline_data is not None:

            # --- Figure 1: Timeline Chart ---
            st.subheader("Figure 1: Production Timeline Chart")
            st.markdown("The size and color of the points represent the **Affected Quantity**. Context lots are in gray.")

            chart_base = alt.Chart(timeline_data).encode(
                x=alt.X('Track Out Timestamp', title='Track Out Time', axis=alt.Axis(format='%m/%d %H:%M')),

                # FIX FOR ValueError: Explicitly define the type as quantitative ('Q')
                y=alt.Y('SEQUENCE_INDEX:Q', title='Production Sequence (Index)'),

                tooltip=['Lot ID', alt.Tooltip('Track Out Timestamp', format='%Y-%m-%d %H:%M:%S', title='Track Out Time'), 'Operator', 'Affected Quantity']
            )

            chart_context = chart_base.transform_filter(alt.FieldRangePredicate('Affected Quantity', [0, 0])).mark_point(filled=True, size=50, color='lightgray')
            chart_affected = chart_base.transform_filter(alt.FieldRangePredicate('Affected Quantity', [1, None])).mark_point(filled=True).encode(
                color=alt.Color('Affected Quantity', scale=alt.Scale(scheme='reds', domainMin=1), legend=alt.Legend(title="Affected Qty")),
                size=alt.Size('Affected Quantity', scale=alt.Scale(range=[50, 400], domainMin=1), legend=None),
            )

            timeline_chart = (chart_context + chart_affected).properties(
                title=f"Production Timeline for {selected_machine}"
            ).interactive()

            st.altair_chart(timeline_chart, use_container_width=True)

            # --- Table 3: Detailed Timeline Table (Updated to show Track Out Lot Quantity) ---
            st.subheader("Table 3: Detailed Timeline Lot Sequence")
            st.markdown("Affected lots are highlighted in **orange/bold**. Sequence is indicated by the precise **Track Out Timestamp**.")

            # FULL EXPANDED TABLE
            timeline_df_styled = style_timeline_table(timeline_data)
            st.dataframe(timeline_df_styled, use_container_width=True)

        else:
            st.warning(f"Could not generate timeline for the selected tool **{selected_machine}**.")
    else:
        st.warning("No tool data found across the critical process steps for the affected lots to perform a timeline analysis.")


def render_pareto_page(df_filtered, all_affected_lots_base, affected_lots_edv_df):
    st.header("4. 🎯 Failure Prioritization (Pareto Analysis)")
    st.divider()

    if df_filtered is None or not all_affected_lots_base:
        st.error("Data or affected lot list is incomplete.")
        return

    pareto_df = generate_pareto_analysis(df_filtered, all_affected_lots_base, affected_lots_edv_df)

    st.subheader("Figure 2: Pareto Chart of Causes (by Affected Lot Count)")
    st.markdown("Identifies the 'Vital Few' Product/Machine combinations contributing most to the affected lots. **Focus your investigation on the categories on the far left.**")

    if not pareto_df.empty:
        # Altair chart for Pareto
        base = alt.Chart(pareto_df).encode(
            x=alt.X('Cause Category', sort="-y", axis=alt.Axis(labelAngle=-45, title="Product/Step/Machine Combination")),
        )

        # Bar chart for Affected Lot Count
        bar = base.mark_bar().encode(
            y=alt.Y('Affected Lot Count', axis=alt.Axis(title="Affected Lot Count", titleColor="#4C78A8")),
            color=alt.value("#4C78A8"),
            tooltip=['Cause Category', 'Affected Lot Count']
        )

        # Line chart for Cumulative Percentage
        line = base.mark_line(point=True, color="orange").encode(
            y=alt.Y('Cumulative Percentage (%)', axis=alt.Axis(title="Cumulative Percentage (%)", titleColor="orange")),
            tooltip=['Cause Category', alt.Tooltip('Cumulative Percentage (%)', format='.1f')]
        )

        # Combine the charts and add title
        chart = alt.layer(bar, line).resolve_scale(
            y='independent'  # Allows two separate Y-axes
        ).properties(
            title="Pareto Analysis: Affected Lots by Cause Category"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # --- Table 4: Pareto Data Table ---
        st.subheader("Table 4: Pareto Prioritization Data")
        st.markdown("The table summarizes the contribution of each cause. The first few rows are your highest priority.")

        st.dataframe(style_pareto_table(pareto_df), use_container_width=True)

        # Decision Helper
        pareto_80 = pareto_df[pareto_df['Cumulative Percentage (%)'] <= 80]
        if not pareto_80.empty:
            st.info(f"💡 **Decision Focus:** The top {len(pareto_80)} categories are responsible for over 80% of the affected lots. The top cause is: **{pareto_80['Cause Category'].iloc[0]}**.")
    else:
        st.warning("Pareto analysis skipped: Insufficient data to group causes.")


# --- UCT Re-factored Rendering Functions ---

def render_uct_box_plot(df_edv_base, edv):
    st.subheader(f"Figure 3: Process Unit Cycle Time (UCT) Baseline Distribution ({edv})")

    ct_steps_for_analysis = {k: v for k, v in CRITICAL_STEPS.items() if k not in ['4280', '4300']}
    steps_with_data = df_edv_base['Process Step Name'].unique()
    step_order = [CRITICAL_STEPS[k] for k in ct_steps_for_analysis.keys() if CRITICAL_STEPS[k] in steps_with_data]

    st.markdown(f"Shows **UCT (min/unit)** distribution for **{edv}** lots across critical steps. **Y-axis is logarithmic**.")

    X_ENCODING = alt.X('Process Step Name', axis=alt.Axis(labelAngle=-45, title='Process Step'), sort=step_order, type='nominal')
    Y_ENCODING = alt.Y('Unit Cycle Time (min/unit)', title='Unit Cycle Time (Minutes/Unit) [Log Scale]', scale=alt.Scale(type="log", domainMin=0.001))

    ct_chart = alt.Chart(df_edv_base).mark_boxplot(extent="min-max", size=25, opacity=0.8 ).encode(
        x=X_ENCODING, y=Y_ENCODING, color=alt.value("darkblue"),
        tooltip=['Process Step Name', 'EQUIPMENT_LINE_NAME', alt.Tooltip('Unit Cycle Time (min/unit)', format='.3f', title='UCT (min/unit)')]
    ).properties(
        title=f"Process Unit Cycle Time Baseline Distribution for {edv}"
    ).interactive()

    st.altair_chart(ct_chart, use_container_width=True)

def render_aggregate_uct_table(comparison_df, edv_lots):
    st.subheader(f"Table 5: Unit Cycle Time (UCT) Comparison (Affected Lots Avg vs. Baseline Avg)")
    if not comparison_df.empty:
        st.markdown(f"Compares Avg **UCT (min/unit)** of the **{len(edv_lots)}** selected lots against the product-specific baseline. **Positive differences (orange)** mean the affected group was slower.")
        # FULL EXPANDED TABLE
        st.dataframe(style_comparison_table(comparison_df, edv_lots), use_container_width=True)
    else:
        st.warning("Aggregate UCT comparison analysis skipped: No valid comparison data found.")

def render_individual_uct_table(lot_comparison_pivot):
    st.subheader("Table 6.1: Individual Lot UCT Performance vs. Baseline (Current Product)")
    if not lot_comparison_pivot.empty:
        st.markdown(r"Shows the **percentage difference** in UCT for each selected lot compared to the product-specific baseline average. **Positive values (orange)** indicate the specific lot ran slower than the baseline average.")
        # Ensure 'Full Lot ID' is set as index before styling
        styled_df = style_lot_comparison_table(lot_comparison_pivot.set_index('Full Lot ID'))

        # FULL EXPANDED TABLE
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("Individual lot UCT comparison skipped: Insufficient data to compare individual lots against the baseline.")


def orchestrate_uct_analysis(df_filtered, selected_lots, affected_lots_edv_df):

    unique_edvs = affected_lots_edv_df['Product Name (EDV)'].unique()

    # Filter out 'NOT FOUND' EDVs to ensure valid comparisons
    valid_edvs = sorted([edv for edv in unique_edvs if edv != 'NOT FOUND IN DATA'])

    if not valid_edvs:
        st.warning("UCT Analysis skipped: No affected lots found with a valid Product Name (EDV) in the filtered data.")
        return pd.DataFrame()

    final_lot_comparison_summary = []

    # Initialize the critical steps for filtering UCT data
    ct_steps_for_analysis = {k: v for k, v in CRITICAL_STEPS.items() if k not in ['4280', '4300']}

    with st.spinner("Performing Product-Specific UCT Analysis..."):

        # 1. Calculate UCT for all filtered data once
        df_ct = calculate_cycle_time(df_filtered)
        df_ct_filtered = df_ct[df_ct['SPECNAME'].isin(ct_steps_for_analysis.keys())].copy()

        if df_ct_filtered.empty:
            st.warning("UCT Analysis skipped: No valid production data found for the critical process steps.")
            return pd.DataFrame()

        df_ct_filtered['Process Step Name'] = df_ct_filtered['SPECNAME'].map(CRITICAL_STEPS).fillna(df_ct_filtered['SPECNAME'])

        st.header("5. 📈 Unit Cycle Time (UCT) Variability Analysis")
        st.divider()

        # 2. Loop through each Product (EDV) for analysis
        for i, edv in enumerate(valid_edvs):
            st.subheader(f"📊 {i+1}. UCT Analysis for Product: **{edv}**")

            # --- Product-Specific Data Sub-setting ---
            # Lots belonging to the current EDV that are also selected in the sub-filter
            edv_lots = affected_lots_edv_df[
                affected_lots_edv_df['Product Name (EDV)'] == edv
            ].index.intersection(pd.Index(selected_lots)).tolist()

            if not edv_lots:
                st.warning(f"No selected affected lots found for product {edv}. Skipping this section.")
                continue

            # Data containing only the current EDV (for baseline)
            df_edv_base = df_ct_filtered[df_ct_filtered['EDV'] == edv].copy()

            # Affected data for this EDV
            df_edv_affected = df_edv_base[df_edv_base['ASSEMBLY_LOT'].isin(edv_lots)].copy()

            if df_edv_base.empty or df_edv_affected.empty:
                st.warning(f"Skipping analysis for {edv}: Insufficient baseline or affected data.")
                continue

            # --- RENDER/CALCULATE PER PRODUCT ---

            # a) Calculate Aggregate Comparison (Table 5)
            comparison_df = generate_cycle_time_comparison_table(df_edv_base, df_edv_affected, edv_lots)

            # b) Calculate Individual Lot Comparison (Table 6 - for this product)
            lot_comparison_pivot = generate_lot_uct_comparison(df_edv_base, df_edv_affected, comparison_df)

            # c) Render Figure 3 (Box Plot)
            render_uct_box_plot(df_edv_base, edv)

            # d) Render Table 5 (Aggregate)
            render_aggregate_uct_table(comparison_df, edv_lots)

            # e) Render Table 6.1 (Individual Lot - for this product)
            render_individual_uct_table(lot_comparison_pivot)

            # 3. Compile Summary Data for Final Table 6
            if not lot_comparison_pivot.empty:
                # Add a column for product grouping before melting
                lot_comparison_pivot['Product Name (EDV)'] = edv

                # Convert the wide pivot table to a long format for final compilation
                lot_comparison_long = lot_comparison_pivot.melt(
                    id_vars=['Full Lot ID', 'Product Name (EDV)'],
                    var_name='Process Step Name',
                    value_name='UCT Difference (%)'
                ).dropna(subset=['UCT Difference (%)'])

                final_lot_comparison_summary.append(lot_comparison_long)

            st.markdown("---")

    # 4. Final Compilation and Summary Table (Consolidated Table 6)
    if final_lot_comparison_summary:
        summary_df = pd.concat(final_lot_comparison_summary, ignore_index=True)
        return summary_df.pivot_table(
            index=['Full Lot ID', 'Product Name (EDV)'],
            columns='Process Step Name',
            values='UCT Difference (%)'
        ).reset_index()
    else:
        return pd.DataFrame()


def render_cycle_time_page(df_filtered, selected_lots, affected_lots_edv_df):

    if df_filtered is None:
        st.error("Data filtering incomplete.")
        return

    # --- Orchestrate Product-Specific Analysis and Compile Summary ---
    summary_pivot_df = orchestrate_uct_analysis(df_filtered, selected_lots, affected_lots_edv_df)

    # --- Render Final Summary Table (Consolidated Table 6) ---
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
    st.subheader("Final Summary: Consolidated Individual Lot UCT Performance")
    st.markdown("### Table 6: Consolidated Individual Lot UCT Performance vs. Product Baseline")

    if not summary_pivot_df.empty:
        st.markdown(r"This table consolidates the **percentage difference** in UCT for all selected lots across **all analyzed products**. **Positive values (orange)** indicate slower performance than the respective product's baseline.")
        # Set index for styling
        summary_pivot_df = summary_pivot_df.set_index(['Full Lot ID', 'Product Name (EDV)']).copy()

        # FULL EXPANDED TABLE
        st.dataframe(style_lot_comparison_table(summary_pivot_df), use_container_width=True)
    else:
        st.info("Consolidated UCT summary table skipped: No valid product-specific UCT data was generated.")

    return

# --- NEW: Human Factor Analysis Page ---
def render_human_factor_page(df_filtered, selected_lots, affected_lots_edv_df):
    st.header("6. 👥 Human Factor Analysis (The 'Who' and 'When')")
    st.divider()

    if df_filtered is None or not selected_lots:
        st.error("Data or selected lot list is incomplete.")
        return

    # 1. Prepare UCT data (requires UCT calculation)
    df_ct = calculate_cycle_time(df_filtered)

    # Filter UCT data to only the selected affected lots
    df_ct_affected = df_ct[
        df_ct['ASSEMBLY_LOT'].isin(selected_lots)
    ].copy()

    if df_ct_affected.empty:
        st.warning("No cycle time data found for the selected affected lots.")
        return

    # Filter to only the top cause step from the Pareto analysis (if available)
    # We will use the most frequently occurring step in the affected lots for focus
    most_frequent_step_id = df_ct_affected['SPECNAME'].value_counts().idxmax()
    step_name = CRITICAL_STEPS.get(most_frequent_step_id, most_frequent_step_id)

    st.info(f"Focused analysis on the most frequent affected step: **{step_name} ({most_frequent_step_id})**")

    # --- A. Figure 4: Operator UCT Variability (Box Plot) ---
    st.subheader(f"Figure 4: Operator UCT Variability at Step {step_name}")
    st.markdown("Compares **Unit Cycle Time (UCT)** across different operators at the most critical step. Look for operators with higher median lines or much wider box plots. **The Y-axis is logarithmic.**")

    df_op_uct = df_ct_affected[df_ct_affected['SPECNAME'] == most_frequent_step_id].copy()

    X_OP = alt.X('OPERATOR', title='Operator ID', sort='descending')
    Y_OP = alt.Y('Unit Cycle Time (min/unit)', title='UCT (Minutes/Unit) [Log Scale]', scale=alt.Scale(type="log", domainMin=0.001))

    op_chart = alt.Chart(df_op_uct).mark_boxplot(extent="min-max", size=40).encode(
        x=X_OP, y=Y_OP,
        color=alt.Color('OPERATOR', legend=None),
        tooltip=['OPERATOR', alt.Tooltip('Unit Cycle Time (min/unit)', format='.3f', title='UCT (min/unit)'), 'ASSEMBLY_LOT']
    ).properties(
        title=f"Operator Performance Distribution at {step_name}"
    ).interactive()

    st.altair_chart(op_chart, use_container_width=True)

    # --- B. Table 7: Shift Handoff Proximity Check (UPDATED FOR 12HR SHIFT) ---
    st.subheader(f"Table 7: Shift Handoff Proximity Check (12-Hour Shift Boundaries: **6:30 AM/PM**)")

    with st.spinner("Checking affected lots near shift boundaries..."):
        # Uses the newly defined 6:30 AM/PM logic
        handoff_df = check_shift_handoffs(df_ct_affected, selected_lots, most_frequent_step_id)

    if not handoff_df.empty:
        st.markdown(f"Lots that tracked out within **30 minutes** of a major shift change (6:30 AM or 6:30 PM). A high count here suggests **handoff procedure failure.**")
        st.dataframe(handoff_df, use_container_width=True)
        num_handoff = len(handoff_df)
        total_affected = len(df_ct_affected[df_ct_affected['SPECNAME'] == most_frequent_step_id]['ASSEMBLY_LOT'].unique())
        if total_affected > 0 and num_handoff > 0.1 * total_affected:
             st.warning(f"🚨 **Actionable Alert:** {num_handoff} out of {total_affected} affected lots ({num_handoff/total_affected:.1%}) occurred near a handoff window. Investigate shift change procedures immediately.")
    else:
        st.info("No affected lots tracked out within the 30-minute shift handoff window at the critical step.")

    # --- C. Table 8: Operator-Machine Co-Occurrence ---
    st.subheader("Table 8: Operator-Machine Co-Occurrence Matrix")

    with st.spinner("Generating Operator-Machine matrix..."):
        matrix_df, top_steps, top_operators = generate_operator_machine_matrix(df_ct_affected)

    if not matrix_df.empty:
        st.markdown("Shows the count of unique affected lots for combinations of the most frequent operators and the machine they used at the most affected steps.")
        st.dataframe(matrix_df.set_index('Operator ID').style.background_gradient(cmap='Reds'), use_container_width=True)
        st.info(f"💡 **Insight:** Look for high counts (darker red). If one operator has high counts across *multiple* machines, the root cause is **operator skill/procedure**. If one machine has high counts across *multiple* operators, the root cause is the **machine itself**.")
    else:
        st.info("Not enough data to generate the Operator-Machine Co-Occurrence Matrix (Need at least 2 different top operators/machines).")

    return

# --- NEW: Substrate Vendor Analysis Page ---
def render_vendor_analysis_page(df_filtered, all_affected_lots_base):
    st.header("7. 🏭 Substrate Vendor Analysis (Material Input)")
    st.divider()

    if df_filtered is None or not all_affected_lots_base:
        st.error("Data or affected lot list is incomplete.")
        return

    vendor_pareto_df, vendor_col_exists = generate_vendor_pareto(df_filtered, all_affected_lots_base)

    if not vendor_col_exists:
        st.warning("Skipped: The required column **'VENDORNAME'** was not found in the uploaded data.")
        return

    st.subheader("Figure 5: Affected Lot Count by Substrate Vendor")
    st.markdown("This bar chart prioritizes substrate vendors based on the number of affected lots they supplied. **Focus your material investigation on the highest bars.**")

    if not vendor_pareto_df.empty:

        # Altair chart for Vendor Pareto
        base = alt.Chart(vendor_pareto_df).encode(
            x=alt.X('Substrate Vendor Name', sort="-y", title="Substrate Vendor Name"),
        )

        # Bar chart for Affected Lot Count
        bar = base.mark_bar().encode(
            y=alt.Y('Affected Lot Count', axis=alt.Axis(title="Affected Lot Count")),
            color=alt.value("darkgreen"),
            tooltip=['Substrate Vendor Name', 'Affected Lot Count', alt.Tooltip('Percentage (%)', format='.1f')]
        ).properties(
            title="Affected Lot Count by Substrate Vendor"
        ).interactive()

        st.altair_chart(bar, use_container_width=True)

        # --- Table 9: Vendor Pareto Data Table ---
        st.subheader("Table 9: Substrate Vendor Prioritization Data")
        st.markdown("The table summarizes the contribution of each vendor. The top row is your highest material investigation priority.")

        st.dataframe(style_vendor_table(vendor_pareto_df), use_container_width=True)

        # Decision Helper
        top_vendor = vendor_pareto_df['Substrate Vendor Name'].iloc[0]
        top_percentage = vendor_pareto_df['Percentage (%)'].iloc[0]

        if top_percentage > 50:
            st.error(f"🚨 **Actionable Alert:** Vendor **{top_vendor}** is responsible for {top_percentage:.1f}% of the affected lots. This strongly suggests a **Material Input (Substrate) Root Cause**. Raise an alert to the Quality team for this vendor.")
        elif top_percentage > 25:
             st.warning(f"⚠️ **Focus:** Vendor **{top_vendor}** accounts for {top_percentage:.1f}% of affected lots. This is a primary suspect for material-related issues.")
        else:
             st.info("The affected lots are broadly distributed across multiple vendors, suggesting the root cause is less likely to be vendor-specific material quality.")
    else:
        st.info("Substrate Vendor analysis skipped: No vendor data available or all lots are from an 'Unknown' vendor.")

    return

# --- Combined Analysis Page ---
def render_combined_analysis_page(df_filtered, all_affected_lots_base, selected_lots, affected_quantity_map, selected_edv, affected_lots_edv_df):
    st.title("Comprehensive Analysis Report")
    st.markdown("This page combines all the key analyses into one scrollable view.")
    st.markdown("---")

    # 1. Staging Time Analysis (Section 1)
    render_staging_page(df_filtered, all_affected_lots_base)
    st.markdown("---")

    # 2. Traceability Analysis (Section 2)
    traceability_df, common_tools, all_trace_machines = render_traceability_page(df_filtered, all_affected_lots_base)
    st.markdown("---")

    # 3. Timeline Analysis (Section 3)
    render_timeline_page(df_filtered, selected_lots, affected_quantity_map, traceability_df, common_tools, all_trace_machines)
    st.markdown("---")

    # 4. Pareto Analysis (Section 4)
    render_pareto_page(df_filtered, all_affected_lots_base, affected_lots_edv_df)
    st.markdown("---")

    # 5. Cycle Time Analysis (Section 5)
    render_cycle_time_page(df_filtered, selected_lots, affected_lots_edv_df)
    st.markdown("---")

    # 6. Human Factor Analysis (Section 6)
    render_human_factor_page(df_filtered, selected_lots, affected_lots_edv_df)
    st.markdown("---")

    # 7. Substrate Vendor Analysis (Section 7 - NEW)
    render_vendor_analysis_page(df_filtered, all_affected_lots_base)

# ======================================================================
# MAIN APP EXECUTION AND NAVIGATION
# ======================================================================

def main():
    # UPDATED: Version includes Vendor Analysis (v1.9) and 12-Hour Shift Logic, plus Timeline Fix
    st.set_page_config(
        layout="wide",
        page_title="Process & Machine Mapping Analyst (v1.9.2 - Timeline Robustness Fix)",
        initial_sidebar_state="expanded"
    )

    st.sidebar.title("🛠️ Analysis Control Panel")

    # Initialize Session State
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.data = None
        st.session_state.affected_quantity_map = {}
        st.session_state.all_affected_lots_base = []
        st.session_state.selected_edv = 'All Products (Clear Filter)'
        st.session_state.selected_lots = []

    # 1. FILE UPLOADER
    uploaded_file = st.sidebar.file_uploader(
        "1. Upload Raw Data (CSV or XLSX)",
        type=['csv', 'xlsx'],
        key="file_uploader"
    )

    # Update Session State if a new file is uploaded
    if uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.data = None
        st.session_state.affected_quantity_map = {}
        st.session_state.all_affected_lots_base = []
        st.session_state.selected_edv = 'All Products (Clear Filter)'
        st.session_state.selected_lots = []

    # --- Load and Process Data ---
    df = load_data(st.session_state.uploaded_file)

    if df is not None:
        st.session_state.data = df

        # 2. AFFECTED LOT INPUT
        st.sidebar.markdown("---")
        st.sidebar.subheader("2. Affected Lot IDs & Quantity")
        st.sidebar.markdown("Format: `FullLotID Quantity` (one per line)")

        affected_lots_input = st.sidebar.text_area(
            "Lot ID Quantity Input:",
            height=150,
            value="",
            key="sidebar_affected_lots_input"
        )

        # Parse Lot Input
        affected_quantity_map = {}
        lines = affected_lots_input.strip().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                full_lot_id = parts[0].strip().split('(')[0]
                try:
                    quantity = int(parts[1].strip())
                    if quantity > 0:
                        affected_quantity_map[full_lot_id] = quantity
                except ValueError:
                    continue

        st.session_state.affected_quantity_map = affected_quantity_map
        st.session_state.all_affected_lots_base = sorted(list(affected_quantity_map.keys()))

        # Get lot EDV info for display and filtering
        affected_lots_edv_df = get_affected_lot_edv_info(df, st.session_state.all_affected_lots_base)

        # --- Check for minimum required input ---
        if not st.session_state.all_affected_lots_base:
            st.title("Welcome to the Process Analysis Tool (v1.9.2)")
            st.warning("Please enter valid Full Lot IDs and quantities in the sidebar to proceed with analysis.")
            st.stop()


        # 3. MASTER PRODUCT FILTER
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. Master Product Filter")

        edv_options = ['All Products (Clear Filter)']

        # Determine available EDVs from the *entire dataset*
        if 'EDV' in df.columns:
            all_edv_options = sorted(df['EDV'].dropna().unique().astype(str).tolist())
            edv_options.extend(all_edv_options)

        # AUTOMATIC FILTER LOGIC:
        default_edv_selection = 'All Products (Clear Filter)'

        try:
            default_index = edv_options.index(default_edv_selection)
        except ValueError:
            default_index = 0

        selected_edv = st.sidebar.selectbox(
            "Filter by Product Name (EDV):",
            options=edv_options,
            index=default_index,
            key='sidebar_product_filter'
        )
        st.session_state.selected_edv = selected_edv

        # Apply Master Filter (This filter sets the baseline data scope)
        if selected_edv != 'All Products (Clear Filter)' and 'EDV' in df.columns:
            df_filtered = df[df['EDV'] == selected_edv].copy()
        else:
            df_filtered = df.copy()

        if df_filtered.empty:
            st.error(f"No data found for the selected product: {selected_edv}. Please check your filter or data.")
            st.stop()

        # 4. LOT SUB-FILTER (Used for comparison/timeline)
        st.sidebar.markdown("---")
        st.sidebar.subheader("4. Lot Sub-Filter (For Focused Analysis)")

        lots_in_filtered_data = df_filtered['ASSEMBLY_LOT'].unique().tolist()
        # Only show affected lots that are present in the *master filtered* data
        available_lots_for_select = sorted(list(set(st.session_state.all_affected_lots_base) & set(lots_in_filtered_data)))

        # Preserve selection if the filtered list still contains the old selection
        default_selected_lots = [lot for lot in st.session_state.selected_lots if lot in available_lots_for_select]
        if not default_selected_lots: # If old selection is invalid, default to all available
            default_selected_lots = available_lots_for_select

        selected_lots = st.sidebar.multiselect(
            "Select Lots for Focused Analysis:",
            options=available_lots_for_select,
            default=default_selected_lots,
            key='sidebar_selected_lots'
        )

        if not selected_lots:
             st.sidebar.warning("Select at least one lot for focused analysis.")
             st.stop()

        st.session_state.selected_lots = selected_lots


        # --- Sidebar Navigation ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("5. Advanced Navigation")

        page = st.sidebar.radio(
            "Go to Section:",
            [
                "🏠 Data Upload & Preview",
                "📈 Comprehensive Analysis Report",
            ]
        )

        # --- Dynamic Page Rendering ---
        if page == "🏠 Data Upload & Preview":
            render_upload_page(
                st.session_state.data,
                affected_quantity_map,
                st.session_state.all_affected_lots_base,
                affected_lots_edv_df
            )

        elif page == "📈 Comprehensive Analysis Report":
            render_combined_analysis_page(
                df_filtered,
                st.session_state.all_affected_lots_base,
                selected_lots,
                affected_quantity_map,
                selected_edv,
                affected_lots_edv_df
            )

    else:
        st.title("Welcome to the Process Analysis Tool (v1.9.2 - Timeline Robustness Fix)")
        st.info("Please upload your manufacturing data file (.csv or .xlsx) using the control panel on the left to start the analysis.")


if __name__ == "__main__":
    main()

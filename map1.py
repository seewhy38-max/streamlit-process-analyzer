import streamlit as st
import pandas as pd
import numpy as np
import altair as alt 
from io import StringIO
from datetime import timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- Global Configuration (Define your critical steps and staging limits here) ---
CRITICAL_STEPS = {
    '4282': 'TIM-Adhesive Dispensing', '4285': 'LID ATTACH', 
    '4276': 'STIFFENER DISPENSE', '4277': 'STIFFENER ATTACH',
    '4271': 'AOI (TOP Inspection)', '4269': 'PGA/LGA CHIP CAP REWORK (Top)',
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
    ('4280', '4300'): {'limit_hours': 6, 'description': 'TIM & ADHESIVE (Old) (Track In) to LID ATTACH (Old) (Track Out)', 'start_ts': 'TRACK_IN_TS', 'end_ts': 'TRACK_OUT_TS'},
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
        date_format = '%d/%m/%Y %H:%M:%S'
        date_parser = lambda x: pd.to_datetime(x, format=date_format, errors='coerce')
        required_cols = ['ASSEMBLY_LOT', 'SPECNAME', 'TRACK_IN_TS', 'TRACK_OUT_TS', 'EQUIPMENT_LINE_NAME', 'OPERATOR', 'TRACKOUT_QTY', 'EDV']

        file_name = uploaded_file.name
        
        if file_name.endswith('.csv') or uploaded_file.type == "text/csv":
            file_string = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(file_string, usecols=required_cols)
        elif file_name.endswith('.xlsx') or uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            # openpyxl must be installed for XLSX support
            df = pd.read_excel(uploaded_file, sheet_name=0, usecols=required_cols)
        else:
            st.error("Unsupported file type. Please upload a CSV or XLSX file.")
            return None

        df['SPECNAME'] = df['SPECNAME'].astype(str)
        df['ASSEMBLY_LOT'] = df['ASSEMBLY_LOT'].astype(str)
        df['EDV'] = df['EDV'].astype(str).str.strip().replace('nan', 'Unknown', regex=False) # Ensure EDV is cleaned
        df['TRACK_OUT_TS'] = df['TRACK_OUT_TS'].apply(date_parser)
        df['TRACK_IN_TS'] = df['TRACK_IN_TS'].apply(date_parser)
        
        if 'TRACKOUT_QTY' in df.columns:
            temp_qty = pd.to_numeric(
                df['TRACKOUT_QTY'].astype(str).str.replace(',', '', regex=False).str.strip('"'),
                errors='coerce'
            )
            df = df[temp_qty > 0].copy()
        
        return df
    except Exception as e:
        st.error(f"Error loading or parsing file: {e}")
        return None

# --- Cycle Time Calculation (NOW INCLUDES UCT) ---
@st.cache_data(show_spinner=False)
def calculate_cycle_time(df):
    df_ct = df.copy()
    date_format = '%d/%m/%Y %H:%M:%S'
    for col in ['SPECNAME', 'TRACKOUT_QTY']:
        if col in df_ct.columns:
            df_ct[col] = df_ct[col].astype(str)
    
    # Ensure timestamps are datetime objects
    df_ct['TRACK_IN_TS'] = pd.to_datetime(df_ct['TRACK_IN_TS'], format=date_format, errors='coerce')
    df_ct['TRACK_OUT_TS'] = pd.to_datetime(df_ct['TRACK_OUT_TS'], format=date_format, errors='coerce')
    
    # Clean and parse quantity
    if 'TRACKOUT_QTY' in df_ct.columns:
        df_ct['Trackout Quantity'] = pd.to_numeric(
            df_ct['TRACKOUT_QTY'].str.replace(',', '', regex=False).str.strip('"'), 
            errors='coerce'
        ).fillna(1).astype(int)
    else:
        df_ct['Trackout Quantity'] = 1 
        
    # 1. Calculate Lot Cycle Time (Original: Time for the entire batch)
    df_ct['Lot Cycle Time (min)'] = (df_ct['TRACK_OUT_TS'] - df_ct['TRACK_IN_TS']).dt.total_seconds() / 60
    
    # 2. Calculate Unit Cycle Time (NEW! Normalized: Time per Unit)
    # Use np.where to avoid division by zero and handle cases where quantity is 1
    df_ct['Unit Cycle Time (min/unit)'] = np.where(
        df_ct['Trackout Quantity'] > 1, # Only divide if quantity is greater than 1
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

    date_format = '%d/%m/%Y %H:%M:%S' 
    for ts_col in ['TRACK_IN_TS', 'TRACK_OUT_TS']:
        if not pd.api.types.is_datetime64_any_dtype(df_staging.get(ts_col)):
            df_staging[ts_col] = pd.to_datetime(df_staging[ts_col], format=date_format, errors='coerce')

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
    all_staging_records = []

    for (start_step, end_step), config in STAGING_LIMITS.items():
        limit_hours = config['limit_hours']
        description = config['description']
        start_ts_type = config['start_ts']
        end_ts_type = config['end_ts']
        
        start_col = start_step
        end_col = end_step
        
        if start_col in staging_pivot.columns and end_col in staging_pivot.columns:
            
            # Calculate time difference using the pre-pivoted TS columns
            staging_pivot[f'Time Diff ({start_step} to {end_step})'] = (
                staging_pivot[end_col] - staging_pivot[start_col]
            ).dt.total_seconds() / 3600

            df_check = staging_pivot.dropna(subset=[start_col, end_col]).copy()
            df_check = df_check[df_check[f'Time Diff ({start_step} to {end_step})'] >= 0]
            
            df_check['Violation'] = df_check[f'Time Diff ({start_step} to {end_step})'] > limit_hours
            
            for index, row in df_check.iterrows():
                # Extracting the TS string with the appropriate suffix for display
                start_ts_str = row[start_col].strftime('%Y-%m-%d %H:%M') + f" ({start_ts_type.replace('TRACK_', '').replace('_TS', '')})"
                end_ts_str = row[end_col].strftime('%Y-%m-%d %H:%M') + f" ({end_ts_type.replace('TRACK_', '').replace('_TS', '')})"

                all_staging_records.append({
                    'Lot ID': row['ASSEMBLY_LOT'],
                    'Staging Check': description,
                    'Actual Time (hrs)': row[f'Time Diff ({start_step} to {end_step})'],
                    'Limit (hrs)': limit_hours,
                    'Start Step TS': start_ts_str,
                    'End Step TS': end_ts_str,
                    'Violation': row['Violation']
                })

    df_report = pd.DataFrame(all_staging_records)

    if not df_report.empty:
        df_report = df_report.sort_values(['Staging Check', 'Actual Time (hrs)'], ascending=[True, False]).reset_index(drop=True)
        df_report = df_report[['Lot ID', 'Staging Check', 'Actual Time (hrs)', 'Limit (hrs)', 'Start Step TS', 'End Step TS', 'Violation']]
    
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

    return traceability, common_tools

# --- Timeline Data Generation (UPDATED to include TRACK_IN_TS) ---
@st.cache_data(show_spinner=False)
def generate_timeline_data(df, affected_lots_base, step_id, common_machine, affected_quantity_map, window=5): 
    df['SPECNAME'] = df['SPECNAME'].astype(str) 
    df['ASSEMBLY_LOT'] = df['ASSEMBLY_LOT'].astype(str)
    
    df_timeline = df[
        (df['EQUIPMENT_LINE_NAME'] == common_machine) & 
        (df['SPECNAME'] == step_id)
    ].sort_values('TRACK_OUT_TS').reset_index(drop=True)

    if df_timeline.empty:
        return None, None

    affected_indices = df_timeline[
        df_timeline['ASSEMBLY_LOT'].isin(affected_lots_base)
    ].index.tolist()

    if not affected_indices:
        return None, None

    start_index = max(0, min(affected_indices) - window)
    end_index = min(len(df_timeline), max(affected_indices) + window + 1)

    timeline_window = df_timeline.iloc[start_index:end_index].copy()
    timeline_window['SEQUENCE_INDEX'] = timeline_window.index 
    
    timeline_window['AFFECTED_QTY'] = timeline_window['ASSEMBLY_LOT'].map(affected_quantity_map).fillna(0).astype(int)

    # UPDATED: Added 'TRACK_IN_TS' to the selected columns
    timeline_window = timeline_window[[
        'ASSEMBLY_LOT', 'SEQUENCE_INDEX', 'TRACK_IN_TS', 
        'TRACK_OUT_TS', 'OPERATOR', 'EQUIPMENT_LINE_NAME', 
        'SPECNAME', 'AFFECTED_QTY'
    ]].copy()
    
    # Renaming the columns
    timeline_window.columns = [
        'Lot ID', 'Sequence Index', 'Track In Timestamp', 
        'Track Out Timestamp', 'Operator', 'Machine ID', 
        'Step ID', 'Affected Quantity'
    ]
    timeline_window['Affected'] = timeline_window['Affected Quantity'] > 0
    
    return timeline_window, affected_lots_base

# --- Cycle Time Comparison Generation (Uses UCT) ---
def generate_cycle_time_comparison_table(df_ct, df_affected_ct, selected_lots):
    
    # Calculate Baseline Avg UCT
    baseline_ct = df_ct.groupby('Process Step Name')['Unit Cycle Time (min/unit)'].mean().reset_index()
    baseline_ct.rename(columns={'Unit Cycle Time (min/unit)': 'Baseline Avg UCT (min/unit)'}, inplace=True)
    
    # Calculate Affected Lot Avg UCT
    affected_ct = df_affected_ct.groupby('Process Step Name')['Unit Cycle Time (min/unit)'].mean().reset_index()
    affected_ct.rename(columns={'Unit Cycle Time (min/unit)': 'Affected Lot Avg UCT (min/unit)'}, inplace=True)

    affected_count = df_affected_ct.groupby('Process Step Name')['ASSEMBLY_LOT'].nunique().reset_index()
    affected_count.rename(columns={'ASSEMBLY_LOT': 'Affected Lot Count'}, inplace=True)

    comparison_df = pd.merge(baseline_ct, affected_ct, on='Process Step Name', how='inner')
    comparison_df = pd.merge(comparison_df, affected_count, on='Process Step Name', how='inner')
    
    # Calculate UCT difference
    comparison_df['UCT Difference (min/unit)'] = comparison_df['Affected Lot Avg UCT (min/unit)'] - comparison_df['Baseline Avg UCT (min/unit)']
    comparison_df['UCT Difference (%)'] = (comparison_df['UCT Difference (min/unit)'] / comparison_df['Baseline Avg UCT (min/unit)']) * 100

    comparison_df.rename(columns={'Affected Lot Count': f'{len(selected_lots)} Selected Full Lot ID(s) Count'}, inplace=True)

    return comparison_df

# --- NEW: Lot-Level UCT Comparison Function ---
def generate_lot_uct_comparison(df_ct, df_affected_ct, baseline_comparison_df):
    """
    Compares the UCT of individual affected lots against the process step baseline UCT.
    """
    if baseline_comparison_df.empty:
        return pd.DataFrame()

    # 1. Get Baseline Averages (from the aggregated comparison table)
    baseline_map = baseline_comparison_df.set_index('Process Step Name')['Baseline Avg UCT (min/unit)'].to_dict()
    
    # Filter the affected lot data to the relevant steps
    df_lot_uct = df_affected_ct.copy()
    
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
        # UPDATED: Added 'Track In Timestamp' to cols_to_style
        cols_to_style = ['Lot ID', 'Track In Timestamp', 'Track Out Timestamp', 'Operator', 'Affected Quantity']
        styles = [style if col in cols_to_style else '' for col in s.index]
        return styles

    # Format both timestamps to show full detail for sequencing
    styled_df = df_window.style.apply(highlight_affected_full_row, axis=1).format(
        {
            'Track In Timestamp': lambda t: t.strftime('%Y-%m-%d %H:%M:%S'), # NEW FORMATTING
            'Track Out Timestamp': lambda t: t.strftime('%Y-%m-%d %H:%M:%S')
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

# --- NEW Styling Function for the Pivot Table ---
def style_lot_comparison_table(df):
    def highlight_positive_difference(series):
        is_positive = series > 0
        return ['background-color: #fce5cd; font-weight: bold;' if v else '' for v in is_positive]

    # Format the entire table, excluding the first column ('Full Lot ID')
    return df.style.format('{:+.1f}%', subset=df.columns[1:]).apply(highlight_positive_difference, axis=0, subset=df.columns[1:])


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
            
            # Show the table with the new EDV column
            st.dataframe(
                affected_lots_edv_df,
                use_container_width=True,
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
    st.header("1. ⏳ Staging Time Analysis")
    st.divider()
    
    if df_filtered is None:
        st.error("Data filtering incomplete.")
        return

    with st.spinner("Calculating Staging Time Violations..."):
        staging_violations_df = calculate_staging_time_violations(df_filtered, all_affected_lots_base)

    st.subheader("Table 2: Material Staging Time Report")
    st.markdown(r"Checks elapsed time against limits. **Violations are highlighted in red.**")
    
    if not staging_violations_df.empty:
        num_violations = staging_violations_df['Violation'].sum()
        
        if num_violations > 0:
            st.error(f"🚨 **{num_violations} Staging Time Violation(s) Found!**")
        else:
            st.success("✅ All checked lots passed the staging time requirements.")
        
        with st.container(height=400):    
            st.dataframe(style_staging_table(staging_violations_df), use_container_width=True)
            
    else:
        st.warning("No data available to calculate staging times for the affected lots at the required steps.")


def render_traceability_page(df_filtered, all_affected_lots_base):
    st.header("2. 🔍 Traceability & Commonality Analysis")
    st.divider()
    
    if df_filtered is None:
        st.error("Data filtering incomplete.")
        return

    with st.spinner("Generating Traceability Matrix..."):
        traceability_df, common_tools = generate_traceability_matrix(df_filtered, all_affected_lots_base)

    st.subheader("Table 1: Tool Traceability Matrix")
    st.markdown("The most common tool for a critical step across all affected lots is highlighted in **green**, indicating a potential suspect machine.")
    
    if not traceability_df.empty:
        with st.container(height=400):
            st.dataframe(style_traceability_table(traceability_df, common_tools), use_container_width=True)
    else:
        st.warning("No traceability data found for the input lots in the critical steps within the current product filter.")
    
    return traceability_df, common_tools


def render_timeline_page(df_filtered, affected_lots_base, affected_quantity_map, traceability_df, common_tools):
    st.header("3. ⏱️ Time-Series Proximity (Timeline) Analysis")
    st.divider()
    
    if df_filtered is None:
        st.error("Data filtering incomplete.")
        return

    if traceability_df.empty:
        st.warning("Timeline analysis requires traceability data. Please check the 'Traceability & Commonality Analysis' section above.")
        return
    
    timeline_options = {}
    for step_id, machine_id in common_tools.items():
        step_name = CRITICAL_STEPS.get(step_id, f"Unknown Step {step_id}")
        option_label = f"{step_name} ({step_id}) - Tool: {machine_id}"
        timeline_options[option_label] = (step_id, machine_id)
    
    if timeline_options:
        
        with st.container():
            selected_label = st.selectbox(
                "Select Process Step and Common Tool to Analyze:",
                options=list(timeline_options.keys()),
                key='timeline_select_key'
            )
        
        selected_step, selected_machine = timeline_options[selected_label]
        
        st.info(f"Focused Analysis on Tool: **{selected_machine}** (Sequence: 5 lots before/after first/last affected lot).")

        with st.spinner(f"Generating timeline for {selected_machine}..."):
            timeline_data, _ = generate_timeline_data(
                df_filtered, affected_lots_base, selected_step, selected_machine, affected_quantity_map
            )
        
        if timeline_data is not None:
            
            # --- Figure 2: Timeline Chart ---
            st.subheader("Figure 2: Production Timeline Chart")
            st.markdown("The size and color of the points represent the **Affected Quantity**. Context lots are in gray.")
            
            chart_base = alt.Chart(timeline_data).encode(
                x=alt.X('Track Out Timestamp', title='Track Out Time', axis=alt.Axis(format='%m/%d %H:%M')), 
                y=alt.Y('Sequence Index', title='Production Sequence (Index)'),
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

            # --- Table 4: Detailed Timeline Table (Updated to show Track In) ---
            st.subheader("Table 4: Detailed Timeline Lot Sequence")
            st.markdown("Affected lots are highlighted in **orange/bold**. Sequence is indicated by the precise **Track Out Timestamp**.")
            
            with st.container(height=300):
                timeline_df_styled = style_timeline_table(timeline_data) 
                st.dataframe(timeline_df_styled, use_container_width=True)
            
        else:
            st.warning(f"Could not generate timeline for the selected tool **{selected_machine}**.")
    else:
        st.warning("No common machines found across the critical process steps for the affected lots to perform a timeline analysis.")


def render_cycle_time_page(df_filtered, affected_lots_base, selected_edv):
    st.header("4. 📈 Unit Cycle Time (UCT) Variability Analysis")
    st.divider()
    
    if df_filtered is None:
        st.error("Data filtering incomplete.")
        return

    with st.spinner("Calculating Unit Cycle Time Metrics..."):
        df_ct = calculate_cycle_time(df_filtered)
        ct_steps_for_analysis = {k: v for k, v in CRITICAL_STEPS.items() if k not in ['4280', '4300']}
        df_ct_filtered = df_ct[df_ct['SPECNAME'].isin(ct_steps_for_analysis.keys())].copy()
        
        comparison_df = pd.DataFrame() 
        lot_comparison_pivot = pd.DataFrame() # Initialize the new DataFrame
        
        if not df_ct_filtered.empty:
            df_ct_filtered['Process Step Name'] = df_ct_filtered['SPECNAME'].map(CRITICAL_STEPS).fillna(df_ct_filtered['SPECNAME'])
            df_affected_ct = df_ct_filtered[df_ct_filtered['ASSEMBLY_LOT'].isin(affected_lots_base)].copy()
            
            # Table 6 (Aggregate Comparison)
            comparison_df = generate_cycle_time_comparison_table(df_ct_filtered, df_affected_ct, affected_lots_base)
            
            # NEW! Table 7 (Lot-level Comparison)
            lot_comparison_pivot = generate_lot_uct_comparison(df_ct_filtered, df_affected_ct, comparison_df)

    # --- Figure 3: Box Plot (UCT) ---
    st.subheader("Figure 3: Process Unit Cycle Time (UCT) Baseline Distribution (Box Plot)")
    
    if not df_ct_filtered.empty:
        steps_with_data = df_ct_filtered['Process Step Name'].unique()
        step_order = [CRITICAL_STEPS[k] for k in ct_steps_for_analysis.keys() if CRITICAL_STEPS[k] in steps_with_data]

        st.markdown(f"Shows **UCT (min/unit)** distribution for **{selected_edv}** lots across critical steps. **Y-axis is logarithmic**.")

        X_ENCODING = alt.X('Process Step Name', axis=alt.Axis(labelAngle=-45, title='Process Step'), sort=step_order, type='nominal')
        # Reference the new Unit Cycle Time column
        Y_ENCODING = alt.Y('Unit Cycle Time (min/unit)', title='Unit Cycle Time (Minutes/Unit) [Log Scale]', scale=alt.Scale(type="log", domainMin=0.001))

        ct_chart = alt.Chart(df_ct_filtered).mark_boxplot(extent="min-max", size=25, opacity=0.8 ).encode(
            x=X_ENCODING, y=Y_ENCODING, color=alt.value("darkblue"), 
            # Update tooltip to show UCT
            tooltip=['Process Step Name', 'EQUIPMENT_LINE_NAME', alt.Tooltip('Unit Cycle Time (min/unit)', format='.3f', title='UCT (min/unit)')]
        ).properties(
            title=f"Process Unit Cycle Time Baseline Distribution for {selected_edv}"
        ).interactive()
        
        st.altair_chart(ct_chart, use_container_width=True) 

    else:
        st.warning("Cycle Time analysis skipped: No valid production data found for the critical process steps.")

    st.markdown("---")

    # --- Table 6: Comparison Table (UCT) ---
    st.subheader(f"Table 6: Unit Cycle Time (UCT) Comparison (Affected Lots Avg vs. Baseline Avg)")
    
    if not comparison_df.empty:
        st.markdown(f"Compares Avg **UCT (min/unit)** of the **{len(affected_lots_base)}** selected lots against the overall baseline. **Positive differences (orange)** mean the affected group was slower.")
        st.dataframe(style_comparison_table(comparison_df, affected_lots_base), use_container_width=True)
    else:
        st.warning("Unit Cycle Time comparison analysis skipped: No valid comparison data found.")

    # --- NEW! Table 7: Lot-Level UCT Comparison ---
    st.markdown("---")
    st.subheader("Table 7: Individual Lot UCT Performance vs. Baseline")
    
    if not lot_comparison_pivot.empty:
        st.markdown(r"Shows the **percentage difference** in UCT for each selected lot compared to the process step's baseline average. **Positive values (orange)** indicate the specific lot ran slower than the baseline average.")
        with st.container(height=300):
            st.dataframe(style_lot_comparison_table(lot_comparison_pivot), use_container_width=True)
    else:
        st.info("Individual lot UCT comparison skipped: Insufficient data to compare individual lots against the baseline.")
    
    return comparison_df

# --- New Combined Analysis Page ---
def render_combined_analysis_page(df_filtered, all_affected_lots_base, selected_lots, affected_quantity_map, selected_edv):
    st.title("Comprehensive Analysis Report")
    st.markdown("This page combines all the key analyses into one scrollable view.")
    st.markdown("---")
    
    # 1. Staging Time Analysis
    render_staging_page(df_filtered, all_affected_lots_base)
    st.markdown("---")
    
    # 2. Traceability Analysis
    traceability_df, common_tools = render_traceability_page(df_filtered, all_affected_lots_base)
    st.markdown("---")
    
    # 3. Timeline Analysis
    render_timeline_page(df_filtered, selected_lots, affected_quantity_map, traceability_df, common_tools)
    st.markdown("---")
    
    # 4. Cycle Time Analysis (Now UCT)
    render_cycle_time_page(df_filtered, selected_lots, selected_edv)
    
# ======================================================================
# MAIN APP EXECUTION AND NAVIGATION
# ======================================================================

def main():
    st.set_page_config(
        layout="wide", 
        page_title="Process & Machine Mapping Analyst (v1.4 - Final)",
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
            st.title("Welcome to the Process Analysis Tool")
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
        lot_edvs = affected_lots_edv_df.reset_index()['Product Name (EDV)'].unique().tolist()
        
        # Filter out the 'NOT FOUND' entry
        lot_edvs = [edv for edv in lot_edvs if edv != 'NOT FOUND IN DATA']
        
        if len(lot_edvs) == 1:
            default_edv_selection = lot_edvs[0]
        else:
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
        
        # Apply Master Filter
        if selected_edv != 'All Products (Clear Filter)' and 'EDV' in df.columns:
            df_filtered = df[df['EDV'] == selected_edv].copy()
        else:
            df_filtered = df.copy()
            
        if df_filtered.empty:
            st.error(f"No data found for the selected product: {selected_edv}. Please check your filter or data.")
            st.stop()

        # 4. LOT SUB-FILTER (Used for comparison/timeline)
        st.sidebar.markdown("---")
        st.sidebar.subheader("4. Lot Sub-Filter (For Tables 4, 6 & 7)")
        
        lots_in_filtered_data = df_filtered['ASSEMBLY_LOT'].unique().tolist()
        available_lots_for_select = sorted(list(set(st.session_state.all_affected_lots_base) & set(lots_in_filtered_data)))

        selected_lots = st.sidebar.multiselect(
            "Select Lots for Focused Analysis:",
            options=available_lots_for_select,
            default=available_lots_for_select,
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
                selected_edv
            )
            
    else:
        st.title("Welcome to the Process Analysis Tool (v1.4 - Final)")
        st.info("Please upload your manufacturing data file (.csv or .xlsx) using the control panel on the left to start the analysis.")


if __name__ == "__main__":
    main()

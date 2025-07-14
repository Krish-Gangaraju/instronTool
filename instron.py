import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, LogFormatter, FuncFormatter
from io import StringIO
import io
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages





def download_all_plots_as_pdf():
    """
    Generates a multipage PDF of all mix plots using the same logic
    you had in col3, and returns an in-memory BytesIO buffer.
    """
    pdf_buffer = io.BytesIO()
    with bpdf.PdfPages(pdf_buffer) as pdf:
        for mix_key, files in uploads.items():
            if not files:
                continue
            fig, ax = plt.subplots(figsize=(3.7, 3.7), constrained_layout=True)
            ax.set_box_aspect(1)
            for idx, f in enumerate(files):
                df = clean_instron_file(f)
                x = df[chosen_strain]
                if metric == "Stress":
                    y = df[chosen_stress]
                else:
                    eps = df[chosen_strain].replace(0, np.nan) / 100
                    cond = (df[chosen_stress] > 0) & (df["Displacement (mm)"] > 1)
                    y = pd.Series(np.nan, index=df.index)
                    y.loc[cond] = (
                        df.loc[cond, chosen_stress] / eps.loc[cond]
                        + df.loc[cond, chosen_stress]
                    )
                lbl = f.name if label_mode == "Filename" else f"{mix_key} - Sample{idx+1}"
                ax.plot(
                    x, y,
                    color=plt.get_cmap("tab20").colors[idx % 20],
                    label=lbl,
                    linewidth=LINEWIDTH
                )

            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{mix_key}: {metric} vs Strain", fontsize=TITLE_FS)
            ax.set_xlabel(chosen_strain, fontsize=LABEL_FS)
            ax.set_ylabel(chosen_stress if metric == "Stress" else "MSV (MPa)", fontsize=LABEL_FS)
            ax.tick_params(labelsize=TICK_FS)

            handles, labels = ax.get_legend_handles_labels()
            pairs = sorted(zip(labels, handles), key=lambda x: x[0])
            sl, sh = zip(*pairs)
            ax.legend(
                sh, sl, title="Samples",
                fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS,
                loc="upper left", frameon=True, edgecolor="black"
            )

            pdf.savefig(fig)
            plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer




# ——— Custom CSS for tighter spacing around checkboxes & text inputs ———
st.set_page_config(page_title="Instron Post-Processing Tool", layout="wide")


def _load_instron_csv(buffer):
    """
    Read a CSV style buffer that may have arbitrary preamble lines.
    We scan line by line until we see a line beginning with "Time" (case-sensitive).
    Everything from that line onward is passed to pandas.read_csv.
    """
    # Rewind buffer to the start so repeated reads work
    if hasattr(buffer, "seek"):
        buffer.seek(0)

    # Read raw bytes or text lines
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, "rb").readlines()

    # Decode each line into a Python string
    lines = [
        L.decode("utf-8", errors="replace") if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]

    # Find the first line that starts with "Time"
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Time"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find a header row starting with 'Time' in the uploaded file.")

    # Build a single string containing that header line + all lines below it
    data_str = "".join(lines[header_idx + 1 :])
    # Let pandas read it (it will infer column names from the header row we skipped)
    return pd.read_csv(StringIO(data_str))



@st.cache_data
def clean_instron_file(buffer):
    """
    1) Load buffer into a pandas DataFrame by scanning for the "Time" header.
    2) Convert every column to numeric (coercing errors → NaN).
    3) Compute MSV = Stress/Strain + Stress, only where Stress > 0 and Displacement > 1 mm.
    """
    # Rewind & load
    df = _load_instron_csv(buffer)

    # Rename columns
    df.columns = [
        "Time (s)",
        "Force (kN)",
        "Displacement (mm)",
        "Strain (%)",
        "Stress (MPa)"
    ]

    # Coerce to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute MSV only under valid conditions
    eps = df["Strain (%)"].replace(0, np.nan) / 100.0
    cond = (df["Stress (MPa)"] > 0) & (df["Displacement (mm)"] > 1)
    msv = pd.Series(np.nan, index=df.index)
    msv.loc[cond] = df.loc[cond, "Stress (MPa)"] / eps.loc[cond] + df.loc[cond, "Stress (MPa)"]
    df["MSV [MPa]"] = msv

    return df



# —— Helpers: equalize rows and average curves ——
def equalize_dataframe_rows(dataframes):
    """
    Pad shorter DataFrames to the length of the longest one by repeating the last row.
    """
    max_rows = max(df.shape[0] for df in dataframes)
    equalized = []
    for df in dataframes:
        if df.shape[0] < max_rows:
            # create a DataFrame of the last row repeated
            last = df.iloc[[-1]]
            pads = pd.concat([last] * (max_rows - df.shape[0]), ignore_index=True)
            df = pd.concat([df, pads], ignore_index=True)
        equalized.append(df)
    return equalized


def average_stress_strain_curves(replicates):
    """
    replicates: list of lists of DataFrames (each sublist is one mix's replicates)
    Returns:
      averaged_dfs: list of DataFrames with ['Strain (%)','Stress (MPa)','MSV [MPa]']
                   each truncated at the **mean** failure, plus one vertical drop
      eq_dfs:      list of the equalized (trimmed & padded) replicates, if you still need them
    """
    averaged_dfs = []
    eq_dfs = []

    for replicate_set in replicates:
        trimmed = []
        failure_strains = []
        failure_msvs    = []

        # 1) Trim each replicate at its own peak‐stress
        for df in replicate_set:
            df = df[df['Strain (%)'] > 0].reset_index(drop=True)
            i_max = df['Stress (MPa)'].idxmax()

            # record each replicate's failure point
            failure_strains.append(df.loc[i_max, 'Strain (%)'])
            failure_msvs.append(   df.loc[i_max, 'MSV [MPa]']   )

            trimmed.append(df.iloc[:i_max+1].reset_index(drop=True))

        # 2) Equalize lengths so we can do a point‐wise mean
        eq_set = equalize_dataframe_rows(trimmed)
        eq_dfs.append(eq_set)

        # 3) Compute the average curve
        avg_strain = pd.concat([d['Strain (%)']   for d in eq_set], axis=1).mean(axis=1)
        avg_stress = pd.concat([d['Stress (MPa)'] for d in eq_set], axis=1).mean(axis=1)
        avg_msv    = pd.concat([d['MSV [MPa]']    for d in eq_set], axis=1).mean(axis=1)

        avg_df = pd.DataFrame({
            'Strain (%)':   avg_strain,
            'Stress (MPa)': avg_stress,
            'MSV [MPa]':    avg_msv
        })

        # 4) Compute the mean failure‐strain & mean failure‐MSV
        mean_strain = np.mean(failure_strains)
        mean_msv    = np.mean(failure_msvs)

        # 5) Truncate the averaged curve at the mean_strain
        if (avg_df['Strain (%)'] >= mean_strain).any():
            cut = avg_df['Strain (%)'].ge(mean_strain).idxmax()
            avg_df = avg_df.iloc[:cut+1].copy()

        # 6) Append two points to create one vertical drop:
        #    (mean_strain, mean_msv) → (mean_strain, 0)
        drop_start = pd.DataFrame({
            'Strain (%)':   [mean_strain],
            'Stress (MPa)': [avg_df['Stress (MPa)'].iloc[-1]],
            'MSV [MPa]':    [mean_msv]
        })
        drop_end = pd.DataFrame({
            'Strain (%)':   [mean_strain],
            'Stress (MPa)': [0.0],
            'MSV [MPa]':    [0.0]
        })

        avg_df = pd.concat([avg_df, drop_start, drop_end], ignore_index=True)
        averaged_dfs.append(avg_df)

    return averaged_dfs, eq_dfs





# ——— 1) Define mixes & upload samples ———
st.title("Instron Post-Processing Tool")
num_mixes = st.number_input("Enter number of mixes:", min_value=1, max_value=20, step=1, value=1)

uploads = {}
for i in range(1, num_mixes+1):
    mix_key = f"Mix{i}"
    files = st.file_uploader(f"**{mix_key}: upload sample CSV files**", type=["csv"], accept_multiple_files=True, key=f"uploader_{mix_key}")
    uploads[mix_key] = files



# Ensure at least one file in at least one mix
if not any(uploads.values()):
    st.info("📂 Please upload at least one CSV file for each mix to continue.")
    st.stop()

# ——— 2) Choose columns X and Y ———
# Process all uploaded to get column candidates
processed_tmp = {}
for mix_key, files in uploads.items():
    for f in files or []:
        try:
            processed_tmp[f.name] = clean_instron_file(f)
        except:
            pass

all_columns = set().union(*(df.columns.tolist() for df in processed_tmp.values()))
strain_candidates = [c for c in all_columns if "strain" in c.lower()]
stress_candidates = [c for c in all_columns if "stress" in c.lower()]

if not strain_candidates or not stress_candidates:
    st.error("❌ Could not detect both strain & stress columns.")
    st.stop()

chosen_strain = strain_candidates[0] if len(strain_candidates)==1 else st.selectbox("Choose X (strain) column:", sorted(strain_candidates))
chosen_stress = stress_candidates[0] if len(stress_candidates)==1 else st.selectbox("Choose Y (stress) column:", sorted(stress_candidates))

tab_graph, tab_comp, tab_key, tab_help = st.tabs(["Graph Interface", "Comparison Interace", "Key Values", "Help & User Manual"])



# right before you loop over mixes, define your sizes
TITLE_FS = 7
LABEL_FS = 6
TICK_FS  = 5
LEGEND_FS = 4.5
LEGEND_TITLE_FS = 5.5
LINEWIDTH = 1.5

### Helper: DRY styling for both graphs and comparison
def style_axes(ax, title, xlabel, ylabel):
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    handles, labels = ax.get_legend_handles_labels()
    pairs = sorted(zip(labels, handles), key=lambda x: x[0])
    if pairs:
        sorted_labels, sorted_handles = zip(*pairs)
        leg = ax.legend(sorted_handles, sorted_labels, title="Samples", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS,
            loc="upper left", frameon=True, edgecolor="black")
        leg.get_frame().set_linewidth(0.5)



# ——— Graph Interface ———
with tab_graph:
    st.subheader("Stress vs Strain — pick curves to plot")

    # — 0) Precompute averages for each mix so we can reuse them —
    average_dfs = {}
    for mix_key, files in uploads.items():
        if not files:
            continue
        dfs = [clean_instron_file(f) for f in files]
        avg_list, _ = average_stress_strain_curves([dfs])
        average_dfs[mix_key] = avg_list[0]

    col1, col2, col3 = st.columns([1,1,1], gap="small")
    with col1:
        metric = st.radio("Metric", ["Stress","MSV"], horizontal=True, key="metric_graph")
    with col2:
        show_grid = st.radio("Grid", ["On","Off"], horizontal=True, key="grid_toggle")
    with col3:
        label_mode = st.radio("Legend labels", ["Filename","Nickname"], horizontal=True, key="label_mode")

    show_avg = st.checkbox("Show average", value=True, key="show_avg_graph")
    st.markdown("---")

    # collect figures for PDF
    graph_figs = []

    for mix_key, files in uploads.items():
        if not files:
            continue

        # text input for custom title
        default_title = f"{mix_key}: {metric} vs Strain"
        title_text = st.text_input(f"Title for {mix_key}", value=default_title, key=f"title_{mix_key}")

        fig, ax = plt.subplots(figsize=(3.5,3.5), constrained_layout=True)
        ax.set_box_aspect(1)

        # load and plot replicates
        dfs = [clean_instron_file(f) for f in files]
        for idx, df in enumerate(dfs):
            x = df[chosen_strain]
            y = df[chosen_stress] if metric=="Stress" else df["MSV [MPa]"]
            raw = files[idx].name.rsplit('.',1)[0]
            lbl = raw if label_mode=="Filename" else f"{mix_key} - Sample{idx+1}"
            ax.plot(x, y,
                    color=plt.get_cmap("tab20").colors[idx%20],
                    linewidth=LINEWIDTH,
                    label=lbl)

        # plot average if toggled
        if show_avg:
            avg_df = average_dfs[mix_key]
            x_avg = avg_df[chosen_strain]
            y_avg = avg_df[chosen_stress] if metric=="Stress" else avg_df["MSV [MPa]"]
            ax.plot(x_avg, y_avg,
                    color="black", linestyle="--",
                    linewidth=LINEWIDTH*0.7,
                    label="Average")

        # toggle grid
        ax.grid(True if show_grid == "On" else False)

        # style and display
        style_axes(ax,
                   title=title_text,
                   xlabel="Strain [%]",
                   ylabel="Stress [MPa]" if metric=="Stress" else "MSV [MPa]")
        st.pyplot(fig, use_container_width=False)
        st.markdown("---")

        graph_figs.append(fig)

    # generate PDF from collected figures on demand
    if graph_figs:
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            for fig in graph_figs:
                pdf.savefig(fig)
        pdf_buffer.seek(0)
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_buffer,
            file_name="instron_all_mixes.pdf",
            mime="application/pdf"
        )




# ——— Comparison Interface ———
with tab_comp:
    st.subheader("Comparison — select representative samples per mix")

    # precompute averages
    average_dfs = {}
    for mix_key, files in uploads.items():
        if not files: continue
        dfs = [clean_instron_file(f) for f in files]
        avg, _ = average_stress_strain_curves([dfs])
        average_dfs[mix_key] = avg[0]

    # build select + checkbox per mix
    compare = {}
    include = {}
    for mix_key, files in uploads.items():
        if not files: continue
        col_sel, col_chk = st.columns([4,0.5], gap="small")
        options = ["Average"] + [f.name for f in files]
        sel = col_sel.selectbox(f"{mix_key}:", options, key=f"comp_{mix_key}")
        chk = col_chk.checkbox("Plot", True, key=f"inc_{mix_key}")
        compare[mix_key] = sel
        include[mix_key] = chk

    metric_cmp = st.radio("Metric", ["Stress","MSV"], horizontal=True, key="comp_metric")

    # ensure selection and at least one plot
    if len(compare)==len([k for k in uploads if uploads[k]]):
        plotted = [k for k,v in include.items() if v]
        if not plotted:
            st.info("✅ Tick at least one ‘Plot’ box to see a comparison.")
        else:
            fig, ax = plt.subplots(figsize=(3.5,3.5), constrained_layout=True)
            ax.set_box_aspect(1)
            palette = plt.get_cmap("tab20").colors

            for idx, mix_key in enumerate(compare):
                if not include[mix_key]: continue
                choice = compare[mix_key]
                raw = choice if choice=="Average" else choice.rsplit('.',1)[0]
                label = f"{mix_key}: {raw}"
                if choice=="Average":
                    df = average_dfs[mix_key]
                else:
                    fobj = next(f for f in uploads[mix_key] if f.name==choice)
                    df = clean_instron_file(fobj)
                x = df[chosen_strain]
                y = df[chosen_stress] if metric_cmp=="Stress" else df["MSV [MPa]"]
                ax.plot(x,y, color=palette[idx%len(palette)], linewidth=LINEWIDTH, label=label)

            style_axes(
                ax,
                title=f"Comparison: {metric_cmp} vs Strain",
                xlabel=chosen_strain,
                ylabel=chosen_stress if metric_cmp=="Stress" else "MSV (MPa)"
            )
            st.pyplot(fig, use_container_width=False)
    else:
        st.info("✅ Please select one sample for each mix to compare.")



# ——— Key Value Interface ———
with tab_key:
    st.subheader("Failure & M-Point Averages by Mix")

    # Helper: compute MSV series
    def msv_series(df):
        eps = df[chosen_strain].replace(0, np.nan) / 100.0
        return df[chosen_stress] / eps + df[chosen_stress]

    # Helper: pick value from a series at nearest strain
    def value_at_strain(series, df, target):
        idx = (df[chosen_strain] - target).abs().idxmin()
        return series.iloc[idx]

    results = {}
    for mix_key, files in uploads.items():
        if not files:
            continue

        # accumulators
        m2, m10, m100, m300 = [], [], [], []
        fs, fr = [], []

        for f in files:
            df = clean_instron_file(f)
            msv = msv_series(df)

            # M-point averages from MSV
            m2.append(value_at_strain(msv, df, 2.0))
            m10.append(value_at_strain(msv, df, 10.0))
            m100.append(value_at_strain(msv, df, 100.0))
            m300.append(value_at_strain(msv, df, 300.0))

            # failure = peak raw stress & its corresponding strain
            idx_max = df[chosen_stress].idxmax()
            fs.append(df[chosen_stress].iloc[idx_max])
            fr.append(df[chosen_strain].iloc[idx_max])

        results[mix_key] = [
            np.mean(m2),
            np.mean(m10),
            np.mean(m100),
            np.mean(m300),
            np.mean(m300) / np.mean(m100),
            np.mean(fs),
            np.mean(fr),
        ]

    df_avgs = pd.DataFrame(
        results,
        index=[
            "M2 [MPa]",
            "M10 [MPa]",
            "M100 [MPa]",
            "M300 [MPa]",
            "M300/M100",
            "Failure Stress (MPa)",
            "Failure Strain (%)"
        ]
    ).round(2)
    st.dataframe(df_avgs, use_container_width=True)



# ——— Help & User Manual ———
with tab_help:
    st.subheader("Help & User Manual")
    st.markdown(""" 
    Welcome to the **Instron Post-Processing Tool**, a Streamlit application designed to help you analyze mechanical test data without any coding! This guide will walk you through:

    - 🖥️ **User Interface Overview**
    - 🔢 **Step 1: Define Mixes & Upload Samples**
    - 📊 **Step 2: Choose Columns**
    - 📈 **Graph Interface**
    - 🔍 **Comparison Interface**
    - 🔑 **Key Values Tab**
    - 💾 **Download Options**
    - 💡 **Tips & Best Practices**


    ## 🖥️ User Interface Overview

    The application is organized into three main tabs:

    1. **Graph Interface**: Plot stress–strain or MSV curves per mix.
    2. **Comparison Interface**: Overlay representative samples across mixes.
    3. **Key Values**: Tabulate M-point averages and failure metrics.

    At the top, you’ll define how many **Mixes** you have and upload CSV test files for each.

    ---

    ## 🔢 Step 1: Define Mixes & Upload Samples

    - **Number of Mixes**: Use the numeric input to specify how many different formulations or groups you tested.
    - **File Uploaders**: For each mix (Mix1, Mix2, …), upload one or more CSV files. Each file should contain a header row starting with `Time` and columns for Force, Displacement, Strain, and Stress.

    > **Note:** The tool automatically cleans and parses your CSV, converting all columns to numbers and computing an additional **MSV** column.

    ---

    ## 📊 Step 2: Choose Columns

    Once files are uploaded, the app detects available columns:

    - **X-axis (Strain)**: Usually named `Strain (%)`.
    - **Y-axis (Stress)**: Usually named `Stress (MPa)`.

    Use the dropdown menus to confirm or change which columns to plot.

    ---

    ## 📈 Graph Interface

    This tab displays individual plots for each mix:

    1. **Metric**: Select **Stress** or **MSV**.
    2. **Legend Labels**: Choose between the original filename or a nickname (`MixX - SampleY`).
    3. **Show Average**: Toggle to overlay the average curve (dashed black line).

    Each mix appears in its own 3.5"×3.5" plot showing replicates (solid lines) and optional average. Horizontal and vertical axes start at zero for clarity.

    ---

    ## 🔍 Comparison Interface

    Here you can compare one representative curve per mix:

    1. **Select Sample**: For each mix, choose either **Average** or any individual CSV.
    2. **Plot Toggle**: Check the box to include that mix in the comparison.
    3. **Metric**: Again, pick Stress or MSV.

    All selected curves are overlaid in a single plot for direct comparison.

    ---

    ## 🔑 Key Values Tab

    Automatically calculates and displays:

    - **M‑Point Averages (MPa)**:
    - **M2**, **M10**, **M100**, **M300** at strains 2%, 10%, 100%, 300%.
    - **M300/M100** ratio.
    - **Failure Metrics**:
    - **Failure Stress** (peak stress averaged across replicates).
    - **Failure Strain** (strain at peak stress averaged across replicates).

    Results are shown in a concise table with one column per mix.

    ---

    ## 💾 Download Options

    - **Individual Plots**: Use the download button under each plot to save as PNG.
    - **All Mixes PDF**: In the Graph Interface, click **⬇️ Download PDF** to get a multipage PDF with every mix’s plot.

    ---

    ## 💡 Tips & Best Practices

    - **Consistent Filenames**: Name your CSVs clearly (e.g., `Mix1_Sample1.csv`).
    - **Batch Upload**: Drag & drop multiple files to speed up processing.
    - **Verify Raw Data**: If parsing errors occur, check that your CSV has a header line starting with `Time`.
    - **Performance**: Files are cached, so refreshing the same uploads is faster.

    ---

    """)

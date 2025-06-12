import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, LogFormatter, FuncFormatter
from io import StringIO
import io
import streamlit as st
import matplotlib.backends.backend_pdf as bpdf





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
            ax.set_ylabel(
                chosen_stress if metric == "Stress" else "MSV (MPa)",
                fontsize=LABEL_FS
            )
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

st.markdown("""
<style>
  /* Reduce bottom margin on each checkbox container */
  .stCheckbox {
    margin-bottom: 4px !important;
    padding-bottom: 0px !important;
  }
  /* Reduce top margin on each text input container */
  .stTextInput {
    margin-top: 0px !important;
    margin-bottom: 12px !important;
  }
</style>
""", unsafe_allow_html=True)


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

chosen_strain = strain_candidates[0] if len(strain_candidates)==1 else st.selectbox(
    "Choose X (strain) column:",
    sorted(strain_candidates)
)
chosen_stress = stress_candidates[0] if len(stress_candidates)==1 else st.selectbox(
    "Choose Y (stress) column:",
    sorted(stress_candidates)
)

tab_graph, tab_comp, tab_key = st.tabs(["Graph Interface", "Comparison Interace", "Key Values"])



# right before you loop over mixes, define your sizes
TITLE_FS = 7
LABEL_FS = 6
TICK_FS  = 5
LEGEND_FS = 5
LEGEND_TITLE_FS = 6
LINEWIDTH = 1.5

with tab_graph:
    st.subheader("Stress vs Strain — pick curves to plot")

    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    with col1:
        metric = st.radio("Metric", ["Stress", "MSV"], horizontal=True, key="metric_graph")
    with col2:
        label_mode = st.radio("Legend labels", ["Filename", "Nickname"], horizontal=True, key="label_mode")
    with col3:
        if st.button("📄 Download all as PDF"):
            pdf_data = download_all_plots_as_pdf()
            st.download_button("⬇️ Download PDF", data=pdf_data, file_name="instron_all_mixes.pdf", mime="application/pdf")

    st.markdown("---")

    for mix_key, files in uploads.items():
        if not files:
            continue

        fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
        ax.set_box_aspect(1)

        # — Clean & average —
        dataframes = [clean_instron_file(f) for f in files]
        averaged_dfs, _ = average_stress_strain_curves([dataframes])
        avg_df = averaged_dfs[0]

        # — Plot individual replicates —
        for idx, df in enumerate(dataframes):
            x = df[chosen_strain]
            y = df[chosen_stress] if metric == "Stress" else df["MSV [MPa]"]
            lbl = (
                files[idx].name
                if label_mode == "Filename"
                else f"{mix_key} - Sample{idx+1}"
            )
            ax.plot(
                x, y,
                color=plt.get_cmap("tab20").colors[idx % 20],
                linewidth=LINEWIDTH,
                label=lbl
            )

        # — Plot averaged curve —
        x_avg = avg_df[chosen_strain]
        if metric == "Stress":
            y_avg = avg_df[chosen_stress]
        else:
            y_avg = avg_df["MSV [MPa]"]
        ax.plot(
            x_avg, y_avg,
            color="black",
            linestyle="--",
            linewidth=LINEWIDTH,
            label="Average"
        )

        # — Axes & legend styling —
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{mix_key}: {metric} vs Strain", fontsize=TITLE_FS)
        ax.set_xlabel(chosen_strain, fontsize=LABEL_FS)
        ax.set_ylabel(
            chosen_stress if metric == "Stress" else "MSV (MPa)",
            fontsize=LABEL_FS
        )
        ax.tick_params(axis="both", which="major", labelsize=TICK_FS)

        handles, labels = ax.get_legend_handles_labels()
        pairs = sorted(zip(labels, handles), key=lambda x: x[0])
        sorted_labels, sorted_handles = zip(*pairs)
        leg = ax.legend(
            sorted_handles,
            sorted_labels,
            title="Samples",
            fontsize=LEGEND_FS,
            title_fontsize=LEGEND_TITLE_FS,
            loc="upper left",
            frameon=True,
            edgecolor="black",
        )
        leg.get_frame().set_linewidth(0.5)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        st.markdown("---")




with tab_comp:
    st.subheader("Comparison — select representative samples per mix")

    # 1) Precompute averages
    average_dfs = {}
    for mix_key, files in uploads.items():
        if not files:
            continue
        # load & average this mix
        dfs = [clean_instron_file(f) for f in files]
        averaged, _ = average_stress_strain_curves([dfs])
        average_dfs[mix_key] = averaged[0]

    # 2) Build selectboxes with "Average" first
    compare = {}
    for mix_key, files in uploads.items():
        if not files:
            continue
        options = ["Average"] + [f.name for f in files]
        sel = st.selectbox(
            f"{mix_key} sample to compare",
            options,
            key=f"comp_{mix_key}"
        )
        compare[mix_key] = sel

    metric = st.radio("Metric", ["Stress", "MSV"], horizontal=True, key="comp_metric")

    # 3) Once every mix has a selection, plot them together
    if len(compare) == len([k for k in uploads if uploads[k]]):
        fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
        ax.set_box_aspect(1)
        palette = plt.get_cmap("tab20").colors

        for idx, mix_key in enumerate(compare):
            choice = compare[mix_key]
            if choice == "Average":
                df = average_dfs[mix_key]
                label = f"{mix_key}: Average"
            else:
                # find the corresponding UploadedFile
                fobj = next(f for f in uploads[mix_key] if f.name == choice)
                df = clean_instron_file(fobj)
                label = f"{mix_key}: {choice}"

            x = df[chosen_strain]
            y = df[chosen_stress] if metric == "Stress" else df["MSV [MPa]"]
            ax.plot(
                x, y,
                color=palette[idx % len(palette)],
                label=label,
                linewidth=LINEWIDTH
            )

        # styling as before…
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        handles, labels = ax.get_legend_handles_labels()
        pairs = sorted(zip(labels, handles), key=lambda x: x[0])
        sorted_labels, sorted_handles = zip(*pairs)
        leg = ax.legend(
            sorted_handles, sorted_labels,
            title="Samples", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS,
            loc="upper left", frameon=True, edgecolor="black"
        )
        leg.get_frame().set_linewidth(0.5)
        ax.set_title(f"Comparison: {metric} vs Strain", fontsize=TITLE_FS)
        ax.set_xlabel(chosen_strain, fontsize=LABEL_FS)
        ax.set_ylabel(
            chosen_stress if metric=="Stress" else "MSV (MPa)",
            fontsize=LABEL_FS
        )
        ax.tick_params(labelsize=TICK_FS)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    else:
        st.info("✅ Please select one sample for each mix to compare.")





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

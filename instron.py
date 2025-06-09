import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, LogFormatter, FuncFormatter
from io import StringIO
import io
import streamlit as st
import matplotlib.backends.backend_pdf as bpdf


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
    """
    # This will rewind and load the CSV content
    df = _load_instron_csv(buffer)

    # Rename to your desired columns
    df.columns = ["Time (s)", "Force (kN)", "Displacement (mm)", "Strain (%)", "Stress (MPa)"]

    # Ensure everything is numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



st.title("Instron Post-Processing Tool")



# ——— 1) Define mixes & upload samples ———
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
LEGEND_FS = 4
LEGEND_TITLE_FS = 5
LINEWIDTH = 1.5

with tab_graph:
    st.subheader("Stress vs Strain — pick curves to plot")

    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    with col1:
        metric = st.radio(
            "Metric",
            ["Stress", "MSV"],
            horizontal=True,
            key="metric_graph"
        )
    with col2:
        label_mode = st.radio(
            "Legend labels",
            ["Filename", "Nickname"],
            horizontal=True,
            key="label_mode"
        )
    with col3:
        if st.button("📄 Download all as PDF"):
            # Generate a PDF in memory
            pdf_buffer = io.BytesIO()
            with bpdf.PdfPages(pdf_buffer) as pdf:
                for mix_key, files in uploads.items():
                    if not files:
                        continue
                    # recreate the same square figure per mix
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
                        lbl = f.name if label_mode=="Filename" else f"{mix_key} - Sample{idx+1}"
                        ax.plot(
                            x, y,
                            color=plt.get_cmap("tab20").colors[idx % 20],
                            label=lbl, linewidth=LINEWIDTH
                        )
                    ax.set_title(f"{mix_key}: {metric} vs Strain", fontsize=TITLE_FS)
                    ax.set_xlabel(chosen_strain, fontsize=LABEL_FS)
                    ax.set_ylabel(
                        chosen_stress if metric=="Stress" else "MSV (MPa)",
                        fontsize=LABEL_FS
                    )
                    ax.tick_params(labelsize=TICK_FS)
                    handles, labels = ax.get_legend_handles_labels()
                    pairs = sorted(zip(labels, handles), key=lambda x: x[0])
                    sl, sh = zip(*pairs)
                    ax.legend(
                        sh, sl, title="Samples",
                        fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS,
                        loc="upper left"
                    )
                    pdf.savefig(fig)
                    plt.close(fig)
            pdf_buffer.seek(0)
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_buffer,
                file_name="instron_all_mixes.pdf",
                mime="application/pdf"
            )

    st.markdown("---")




    for mix_key, files in uploads.items():
        if not files:
            continue

        # square figure
        fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
        ax.set_box_aspect(1)

        # plotting
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
            # choose legend label based on mode
            if label_mode == "Filename":
                lbl = f.name
            else:
                lbl = f"{mix_key} - Sample{idx+1}"

            ax.plot(x, y, color=plt.get_cmap("tab20").colors[idx % 20], label=lbl, linewidth=LINEWIDTH)

        # apply font sizes
        ax.set_title(f"{mix_key}: {metric} vs Strain", fontsize=TITLE_FS)
        ax.set_xlabel(chosen_strain, fontsize=LABEL_FS)
        ax.set_ylabel(
            chosen_stress if metric == "Stress" else "MSV (MPa)",
            fontsize=LABEL_FS
        )

        # ticks
        ax.tick_params(axis="both", which="major", labelsize=TICK_FS)

        # sort and draw legend
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

    # Allow user to pick one sample per mix
    compare = {}
    for mix_key, files in uploads.items():
        if not files:
            continue
        options = [f.name for f in files]
        sel = st.selectbox(
            f"{mix_key} sample to compare",
            options,
            key=f"comp_{mix_key}"
        )
        compare[mix_key] = sel

    # Metric radio (same as in graph)
    metric = st.radio("Metric", ["Stress", "MSV"], horizontal=True, key="comp_metric")

    # Once every mix has a selection, plot them together
    if len(compare) == len([k for k in uploads if uploads[k]]):
        fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)        
        ax.set_box_aspect(1)
        palette = plt.get_cmap("tab20").colors

        for idx, mix_key in enumerate(compare):
            fname = compare[mix_key]
            # find the corresponding UploadedFile object
            fobj = next(f for f in uploads[mix_key] if f.name == fname)
            df = clean_instron_file(fobj)
            x = df[chosen_strain]
            if metric == "Stress":
                y = df[chosen_stress]
            else:
                eps = df[chosen_strain].replace(0, np.nan) / 100.0
                cond = (df[chosen_stress] > 0) & (df["Displacement (mm)"] > 1)
                y = pd.Series(np.nan, index=df.index)
                y.loc[cond] = (
                    df.loc[cond, chosen_stress] / eps.loc[cond]
                    + df.loc[cond, chosen_stress]
                )

            ax.plot(
                x,
                y,
                color=palette[idx % len(palette)],
                label=f"{mix_key}: {fname}",
                linewidth=LINEWIDTH
            )

        # sort legend
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

        ax.set_title(f"Comparison: {metric} vs Strain", fontsize=TITLE_FS)
        ax.set_xlabel(chosen_strain, fontsize=LABEL_FS)
        ax.set_ylabel(
            chosen_stress if metric == "Stress" else "MSV (MPa)",
            fontsize=LABEL_FS
        )
        ax.tick_params(labelsize=TICK_FS)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    else:
        st.info("✅ Please select one sample for each mix to compare.")





with tab_key:
    st.subheader("Failure & M-Point Averages by Mix")

    results = {}
    def stress_at_strain(df, target):
        idx = (df[chosen_strain] - target).abs().idxmin()
        return df[chosen_stress].iloc[idx]

    for mix_key, files in uploads.items():
        if not files:
            continue
        fs, fr = [], []
        m2, m10, m100, m300 = [], [], [], []
        for f in files:
            df = clean_instron_file(f)
            idx_max = df[chosen_stress].idxmax()
            fs.append(df[chosen_stress].iloc[idx_max])
            fr.append(df[chosen_strain].iloc[idx_max])
            m2.append(stress_at_strain(df, 2.0))
            m10.append(stress_at_strain(df, 10.0))
            m100.append(stress_at_strain(df, 100.0))
            m300.append(stress_at_strain(df, 300.0))
        results[mix_key] = [
            np.mean(fs), np.mean(fr),
            np.mean(m2), np.mean(m10), np.mean(m100), np.mean(m300)
        ]

    df_avgs = pd.DataFrame(
        results,
        index=[
            "Failure Stress (MPa)",
            "Failure Strain (%)",
            "M2 (MPa)",
            "M10 (MPa)",
            "M100 (MPa)",
            "M300 (MPa)"
        ]
    )
    st.dataframe(df_avgs, use_container_width=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, LogFormatter, FuncFormatter
from io import StringIO
import io
import streamlit as st


# ——— Custom CSS for tighter spacing around checkboxes & text inputs ———
st.set_page_config(page_title="Instron Post-Processing Tool", layout="wide")

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


def _load_instron_csv(buffer):
    """
    Read a CSV‐style buffer that may have arbitrary preamble lines.
    We scan line by line until we see a line beginning with "Time" (case-sensitive).
    Everything from that line onward is passed to pandas.read_csv.
    """
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, "rb").readlines()
    lines = [
        L.decode("utf-8", errors="replace") if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Time"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find a header row starting with 'Time' in the uploaded file.")

    data_str = "".join(lines[header_idx+1:])
    df = pd.read_csv(StringIO(data_str))
    return df


@st.cache_data
def clean_instron_file(buffer):
    """
    1) Load buffer into a pandas DataFrame by scanning for the "Time" header.
    2) Convert every column to numeric (coercing errors → NaN).
    """
    df = _load_instron_csv(buffer)
    df.columns = ["Time (s)", "Force (kN)", "Displacement (mm)", "Strain (%)", "Stress (MPa)"]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


st.title("Instron Post-Processing Tool")


uploaded_files = st.file_uploader("Upload Instron CSV files", type=["csv"], accept_multiple_files=True, key="uploader_instron")



if not uploaded_files:
    st.info("📂 Please upload one or more CSV files to continue.")
    st.stop()

# Process each file and store in a dict: filename → DataFrame
processed = {}
for uploaded in uploaded_files:
    try:
        df_clean = clean_instron_file(uploaded)
        processed[uploaded.name] = df_clean
    except Exception as e:
        st.error(f"⚠️ Failed to process **{uploaded.name}**: {e}")

if not processed:
    st.stop()

# Gather all column names (lowercased) across all DataFrames to detect strain/stress candidates
all_columns = set()
for df in processed.values():
    all_columns.update(df.columns.tolist())

strain_candidates = [c for c in all_columns if "strain" in c.lower()]
stress_candidates = [c for c in all_columns if "stress" in c.lower()]

if not strain_candidates:
    st.error("❌ No column containing “strain” was found in any uploaded file.")
    st.stop()
if not stress_candidates:
    st.error("❌ No column containing “stress” was found in any uploaded file.")
    st.stop()

if len(strain_candidates) == 1:
    chosen_strain = strain_candidates[0]
else:
    chosen_strain = st.selectbox(
        "Multiple “strain” columns detected. Please choose which one to use for X:",
        options=sorted(strain_candidates),
    )

if len(stress_candidates) == 1:
    chosen_stress = stress_candidates[0]
else:
    chosen_stress = st.selectbox(
        "Multiple “stress” columns detected. Please choose which one to use for Y:",
        options=sorted(stress_candidates),
    )

tab_graph, tab_key, tab_data = st.tabs(["Graph Interface", "Key Values", "Data Interface"])




with tab_graph:
    st.subheader("Stress vs Strain — pick curves to plot")

    # 1) Metric selector: Stress or MSV
    metric = st.radio("Metric", ["Stress", "MSV"], horizontal=True)

    filenames = sorted(processed.keys())
    select_all = st.checkbox("Select All Files", value=True)

    files_to_plot = []
    display_names = {}

    # Helper to chunk filenames into rows of 3
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for row_files in chunk_list(filenames, 3):
        cols = st.columns(3)
        for idx, fname in enumerate(row_files):
            with cols[idx]:
                checked = st.checkbox(f"*{fname}*", value=select_all, key=f"cb_{fname}")
                new_label = st.text_input("Rename for plot (optional)", value=fname, key=f"rename_{fname}")

                if checked:
                    files_to_plot.append(fname)
                    display_names[fname] = new_label

        # Small gap after each row
        st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

        # Placeholders if fewer than 3 files
        for empty_idx in range(len(row_files), 3):
            with cols[empty_idx]:
                st.write("")

    if not files_to_plot:
        st.info("✅ Please select at least one file to plot.")
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

        palette = plt.get_cmap("tab20").colors
        color_map = {f: palette[i % len(palette)] for i, f in enumerate(filenames)}

        plotted_any = False
        for name in files_to_plot:
            df = processed[name]

            # Ensure required columns exist
            if chosen_strain not in df.columns or chosen_stress not in df.columns or "Displacement (mm)" not in df.columns:
                continue

            x = df[chosen_strain]  # Strain (%)

            if metric == "Stress":
                y = df[chosen_stress]

            else:
                # 1) Convert Strain (%) -> decimal, avoid zero division
                epsilon = df[chosen_strain].replace(0, np.nan) / 100.0

                # 2) Compute MSV only where Stress>0 and Displacement>1; else NaN
                cond_valid = (df[chosen_stress] > 0) & (df["Displacement (mm)"] > 1)
                y = pd.Series(np.nan, index=df.index)
                y.loc[cond_valid] = (
                    df.loc[cond_valid, chosen_stress] / epsilon.loc[cond_valid]
                    + df.loc[cond_valid, chosen_stress]
                )

            label_to_use = display_names.get(name, name)
            ax.plot(x, y, color=color_map[name], label=label_to_use, linewidth=1.5)
            plotted_any = True

        if not plotted_any:
            st.warning("No valid data found for the selected metric (Stress or MSV).")
        else:
            ax.set_title(f"Instron Test: {metric} vs Strain")
            ax.set_xlabel(chosen_strain)  # "Strain (%)"
            if metric == "Stress":
                ax.set_ylabel(chosen_stress)  # "Stress (MPa)"
            else:
                ax.set_ylabel("MSV (MPa)")
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.legend(title="Mixes", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            st.download_button("Download plot as PNG", data=buf, file_name=f"instron_{metric.lower()}_vs_strain.png", mime="image/png")





with tab_key:
    st.subheader("Key Values — Assign Mixes & Compute Failure Points")

    # 1) Let user specify how many mixes
    num_mixes = st.number_input("Enter number of mixes:", min_value=1, max_value=len(processed), step=1, value=1)

    # 2) For each mix, allow the user to select which files belong to that mix,
    #    enforcing that no file can appear in more than one mix.
    mix_assignments = {}
    filenames = sorted(processed.keys())
    available = set(filenames)

    for i in range(int(num_mixes)):
        mix_key = f"Mix{i+1}"
        mix_prompt = f"{mix_key} - select samples"
        options = sorted(available)
        selected = st.multiselect(mix_prompt, options=options, default=[], key=f"mix_sel_{i}")
        for s in selected:
            if s in available:
                available.remove(s)
        mix_assignments[mix_key] = selected

    # 3) Compute average failure stress & strain + M2, M10, M100, M300 for each mix
    if st.button("Compute Values"):
        failure_averages = {}

        # Helper to find stress at a given target strain (nearest point)
        def stress_at_strain(df, target_strain):
            if chosen_strain not in df.columns or chosen_stress not in df.columns:
                return np.nan
            idx = (df[chosen_strain] - target_strain).abs().idxmin()
            return df[chosen_stress].iloc[idx]

        for mix_key, file_list in mix_assignments.items():
            failure_stresses = []
            failure_strains = []
            m2_list = []
            m10_list = []
            m100_list = []
            m300_list = []

            for fname in file_list:
                df = processed[fname]
                if chosen_stress not in df.columns or chosen_strain not in df.columns:
                    continue

                # Failure: max stress and corresponding strain
                max_idx = df[chosen_stress].idxmax()
                failure_stresses.append(df[chosen_stress].iloc[max_idx])
                failure_strains.append(df[chosen_strain].iloc[max_idx])

                # M2, M10, M100, M300 (stress at those strain percentages)
                m2_list.append(stress_at_strain(df, 2.0))
                m10_list.append(stress_at_strain(df, 10.0))
                m100_list.append(stress_at_strain(df, 100.0))
                m300_list.append(stress_at_strain(df, 300.0))

            avg_failure_stress = np.mean(failure_stresses) if failure_stresses else np.nan
            avg_failure_strain = np.mean(failure_strains) if failure_strains else np.nan
            avg_m2 = np.mean(m2_list) if m2_list else np.nan
            avg_m10 = np.mean(m10_list) if m10_list else np.nan
            avg_m100 = np.mean(m100_list) if m100_list else np.nan
            avg_m300 = np.mean(m300_list) if m300_list else np.nan

            failure_averages[mix_key] = {
                "M2 (MPa)": avg_m2,
                "M10 (MPa)": avg_m10,
                "M100 (MPa)": avg_m100,
                "M300 (MPa)": avg_m300,
                "Failure Stress (MPa)": avg_failure_stress,
                "Failure Strain (%)": avg_failure_strain
            }

        # Build DataFrame: columns = Mix1, Mix2, …, rows = metrics
        df_avgs = pd.DataFrame(
            {mix_key: list(vals.values()) for mix_key, vals in failure_averages.items()},
            index=list(next(iter(failure_averages.values())).keys())
        )

        st.write("### Failure & M-Point Averages by Mix")
        st.dataframe(df_avgs, use_container_width=True)






with tab_data:
    st.subheader("Inspect the Cleaned DataFrames")
    for filename, df in processed.items():
        st.markdown(f"**{filename}**")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("---")

# Instron Post-Processing Tool

Welcome to the **Instron Post-Processing Tool**, a Streamlit application designed to help you analyze mechanical test data without any coding! This guide will walk you through:

- 🔧 **Installation & Launch**
- 🖥️ **User Interface Overview**
- 🔢 **Step 1: Define Mixes & Upload Samples**
- 📊 **Step 2: Choose Columns**
- 📈 **Graph Interface**
- 🔍 **Comparison Interface**
- 🔑 **Key Values Tab**
- 💾 **Download Options**
- 💡 **Tips & Best Practices**

---

## 🔧 Installation & Launch

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/instron-post-processing-tool.git
   cd instron-post-processing-tool
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser**
   Go to `http://localhost:8501` to start using the tool.

---

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

> Need help? Open an issue or send feedback—happy testing! 🚀

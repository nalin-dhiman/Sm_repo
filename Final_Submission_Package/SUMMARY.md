# Sm Cell Paper - Project Submission Package

This folder contains all the source code, data reports, and generated figures for the Sm Cell Network Analysis.

## 1. Top-Level Scripts (Root Directory)
*   **`nature_revision_pipeline.py`**: **[MAIN SCRIPT]** The primary end-to-end pipeline. Runs Theory (Phase 1), Bifurcation (Phase 2), Simulation (Phase 3), Structural Stats (Phase 4), Clock Mode (Phase 5), and Comparisons (Phase 6).
*   **`fetch_full_connectome.py`**: Utility to download and cache the full 4463-neuron connectivity matrix from FlyWire.
*   **`robustness_pipeline.py`**: Generates supplementary robustness figures.
*   **`main_paper_analysis.py`**: Legacy analysis script.

## 2. Source Code (`src/`)
Contains the core logic modules:
*   **`config.py`**: Simulation parameters (Units: pA, mV, ms).
*   **`simulation.py`**: `AdExNetwork` class (Vectorized simulation engine).
*   **`theory.py`**: Mean-field theory equations.
*   **`data_loader.py`**: FlyWire API client and matrix caching logic.
*   **`statistics.py`**: Modularity and spectral analysis tools.

## 3. Generated Figures (`figures/`)
Final publication-ready plots:
*   **`fig1_validation_robust.png`**: Empirical F-I Curve (Phase 1).
*   **`fig2_bifurcation.png`**: **[KEY RESULT]** Bifurcation Diagram showing 3 Fixed Points at J=50 pA.
*   **`fig3_scaling_robust.png`**: Memory Persistence vs Network Size (up to N=4000).
*   **`fig4c_eigenvalues.png`**: Spectral histogram of the connectome.
*   **`fig5_clock_mode.png`**: Oscillatory dynamics with high adaptation.
*   **`fig6_effectome_comparison.png`**: Linear vs Nonlinear feature comparison.

## 4. Data & Reports (`data/`)
*   **`paper_statistics_report.txt`**: Detailed log of fixed point counts, modularity p-values, and simulation probabilities.
*   **`parameters.csv`**: CSV table of physical parameters used in the simulation.

## How to Replicate
1.  Ensure you have the cached connectome (automatically handled by logic).
2.  Run: `python nature_revision_pipeline.py`
3.  Check `figures/` for outputs and `data/paper_statistics_report.txt` for logs.

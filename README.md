# MicroGeoFormer: Supplementary Code

This archive contains all code used to produce the simulation benchmark, the
real-data application, and the figures reported in "MicroGeoFormer: a
locus-attention deep learning framework for fine-scale geographic origin
assignment from sparse genetic marker panels."

## Contents

| File | Purpose |
|---|---|
| `simulate.py` | Forward-time Wright-Fisher stepping-stone population genetic simulator (microsatellite and SNP marker regimes). |
| `geomethods.py` | Core library: genotype encoding, MicroGeoFormer model (`LocusGatedMLP`, `microgeoformer_predict`), baseline methods (PCA+kNN, Locator-style MLP, GeoGenIE-style MLP), Mendelian resampling augmentation, multi-modal extensions, and learned-feature extraction. |
| `load_real_data.py`, `load_real_data2.py` | Loaders for the Ortego et al. (2015) grasshopper microsatellite dataset (`.xls` genotype/phenotype file), mapping sample localities to the geographic coordinates in Table 1 of the source publication. |
| `00_make_architecture_figure.py` | Draws Figure 1 (network architecture schematic). No data dependencies. |
| `01_simulation_benchmark_microsatellite.py` | Runs the four-method benchmark (PCA+kNN, Locator-style, GeoGenIE-style, MicroGeoFormer) on the simulated 15-locus microsatellite panel across the reference-sample-size gradient (5-55 individuals/locality). Produces `gradient_results_*.json`. |
| `02_simulation_benchmark_snp.py` | As above, for the simulated 50-locus compact SNP panel. Produces `gradient_snp_results_*.json`. |
| `03_real_data_4fold_cv.py` | Runs stratified 4-fold cross-validation on the real grasshopper data for one species at a time (pass species code `Mw`, `Ci`, or `Od` as a command-line argument), producing out-of-fold genotype-based predictions for every individual. Produces `kfold_results_{code}.pkl`. |
| `04_extract_learned_features.py` | Trains MicroGeoFormer on the full real dataset for one species and extracts the 96-dimensional penultimate-layer representation for every individual (used for Supplementary Fig. S2). Produces `learned_features_{code}.pkl`. |
| `05_make_figures.py` | Draws Figure 2, Figure 3, Supplementary Figure S1, and Supplementary Figure S2 from the JSON/pickle outputs of the scripts above. |

## Reproducing the results

```bash
pip install numpy scipy scikit-learn torch matplotlib pandas xlrd

# 1. Simulation benchmarks (microsatellite and SNP panels)
python 01_simulation_benchmark_microsatellite.py 5,15,25,35,45,55
python 02_simulation_benchmark_snp.py 5,15,25,35,45,55
# Merge the resulting per-chunk JSON files into gradient_results_merged.json
# and gradient_snp_results_merged.json (simple dict union across files; see
# comments in 05_make_figures.py for the expected structure).

# 2. Real-data cross-validation (repeat for each species)
python 03_real_data_4fold_cv.py Mw
python 03_real_data_4fold_cv.py Ci
python 03_real_data_4fold_cv.py Od

# 3. Learned-feature extraction (repeat for each species)
python 04_extract_learned_features.py Mw
python 04_extract_learned_features.py Ci
python 04_extract_learned_features.py Od

# 4. Figures
python 00_make_architecture_figure.py
python 05_make_figures.py
```

Script `03_real_data_4fold_cv.py` and `04_extract_learned_features.py` expect
the raw genotype/phenotype spreadsheet (`PhenotypicGenotypicData.xls`, from
Ortego et al. 2015, archived at Dryad, doi:10.5061/dryad.3nr2f) to be present
in the working directory and referenced by `load_real_data.py` /
`load_real_data2.py`.

Figure 3 additionally requires a province-boundary GeoJSON file for the
Toledo, Cuenca, Ciudad Real, Albacete, Madrid, and Guadalajara provinces of
Spain (`spain_provinces.geojson`), used only for basemap rendering and not
for any analysis. A suitable file was obtained from the public
`click_that_hood` administrative-boundary repository
(https://github.com/codeforgermany/click_that_hood).

## Software environment

Python 3.12, NumPy, SciPy, scikit-learn 1.8, PyTorch 2.12, Matplotlib,
pandas. All models were trained on CPU; no GPU is required to reproduce the
reported results (individual MicroGeoFormer fits complete in under 30
seconds on a standard CPU core).

## Notes on baseline implementations

The PCA+kNN, Locator-style, and GeoGenIE-style baselines in `geomethods.py`
are controlled re-implementations of each method's core published design
logic, built independently for matched comparison under identical data
splits and evaluation code. They are not verbatim runs of the original
Locator (https://github.com/kr-colab/locator) or GeoGenIE
(https://github.com/btmartin721/GeoGenIE) software packages, which include
additional features, such as automated hyperparameter search and synthetic
minority oversampling, that were not reproduced here. Researchers wishing to
compare directly against the original published tools should run them from
their respective repositories under the same data splits provided by this
codebase.

## License

Code is provided for the purpose of reproducing the results in this
manuscript. Contact the corresponding author for reuse terms.

# MicroGeoGate

A locus-attention deep learning model for predicting the geographic origin (latitude, longitude) of an individual organism from its multilocus genotype data (microsatellite or SNP). MicroGeoGate is purpose-built for **sparse genetic marker panels** — the kind of low-locus, small-reference-sample data typical of legacy agricultural pest, forestry, and wildlife genetic-monitoring programmes — and is applicable to migratory-pest source-tracing, invasive-species incursion tracing, and endangered-species provenance verification.

This repository contains the core, reusable model. For the full simulation benchmark and manuscript reproduction code, see the companion `MicroGeoGate-paper` repository.

## Key features

- **Locus-attention gate**: learns which loci carry the most geographic signal and returns a per-locus importance score, giving interpretability without a separate post hoc explainability step.
- **Mendelian-resampling augmentation**: synthesizes additional, biologically valid training genotypes from individuals collected at the same reference locality, specifically targeting small per-locality sample sizes.
- **Monte Carlo dropout uncertainty**: every prediction comes with a calibrated per-individual uncertainty estimate, not just a point estimate.
- **scikit-learn-style API**: `.fit()`, `.predict()`, `.save()`, `.load()` — no need to touch the underlying PyTorch code to use the model.

## Installation

```bash
git clone https://github.com/xinghuq/MicroGeoGate.git
cd MicroGeoGate
pip install -r requirements.txt
pip install -e .
```

Requirements: Python ≥3.9, NumPy, SciPy, scikit-learn, PyTorch ≥2.0. No GPU is required; a typical fit on a few hundred individuals and 5–20 loci completes in under a minute on a single CPU core.

## Quick start

```python
import numpy as np
from MicroGeoGate import MicroGeoGate

# X: genotypes, shape (n_individuals, n_loci, 2) — two allele calls per locus,
#    as integers (e.g. microsatellite repeat-size alleles or SNP 0/1 codes)
# Y: known geographic origin, shape (n_individuals, 2) — (latitude, longitude)
# locality_id: which reference locality each individual was sampled from,
#    shape (n_individuals,) — used only for the Mendelian-augmentation step
X = np.load("genotypes.npy")
Y = np.load("coordinates.npy")
locality_id = np.load("locality_id.npy")

# Train/test split, stratified by locality so every locality appears in both
from sklearn.model_selection import train_test_split
idx_train, idx_test = train_test_split(
    np.arange(len(X)), test_size=0.25, stratify=locality_id, random_state=0
)

model = MicroGeoGate(epochs=400)
model.fit(X[idx_train], Y[idx_train], locality_id=locality_id[idx_train])

# Point prediction + calibrated uncertainty + per-locus importance
Y_pred, Y_std, locus_importance = model.predict(
    X[idx_test], return_std=True, return_locus_importance=True
)

# Median prediction error, in kilometres, on the held-out individuals
print("Median error (km):", model.score(X[idx_test], Y[idx_test]))
```

Save and reload a trained model:

```python
model.save("MicroGeoGate_trained.pkl")
model = MicroGeoGate.load("MicroGeoGate_trained.pkl")
```

> **Note on reproducibility of predictions.** MicroGeoGate uses Monte Carlo dropout at inference to produce calibrated uncertainty estimates, so two calls to `.predict()` on the same fitted model will return very slightly different point predictions (this is expected — it is exactly what makes `Y_std` meaningful, not a bug). If you need bit-for-bit reproducible output, set `mc_samples=1` and call `torch.manual_seed()` immediately before `.predict()`.

## When to use Mendelian-resampling augmentation

Augmentation (`augment=True`, the default) is recommended whenever your reference localities have fewer than roughly 15–20 individuals each — the regime MicroGeoGate is specifically designed for. If your reference panel already has large, well-balanced samples per locality (dozens of individuals each) and a dense marker panel (hundreds of SNPs or more), a simpler classical method such as PCA + *k*-nearest-neighbours may perform as well or better at a fraction of the computational cost; see the benchmark results in the accompanying paper for guidance on when MicroGeoGate's advantage is, and is not, expected to hold.

## Handling missing data

Missing genotype calls can be encoded with any consistent sentinel integer not otherwise used as an allele value (e.g. `-999`). The encoder treats each observed value, including the sentinel, as its own allele state, so a missing call at a given locus contributes no dosage signal for that locus rather than causing an error.

## API reference

### `MicroGeoGate(hidden=96, depth=4, dropout=0.25, epochs=400, lr=1.5e-3, weight_decay=5e-4, loss_weights=(0.4, 0.6), augment=True, n_synth_per_locality=20, mc_samples=15, density_weighting=True, random_state=0)`

| Parameter | Description |
|---|---|
| `hidden` | Width of each fully connected layer in the regression backbone. |
| `depth` | Number of fully connected layers. |
| `dropout` | Dropout probability (also used for Monte Carlo uncertainty at inference). |
| `epochs` | Training epochs. |
| `lr`, `weight_decay` | Adam optimizer settings. |
| `loss_weights` | Weights `(w_mse, w_haversine)` for the combined training loss. |
| `augment` | Whether to apply Mendelian-resampling augmentation using `locality_id`. |
| `n_synth_per_locality` | Synthetic individuals generated per reference locality when `augment=True`. |
| `mc_samples` | Monte Carlo dropout passes averaged at inference. |
| `density_weighting` | Whether to weight the training loss by inverse local reference-population density. |
| `random_state` | Seed for reproducible training. |

**Methods**

- `fit(X, Y, locality_id=None, verbose=False)` — train the model.
- `predict(X, return_std=False, return_locus_importance=False)` — predict `(latitude, longitude)`, optionally with uncertainty and/or per-locus importance weights.
- `score(X, Y)` — median Haversine prediction error (km) on a labelled test set.
- `save(path)` / `MicroGeoGate.load(path)` — persist and reload a trained model.

Lower-level utilities (`DosageEncoder`, `mendelian_augment`, `haversine_km`) are also importable from `MicroGeoGate` directly if you want to build a custom pipeline.
# MicroGeoGate: Supplementary Code

This archive contains all code used to produce the simulation benchmark, the
real-data application, and the figures reported in "MicroGeoGate: a
locus-attention deep learning framework for fine-scale geographic origin
assignment from sparse genetic marker panels."

## Contents

| File | Purpose |
|---|---|
| `simulate.py` | Forward-time Wright-Fisher stepping-stone population genetic simulator (microsatellite and SNP marker regimes). |
| `geomethods.py` | Core library: genotype encoding, MicroGeoGate model (`LocusGatedMLP`, `MicroGeoGate_predict`), baseline methods (PCA+kNN, Locator-style MLP, GeoGenIE-style MLP), Mendelian resampling augmentation, multi-modal extensions, and learned-feature extraction. |
| `load_real_data.py`, `load_real_data2.py` | Loaders for the Ortego et al. (2015) grasshopper microsatellite dataset (`.xls` genotype/phenotype file), mapping sample localities to the geographic coordinates in Table 1 of the source publication. |
| `00_make_architecture_figure.py` | Draws Figure 1 (network architecture schematic). No data dependencies. |
| `01_simulation_benchmark_microsatellite.py` | Runs the four-method benchmark (PCA+kNN, Locator-style, GeoGenIE-style, MicroGeoGate) on the simulated 15-locus microsatellite panel across the reference-sample-size gradient (5-55 individuals/locality). Produces `gradient_results_*.json`. |
| `02_simulation_benchmark_snp.py` | As above, for the simulated 50-locus compact SNP panel. Produces `gradient_snp_results_*.json`. |
| `03_real_data_4fold_cv.py` | Runs stratified 4-fold cross-validation on the real grasshopper data for one species at a time (pass species code `Mw`, `Ci`, or `Od` as a command-line argument), producing out-of-fold genotype-based predictions for every individual. Produces `kfold_results_{code}.pkl`. |
| `04_extract_learned_features.py` | Trains MicroGeoGate on the full real dataset for one species and extracts the 96-dimensional penultimate-layer representation for every individual (used for Supplementary Fig. S2). Produces `learned_features_{code}.pkl`. |
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
reported results (individual MicroGeoGate fits complete in under 30
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



## Citation

If you use MicroGeoGate in your research, please cite:

> Qin., X et al. 2026. MicroGeoGate: resolving the geographic source-tracing challenge for migratory pests, biological invasions and trafficked wildlife with a locus-attention deep learning framework.

## License

MIT License (see `LICENSE`). See `NOTICE.md` for attribution of design ideas adapted from Locator (Battey et al. 2020, *eLife*) and GeoGenIE (Martin et al. 2025, *Bioinformatics Advances*), which this project benchmarks against but does not reuse code from.

Code is provided for the purpose of reproducing the results in this
manuscript. Contact the corresponding author for reuse terms.
qinxinghu@gmail.com

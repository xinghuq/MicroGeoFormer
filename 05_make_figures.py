"""
Reproduces Figures 1-3 and Supplementary Figures S1-S2 of the manuscript
"MicroGeoGate: a locus-attention deep learning framework for fine-scale
geographic origin assignment from sparse genetic marker panels".

This script assumes the outputs of scripts 01-04 (JSON result files and
pickled cross-validation predictions) are present in the working directory.
Figure-drawing code is provided in full below; see comments for which
upstream script produces each required input file.

Figure 1 (network architecture) is a hand-drawn schematic and does not
depend on analysis outputs; its drawing code is included at the end of
this file for completeness.
"""
import json, pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Ellipse, Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# Figure 2: simulation benchmark bar chart with significance annotations
# Requires: gradient_results_merged.json (produced by 01_simulation_benchmark_microsatellite.py)
# ----------------------------------------------------------------------
def make_figure2():
    with open('gradient_results_merged.json') as f:
        msat = json.load(f)
    levels = sorted(msat.keys(), key=lambda x: int(x))
    methods = ['PCA_kNN', 'Locator_MLP', 'GeoGenIE_style', 'MicroGeoGate']
    labels = {'PCA_kNN': 'PCA+kNN (KLFDAPC-like)', 'Locator_MLP': 'Locator-style MLP',
              'GeoGenIE_style': 'GeoGenIE-style MLP', 'MicroGeoGate': 'MicroGeoGate (proposed)'}
    colors = {'PCA_kNN': '#1b4f72', 'Locator_MLP': '#186a3b', 'GeoGenIE_style': '#b9770e', 'MicroGeoGate': '#922b21'}

    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(levels))
    width = 0.2
    offsets = {'PCA_kNN': -1.5*width, 'Locator_MLP': -0.5*width, 'GeoGenIE_style': 0.5*width, 'MicroGeoGate': 1.5*width}

    for m in methods:
        means = np.array([np.mean(msat[l][m]) for l in levels])
        sems = np.array([np.std(msat[l][m])/np.sqrt(5) for l in levels])
        ax.bar(x+offsets[m], means, width, yerr=sems, color=colors[m], capsize=3,
               label=labels[m].replace('\n', ' '), edgecolor='white', linewidth=0.4)

    for i, l in enumerate(levels):
        ours = msat[l]['MicroGeoGate']
        for m in ['PCA_kNN', 'Locator_MLP', 'GeoGenIE_style']:
            comp = msat[l][m]
            _, p = ttest_ind(ours, comp)
            if p < 0.05:
                sig = '**' if p < 0.01 else '*'
                y_top = np.mean(comp) + np.std(comp)/np.sqrt(5)
                ax.text(x[i]+offsets[m], y_top+1.0, sig, ha='center', fontsize=11, color='#333333', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'n={l}' for l in levels], fontsize=10)
    ax.set_xlabel('Training individuals per reference locality', fontsize=11)
    ax.set_ylabel('Prediction error (km)', fontsize=11)
    ax.set_ylim(0, 68)
    ax.legend(fontsize=8.5, loc='upper right', ncol=2, framealpha=0.95)
    plt.tight_layout()
    plt.savefig('Figure2.png', dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------------------
# Supplementary Figure S1: same, for the SNP panel
# Requires: gradient_snp_results_merged.json (produced by 02_simulation_benchmark_snp.py)
# ----------------------------------------------------------------------
def make_figureS1():
    with open('gradient_snp_results_merged.json') as f:
        snp = json.load(f)
    levels = sorted(snp.keys(), key=lambda x: int(x))
    methods = ['PCA_kNN', 'Locator_MLP', 'GeoGenIE_style', 'MicroGeoGate']
    labels = {'PCA_kNN': 'PCA+kNN (KLFDAPC-like)', 'Locator_MLP': 'Locator-style MLP',
              'GeoGenIE_style': 'GeoGenIE-style MLP', 'MicroGeoGate': 'MicroGeoGate (proposed)'}
    colors = {'PCA_kNN': '#1b4f72', 'Locator_MLP': '#186a3b', 'GeoGenIE_style': '#b9770e', 'MicroGeoGate': '#922b21'}

    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(levels))
    width = 0.2
    offsets = {'PCA_kNN': -1.5*width, 'Locator_MLP': -0.5*width, 'GeoGenIE_style': 0.5*width, 'MicroGeoGate': 1.5*width}
    for m in methods:
        means = np.array([np.mean(snp[l][m]) for l in levels])
        sems = np.array([np.std(snp[l][m])/np.sqrt(5) for l in levels])
        ax.bar(x+offsets[m], means, width, yerr=sems, color=colors[m], capsize=3, label=labels[m], edgecolor='white', linewidth=0.4)
    for i, l in enumerate(levels):
        ours = snp[l]['MicroGeoGate']
        for m in ['PCA_kNN', 'Locator_MLP', 'GeoGenIE_style']:
            comp = snp[l][m]
            _, p = ttest_ind(ours, comp)
            if p < 0.05 and np.mean(ours) < np.mean(comp):
                sig = '**' if p < 0.01 else '*'
                y_top = np.mean(comp) + np.std(comp)/np.sqrt(5)
                ax.text(x[i]+offsets[m], y_top+0.8, sig, ha='center', fontsize=11, color='#333333', fontweight='bold')
            elif p < 0.05 and np.mean(ours) > np.mean(comp):
                y_top = np.mean(ours) + np.std(ours)/np.sqrt(5)
                ax.text(x[i]+offsets['MicroGeoGate'], y_top+0.8, '#', ha='center', fontsize=10, color='#922b21', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'n={l}' for l in levels], fontsize=10)
    ax.set_xlabel('Training individuals per reference locality', fontsize=11)
    ax.set_ylabel('Prediction error (km)', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right', ncol=2, framealpha=0.95)
    plt.tight_layout()
    plt.savefig('SupplementaryFigure1.png', dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------------------
# Figure 3: real-data maps with median-error radius circles
# Requires: kfold_results_{Mw,Ci,Od}.pkl (produced by 03_real_data_4fold_cv.py)
#           spain_provinces.geojson (province boundary data; see README for source)
# ----------------------------------------------------------------------
def geom_to_patch(geom, **kwargs):
    if geom['type'] == 'Polygon':
        polys = [geom['coordinates']]
    else:
        polys = geom['coordinates']
    verts, codes = [], []
    for poly in polys:
        outer = poly[0]; n = len(outer)
        verts += outer
        codes += [Path.MOVETO] + [Path.LINETO]*(n-2) + [Path.CLOSEPOLY]
    return PathPatch(Path(verts, codes), **kwargs)

def make_figure3():
    with open('spain_provinces.geojson') as f:
        gj = json.load(f)
    target_provinces = ['Toledo', 'Cuenca', 'Ciudad Real', 'Albacete', 'Madrid', 'Guadalajara']
    prov_geoms = {feat['properties']['name']: feat['geometry'] for feat in gj['features']
                  if feat['properties']['name'] in target_provinces}

    SPECIES = [('Mw', 'Mioscirtus wagneri', 'A', '#1b4f72', '#922b21'),
               ('Ci', 'Calliptamus italicus', 'B', '#186a3b', '#922b21'),
               ('Od', 'Oedaleus decorus', 'C', '#6c3483', '#922b21')]

    XMIN, XMAX, YMIN, YMAX = -4.05, -2.55, 39.15, 40.15
    lat0 = 39.7
    aspect = 1.0/np.cos(np.radians(lat0))
    KM_PER_DEG_LAT = 111.0
    KM_PER_DEG_LON = 111.0*np.cos(np.radians(lat0))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
    for ax, (code, latin, panel, ctrue, cpred) in zip(axes, SPECIES):
        with open(f'kfold_results_{code}.pkl', 'rb') as f:
            d = pickle.load(f)['geno']
        true, pred, err = d['true'], d['pred'], d['err']
        med_err_km = np.median(err)
        r_lat = med_err_km / KM_PER_DEG_LAT
        r_lon = med_err_km / KM_PER_DEG_LON

        for name, geom in prov_geoms.items():
            core = name in ['Toledo', 'Cuenca', 'Ciudad Real']
            ax.add_patch(geom_to_patch(geom, facecolor='#d9d9d9' if core else '#ececec',
                                        edgecolor='#999999', linewidth=1.0 if core else 0.6, zorder=1))
        uniq_coords = np.array(sorted(set([tuple(c) for c in true.tolist()])))
        for lat_c, lon_c in uniq_coords:
            ell = Ellipse((lon_c, lat_c), width=2*r_lon, height=2*r_lat, facecolor='#a9d4f0',
                          edgecolor='#5fa8d3', linewidth=0.8, alpha=0.4, zorder=2)
            ax.add_patch(ell)
        ax.scatter(true[:, 1], true[:, 0], s=24, c=ctrue, marker='o', alpha=0.85, zorder=4, linewidths=0)
        ax.scatter(pred[:, 1], pred[:, 0], s=30, c=cpred, marker='x', linewidths=1.1, alpha=0.8, zorder=3)
        ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
        ax.set_aspect(aspect, adjustable='box')
        ax.set_xlabel('Longitude', fontsize=9.5)
        ax.text(-0.03, 1.03, panel, transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')
        ax.text(0.97, 0.965, f'Median error:\n{med_err_km:.1f} km', transform=ax.transAxes,
                fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#999999', alpha=0.95))
        ax.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor=ctrue, markersize=8, label='Sampled location'),
            Line2D([0], [0], marker='x', color=cpred, markersize=8, linestyle='None', markeredgewidth=1.4, label='Predicted location'),
        ], fontsize=8, loc='lower left', framealpha=0.92)
    axes[0].set_ylabel('Latitude', fontsize=9.5)
    plt.tight_layout()
    plt.savefig('Figure3.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ----------------------------------------------------------------------
# Supplementary Figure S2: learned-feature PCA colored by locality
# Requires: learned_features_{Mw,Ci,Od}.pkl (produced by 04_extract_learned_features.py)
# ----------------------------------------------------------------------
def make_figureS2():
    SPECIES = [('Mw', 'Mioscirtus wagneri'), ('Ci', 'Calliptamus italicus'), ('Od', 'Oedaleus decorus')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    for ax, (code, latin) in zip(axes, SPECIES):
        with open(f'learned_features_{code}.pkl', 'rb') as f:
            d = pickle.load(f)
        feats, locs = d['feats'], d['locs']
        pca = PCA(n_components=2).fit(feats)
        P_ = pca.transform(feats)
        uniq_locs = sorted(set(locs.tolist()))
        cmap = plt.get_cmap('tab20', len(uniq_locs))
        for i, loc in enumerate(uniq_locs):
            mask = locs == loc
            ax.scatter(P_[mask, 0], P_[mask, 1], s=22, color=cmap(i), alpha=0.8, edgecolors='none')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9.5)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9.5)
        ax.text(0.02, 1.10, latin, transform=ax.transAxes, fontsize=11, fontstyle='italic', fontweight='bold')
        ax.text(0.02, 1.03, f'{len(uniq_locs)} reference localities (colours), n={len(locs)}',
                transform=ax.transAxes, fontsize=8, color='#555555')
    plt.tight_layout()
    plt.savefig('SupplementaryFigure2.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    make_figure2()
    make_figureS1()
    make_figure3()
    make_figureS2()
    print("All figures written to working directory.")

"""
Draws Figure 1 (MicroGeoGate network architecture schematic).
No data dependencies; produces Figure1.png directly.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import numpy as np

fig, ax = plt.subplots(figsize=(15, 6.5))
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 6.8)
ax.axis('off')

def stack3d(x, y, w, h, n, dx=0.10, dy=0.06, fc='#a9d4f0', ec='#2c6e91', lw=0.9):
    for i in range(n):
        xi = x + i*dx
        yi = y + i*dy
        rect = Rectangle((xi, yi), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=10+i)
        ax.add_patch(rect)

def arrow(x1, y1, x2, y2, color='#333333', lw=1.4):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, color=color, lw=lw, zorder=30)
    ax.add_patch(a)

rect = Rectangle((0.3, 2.0), 1.5, 2.0, facecolor='#7a4fa3', edgecolor='#3d2657', linewidth=1.2, zorder=10)
ax.add_patch(rect)
ax.text(1.05, 4.35, 'Input genotype', ha='center', fontsize=9, fontweight='bold')
ax.text(1.05, 1.75, 'L loci x 2 alleles', ha='center', fontsize=8, color='#333333')

arrow(1.85, 3.0, 2.5, 3.0)
ax.text(2.15, 3.3, 'Allele-\ndosage\nencoding', ha='center', fontsize=6.8, color='#444444')

stack3d(2.55, 1.9, 1.0, 1.9, 7, dx=0.09, dy=0.055, fc='#bfe0f7', ec='#2c6e91')
ax.text(3.55, 4.35, 'Per-locus dosage\nvectors', ha='center', fontsize=8.7, fontweight='bold')
ax.text(3.55, 1.6, 'L loci (0/1/2 dosage)', ha='center', fontsize=7.6, color='#333333')

arrow(4.35, 3.0, 5.15, 3.0)
ax.text(4.75, 3.35, 'Layer 1:\nlocus-\nattention\ngate', ha='center', fontsize=6.8, color='#444444')

stack3d(5.2, 2.0, 0.9, 1.75, 7, dx=0.085, dy=0.05, fc='#fbdca0', ec='#b9770e')
ax.text(6.15, 4.35, 'Gated locus\nvectors (w_l x dosage)', ha='center', fontsize=8.5, fontweight='bold')
ax.text(6.15, 1.75, 'softmax importance\nweights', ha='center', fontsize=7.3, color='#333333')

arrow(6.9, 3.0, 7.7, 3.0)
ax.text(7.3, 3.3, 'Concat', ha='center', fontsize=7.2, color='#444444')

rect = Rectangle((7.75, 1.9), 0.32, 2.2, facecolor='#bfe0f7', edgecolor='#2c6e91', linewidth=1.1, zorder=10)
ax.add_patch(rect)
ax.text(7.91, 4.35, 'Flattening\nlayer', ha='center', fontsize=8.3, fontweight='bold')
ax.text(7.91, 1.65, 'D_in', ha='center', fontsize=7.8, color='#333333')

for yy in np.linspace(2.0, 3.9, 5):
    ax.plot([6.9, 7.75], [2.0 + (yy-2.9)*0.15+0.95, yy], color='#888888', lw=0.5, zorder=5)

arrow(8.15, 3.0, 8.75, 3.0)

layer_x = [9.1, 10.3, 11.5, 12.7]
n_nodes = 6
node_ys = np.linspace(1.3, 4.7, n_nodes)
node_positions = []
for lx in layer_x:
    ys = node_ys + np.random.default_rng(int(lx*10)).uniform(-0.05, 0.05, n_nodes)
    node_positions.append(ys)

for li in range(len(layer_x)-1):
    for y1 in node_positions[li]:
        for y2 in node_positions[li+1]:
            ax.plot([layer_x[li], layer_x[li+1]], [y1, y2], color='#c9c9c9', lw=0.35, zorder=5)
for y2 in node_positions[0]:
    ax.plot([8.75, layer_x[0]], [3.0, y2], color='#c9c9c9', lw=0.35, zorder=5)

for li, lx in enumerate(layer_x):
    for y in node_positions[li]:
        ax.add_patch(Circle((lx, y), 0.16, facecolor='#bfe0f7', edgecolor='#2c6e91', linewidth=1.0, zorder=20))
    ax.text(lx, 5.05, 'Dense(96)\n+ELU+Dropout' if li > 0 else 'Dense(D_in->96)\n+ELU+Dropout', ha='center', fontsize=6.6)

out_x = 14.1
out_ys = [3.5, 2.5]
out_labels = ['Latitude', 'Longitude']
for y2 in out_ys:
    for y1 in node_positions[-1]:
        ax.plot([layer_x[-1], out_x], [y1, y2], color='#c9c9c9', lw=0.4, zorder=5)
for y, lab in zip(out_ys, out_labels):
    ax.add_patch(Circle((out_x, y), 0.18, facecolor='#f7c59f', edgecolor='#b9530e', linewidth=1.1, zorder=20))
    ax.text(out_x+0.45, y, lab, ha='left', va='center', fontsize=9.5, fontweight='bold')
ax.text(out_x, 5.05, 'Output\nDense(96->2)', ha='center', fontsize=7.2)

plt.tight_layout()
plt.savefig('Figure1.png', dpi=300, bbox_inches='tight')
print("Figure1.png written.")

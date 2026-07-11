"""
Paired significance tests for the architecture comparison (Supplementary
Table S2 / Figure S3): MicroGeoGate vs the two Transformer variants.

The three models are evaluated on the SAME five train/test splits per
sample-size level, so a PAIRED t-test is appropriate. With only five
replicate splits the Wilcoxon signed-rank test cannot reach P<0.05 even
when all five agree (minimum attainable P = 0.0625), so paired t-tests
are reported as indicative.

Input: transformer_benchmark_final.json produced by
06_transformer_architecture_benchmark.py
"""
import json, numpy as np
from scipy import stats

tf = json.load(open('transformer_benchmark_final.json'))
levels = [5, 15, 25, 35, 45, 55]

def paired_t(a, b):
    return stats.ttest_rel(np.array(a), np.array(b)).pvalue

def star(p):
    return '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

print(f"{'n':>4} | {'mgg':>6} {'v1':>6} {'v2':>6} | {'v1 vs mgg':>16} | {'v2 vs mgg':>16}")
print("-" * 66)
for l in levels:
    mgg, v1, v2 = tf[str(l)]['mgg'], tf[str(l)]['v1'], tf[str(l)]['v2']
    p1, p2 = paired_t(mgg, v1), paired_t(mgg, v2)
    print(f"{l:>4} | {np.mean(mgg):>6.1f} {np.mean(v1):>6.1f} {np.mean(v2):>6.1f} | "
          f"{star(p1):>3} (P={p1:>5.3f}) | {star(p2):>3} (P={p2:>5.3f})")

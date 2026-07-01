import sys, numpy as np, pickle, time
import geomethods as gm
from load_real_data import load_species

SHEETS = {'Mw':'Mioscirtus_wagneri','Ci':'Calliptamus_italicus','Od':'Oedaleus_decorus'}

def deme_ids(locs):
    uniq = sorted(set(locs.tolist())); m = {u:i for i,u in enumerate(uniq)}
    return np.array([m[l] for l in locs])

code = sys.argv[1]
sheet = SHEETS[code]
X, Y, locs, ids, n_loci = load_species(sheet)
D = deme_ids(locs)
t0=time.time()
feats = gm.microgeoformer_extract_features(X, Y, D, epochs=400, n_synth_per_deme=15)
print(code, 'features shape', feats.shape, 'time', round(time.time()-t0,1))
with open(f'learned_features_{code}.pkl','wb') as f:
    pickle.dump(dict(feats=feats, Y=Y, locs=locs), f)

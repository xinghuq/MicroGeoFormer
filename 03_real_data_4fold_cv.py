import sys, numpy as np, pickle, json
from sklearn.model_selection import StratifiedKFold
import geomethods as gm
from load_real_data2 import load_species

SHEETS = {'Mw':'Mioscirtus_wagneri','Ci':'Calliptamus_italicus','Od':'Oedaleus_decorus'}

def deme_ids(locs):
    uniq = sorted(set(locs.tolist())); m = {u:i for i,u in enumerate(uniq)}
    return np.array([m[l] for l in locs])

code = sys.argv[1]
sheet = SHEETS[code]
X, Y, locs, ids, femur, sex, n_loci = load_species(sheet)
D = deme_ids(locs)
male_mask = (sex=='Male') & (~np.isnan(femur))

K = 4
out = {}

# ---- genotype-only, full sample, K-fold out-of-fold predictions ----
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
pred_geno = np.zeros_like(Y)
for tr_idx, te_idx in skf.split(X, D):
    Xtr,Ytr,Dtr = X[tr_idx],Y[tr_idx],D[tr_idx]
    Xte = X[te_idx]
    p,_,_ = gm.multimodal_geo_predict(Xtr,Ytr,Xte,Dtr=Dtr,mode='geno',n_synth_per_deme=20,epochs=300)
    pred_geno[te_idx] = p
err_geno = gm.haversine_km(Y, pred_geno)
out['geno'] = dict(true=Y, pred=pred_geno, err=err_geno, ids=ids, locs=locs)
print(code, 'genotype-only median err (full K-fold):', np.median(err_geno), flush=True)

# ---- males-only subset: genotype-only, phenotype-only, joint ----
Xm, Ym, Dm, Pm = X[male_mask], Y[male_mask], D[male_mask], femur[male_mask].reshape(-1,1)
idsm, locsm = ids[male_mask], locs[male_mask]
skf2 = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
pred_g_m = np.zeros_like(Ym); pred_p_m = np.zeros_like(Ym); pred_j_m = np.zeros_like(Ym)
for tr_idx, te_idx in skf2.split(Xm, Dm):
    Xtr,Ytr,Dtr,Ptr = Xm[tr_idx],Ym[tr_idx],Dm[tr_idx],Pm[tr_idx]
    Xte,Pte = Xm[te_idx],Pm[te_idx]
    p,_,_ = gm.multimodal_geo_predict(Xtr,Ytr,Xte,Dtr=Dtr,mode='geno',n_synth_per_deme=20,epochs=300)
    pred_g_m[te_idx] = p
    p,_,_ = gm.multimodal_geo_predict(None,Ytr,None,Dtr=Dtr,Ptr=Ptr,Pte=Pte,mode='pheno',n_synth_per_deme=20,epochs=300)
    pred_p_m[te_idx] = p
    p,_,_ = gm.multimodal_geo_predict(Xtr,Ytr,Xte,Dtr=Dtr,Ptr=Ptr,Pte=Pte,mode='joint',n_synth_per_deme=20,epochs=300)
    pred_j_m[te_idx] = p

out['male_subset'] = dict(true=Ym, pred_geno=pred_g_m, pred_pheno=pred_p_m, pred_joint=pred_j_m,
                           err_geno=gm.haversine_km(Ym,pred_g_m), err_pheno=gm.haversine_km(Ym,pred_p_m),
                           err_joint=gm.haversine_km(Ym,pred_j_m), ids=idsm, locs=locsm, femur=Pm[:,0])
print(code, 'males n=', len(Ym),
      'geno med', np.median(out['male_subset']['err_geno']),
      'pheno med', np.median(out['male_subset']['err_pheno']),
      'joint med', np.median(out['male_subset']['err_joint']), flush=True)

with open(f'kfold_results_{code}.pkl','wb') as f:
    pickle.dump(out, f)

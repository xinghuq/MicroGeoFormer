import sys, pickle, numpy as np, json, time
import geomethods as gm

with open('sim_data_gradient.pkl','rb') as f:
    data = pickle.load(f)
X,Y,D = data['X'],data['Y'],data['D']

def split(X,Y,D,n_train,n_test,seed):
    rng=np.random.default_rng(seed)
    demes=np.unique(D)
    tr,te=[],[]
    for d in demes:
        m=np.nonzero(D==d)[0]; rng.shuffle(m)
        te+=m[:n_test].tolist(); tr+=m[n_test:n_test+n_train].tolist()
    return np.array(tr), np.array(te)

levels = [int(x) for x in sys.argv[1].split(',')]
n_reps = 5
methods = ['PCA_kNN','Locator_MLP','GeoGenIE_style','MicroGeoFormer']
results = {lvl: {m: [] for m in methods} for lvl in levels}

for lvl in levels:
    for rep in range(n_reps):
        tr, te = split(X,Y,D,lvl,5,seed=rep)
        Xtr,Ytr,Dtr,Xte,Yte = X[tr],Y[tr],D[tr],X[te],Y[te]

        p = gm.baseline_pca_knn(Xtr,Ytr,Xte,k=min(5,len(Ytr)))
        results[lvl]['PCA_kNN'].append(float(np.median(gm.haversine_km(Yte,p))))

        p = gm.baseline_locator_mlp(Xtr,Ytr,Xte,n_ensemble=5,epochs=150)
        results[lvl]['Locator_MLP'].append(float(np.median(gm.haversine_km(Yte,p))))

        p = gm.baseline_geogenie_style(Xtr,Ytr,Xte,epochs=250)
        results[lvl]['GeoGenIE_style'].append(float(np.median(gm.haversine_km(Yte,p))))

        p,_,_ = gm.microgeoformer_predict(Xtr,Ytr,Xte,Dtr=Dtr,n_synth_per_deme=15,epochs=220)
        results[lvl]['MicroGeoFormer'].append(float(np.median(gm.haversine_km(Yte,p))))
        print(lvl,'rep',rep,'done',flush=True)

with open(f'gradient_results_{"_".join(map(str,levels))}.json','w') as f:
    json.dump(results, f, indent=2)
print('DONE', levels)

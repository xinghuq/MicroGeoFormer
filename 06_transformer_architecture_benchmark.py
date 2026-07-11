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
methods = ['Transformer_v1_selfattn','Transformer_v2_lightattn','MicroGeoGate']
results = {lvl: {m: [] for m in methods} for lvl in levels}

t0=time.time()
for lvl in levels:
    for rep in range(n_reps):
        tr, te = split(X,Y,D,lvl,5,seed=rep)
        Xtr,Ytr,Dtr,Xte,Yte = X[tr],Y[tr],D[tr],X[te],Y[te]

        p = gm.baseline_transformer_v1(Xtr,Ytr,Xte,Dtr=Dtr,epochs=220)
        results[lvl]['Transformer_v1_selfattn'].append(float(np.median(gm.haversine_km(Yte,p))))

        p = gm.baseline_transformer_v2(Xtr,Ytr,Xte,Dtr=Dtr,epochs=220)
        results[lvl]['Transformer_v2_lightattn'].append(float(np.median(gm.haversine_km(Yte,p))))

        p,_,_ = gm.MicroGeoGate_predict(Xtr,Ytr,Xte,Dtr=Dtr,n_synth_per_deme=15,epochs=220)
        results[lvl]['MicroGeoGate'].append(float(np.median(gm.haversine_km(Yte,p))))
        print(lvl,'rep',rep,'done  t=%.0fs'%(time.time()-t0),flush=True)

with open(f'transformer_results_{"_".join(map(str,levels))}.json','w') as f:
    json.dump(results, f, indent=2)
print('DONE', levels)

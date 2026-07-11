import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(4)
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

EARTH_R = 6371.0088

def haversine_km(y_true, y_pred):
    lat1, lon1 = np.radians(y_true[:,0]), np.radians(y_true[:,1])
    lat2, lon2 = np.radians(y_pred[:,0]), np.radians(y_pred[:,1])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*EARTH_R*np.arcsin(np.sqrt(np.clip(a,0,1)))

def haversine_loss_torch(y_true, y_pred):
    lat1 = torch.deg2rad(y_true[:,0]); lon1 = torch.deg2rad(y_true[:,1])
    lat2 = torch.deg2rad(y_pred[:,0]); lon2 = torch.deg2rad(y_pred[:,1])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    a = torch.clamp(a, 1e-12, 1-1e-12)
    d = 2*EARTH_R*torch.arcsin(torch.sqrt(a))
    return d.mean()

class DosageEncoder:
    def __init__(self):
        self.locus_alleles = None
    def fit(self, X):
        n, n_loci, _ = X.shape
        self.locus_alleles = []
        for l in range(n_loci):
            alleles = np.unique(X[:,l,:])
            self.locus_alleles.append(alleles)
        return self
    def transform(self, X):
        n, n_loci, _ = X.shape
        feats = []
        for l in range(n_loci):
            alleles = self.locus_alleles[l]
            na = len(alleles)
            dos = np.zeros((n, na), dtype=np.float32)
            for c in range(2):
                col = X[:, l, c]
                pos = np.searchsorted(alleles, col)
                pos_clip = np.clip(pos, 0, na-1)
                valid = (pos_clip < na) & (alleles[pos_clip] == col)
                rows = np.nonzero(valid)[0]
                np.add.at(dos, (rows, pos_clip[valid]), 1.0)
            feats.append(dos)
        return np.concatenate(feats, axis=1)
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def encode_genotypes_grouped(X):
    enc = DosageEncoder().fit(X)
    F = enc.transform(X)
    ranges = []
    pos = 0
    for l in range(X.shape[1]):
        na = len(enc.locus_alleles[l])
        ranges.append((pos, pos+na))
        pos += na
    return F, ranges, enc

def baseline_pca_knn(Xtr_geno, Ytr, Xte_geno, k=5, n_pc=10, use_lfda=True):
    enc = DosageEncoder().fit(Xtr_geno)
    Ftr = enc.transform(Xtr_geno); Fte = enc.transform(Xte_geno)
    n_pc_use = min(n_pc, Ftr.shape[1]-1, Ftr.shape[0]-1)
    pca = PCA(n_components=max(2,n_pc_use)).fit(Ftr)
    Ptr, Pte = pca.transform(Ftr), pca.transform(Fte)
    if use_lfda:
        from sklearn.cluster import KMeans
        k_clusters = min(15, len(Ytr)//3) if len(Ytr) >= 6 else 2
        km = KMeans(n_clusters=max(2,k_clusters), n_init=5, random_state=0).fit(Ytr)
        labels = km.labels_
        try:
            lda = LinearDiscriminantAnalysis(n_components=min(len(set(labels))-1, Ptr.shape[1]))
            lda.fit(Ptr, labels)
            Ptr2, Pte2 = lda.transform(Ptr), lda.transform(Pte)
            Ptr, Pte = np.concatenate([Ptr, Ptr2],1), np.concatenate([Pte, Pte2],1)
        except Exception:
            pass
    knn = KNeighborsRegressor(n_neighbors=min(k, len(Ytr)), weights='distance')
    knn.fit(Ptr, Ytr)
    return knn.predict(Pte)

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, depth=3, out_dim=2, dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_mlp(Xtr, Ytr, Xte, loss_fn='mse', epochs=300, lr=1e-3, hidden=64, depth=3,
              weight_decay=1e-4, sample_weights=None):
    mu, sd = Ytr.mean(0), Ytr.std(0)+1e-8
    Ytr_n = (Ytr-mu)/sd
    model = SimpleMLP(Xtr.shape[1], hidden=hidden, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr_n, dtype=torch.float32)
    Ytrue = torch.tensor(Ytr, dtype=torch.float32)
    w = torch.tensor(sample_weights, dtype=torch.float32) if sample_weights is not None else None
    mu_t = torch.tensor(mu, dtype=torch.float32); sd_t = torch.tensor(sd, dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred_n = model(Xt)
        if loss_fn == 'mse':
            loss = ((pred_n-Yt)**2)
            loss = (loss.mean(1)*w).mean() if w is not None else loss.mean()
        else:
            pred = pred_n*sd_t+mu_t
            loss = haversine_loss_torch(Ytrue, pred)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.tensor(Xte, dtype=torch.float32)).numpy()
    return pred_n*sd+mu

def baseline_locator_mlp(Xtr_geno, Ytr, Xte_geno, n_ensemble=8, loci_frac=0.8, epochs=250):
    _, ranges, enc = encode_genotypes_grouped(Xtr_geno)
    Ftr = enc.transform(Xtr_geno); Fte = enc.transform(Xte_geno)
    n_loci = Xtr_geno.shape[1]
    rng = np.random.default_rng(0)
    preds = []
    for e in range(n_ensemble):
        sel_loci = rng.choice(n_loci, size=max(2,int(n_loci*loci_frac)), replace=False)
        cols = np.concatenate([np.arange(ranges[l][0], ranges[l][1]) for l in sel_loci])
        p = train_mlp(Ftr[:,cols], Ytr, Fte[:,cols], loss_fn='mse', epochs=epochs, hidden=64, depth=3)
        preds.append(p)
    preds = np.stack(preds,0)
    return np.median(preds,0)

def baseline_geogenie_style(Xtr_geno, Ytr, Xte_geno, epochs=400):
    enc = DosageEncoder().fit(Xtr_geno)
    Ftr = enc.transform(Xtr_geno); Fte = enc.transform(Xte_geno)
    from sklearn.neighbors import NearestNeighbors
    k = min(6, len(Ytr)-1)
    nn_ = NearestNeighbors(n_neighbors=k).fit(Ytr)
    dist,_ = nn_.kneighbors(Ytr)
    density = 1.0/(dist.mean(1)+1e-3)
    w = (1.0/density); w = w/w.mean()
    pred = train_mlp(Ftr, Ytr, Fte, loss_fn='mse', epochs=epochs, hidden=96, depth=4,
                      weight_decay=1e-3, sample_weights=w)
    return pred

class LocusGatedMLP(nn.Module):
    """Hybrid architecture: (1) a lightweight per-locus attention gate learns which loci
    carry the strongest geographic signal (interpretable, SHAP-like weighting) and
    rescales each locus's allele-dosage block accordingly; (2) the gated features feed a
    deep residual MLP regression backbone (in the spirit of GeoGenIE's tuned MLP) trained
    with a combined normalized-MSE + Haversine loss."""
    def __init__(self, ranges, hidden=96, depth=4, out_dim=2, dropout=0.25):
        super().__init__()
        self.ranges = ranges
        n_loci = len(ranges)
        gate_hidden = 8
        self.locus_gate = nn.ModuleList([nn.Sequential(
            nn.Linear(2, gate_hidden), nn.Tanh(), nn.Linear(gate_hidden, 1)) for _ in ranges])
        in_dim = ranges[-1][1]
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ELU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)
    def forward(self, x, return_hidden=False):  # x: (batch, total_dosage_features)
        n_loci = len(self.ranges)
        gates = []
        gated_blocks = []
        for i,(s,e) in enumerate(self.ranges):
            block = x[:, s:e]
            summary = torch.stack([block.mean(1), block.max(1).values], dim=1)  # (batch,2)
            g = self.locus_gate[i](summary)  # (batch,1)
            gates.append(g)
        gates_cat = torch.cat(gates, dim=1)  # (batch, n_loci)
        gate_w = torch.softmax(gates_cat, dim=1) * n_loci  # rescaled so mean weight ~1
        for i,(s,e) in enumerate(self.ranges):
            gated_blocks.append(x[:, s:e] * gate_w[:, i:i+1])
        xg = torch.cat(gated_blocks, dim=1)
        h = self.backbone(xg)
        out = self.head(h)
        if return_hidden:
            return out, gate_w, h
        return out, gate_w

def MicroGeoGate_predict(Xtr_geno, Ytr, Xte_geno, Dtr=None, epochs=400,
                            augment=True, n_synth_per_deme=15, mc_samples=15,
                            hidden=96, depth=4):
    """MicroGeoGate: locus-attention-gated deep MLP for geographic-origin prediction
    from microsatellite / small SNP panels, with Mendelian-resampling augmentation for
    sparse reference panels and MC-dropout uncertainty quantification."""
    _, ranges, enc = encode_genotypes_grouped(Xtr_geno)

    Xaug, Yaug = Xtr_geno.copy(), Ytr.copy()
    if augment and Dtr is not None:
        Xs, Ys, _ = mendelian_augment(Xtr_geno, Ytr, Dtr, n_synth_per_deme=n_synth_per_deme)
        if len(Xs):
            Xaug = np.concatenate([Xtr_geno, Xs], axis=0)
            Yaug = np.concatenate([Ytr, Ys], axis=0)

    Faug = enc.transform(Xaug)
    Fte = enc.transform(Xte_geno)

    mu, sd = Yaug.mean(0), Yaug.std(0)+1e-8
    Yn = (Yaug-mu)/sd
    mu_t = torch.tensor(mu, dtype=torch.float32); sd_t = torch.tensor(sd, dtype=torch.float32)

    # density-inverted sample weighting (as in GeoGenIE) applied on top of our loss
    from sklearn.neighbors import NearestNeighbors
    k = min(6, len(Yaug)-1)
    nnb = NearestNeighbors(n_neighbors=k).fit(Yaug)
    dist,_ = nnb.kneighbors(Yaug)
    dens = 1.0/(dist.mean(1)+1e-3)
    w = (1.0/dens); w = w/w.mean()
    w_t = torch.tensor(w, dtype=torch.float32)

    model = LocusGatedMLP(ranges, hidden=hidden, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    Xb = torch.tensor(Faug, dtype=torch.float32)
    Yb_n = torch.tensor(Yn, dtype=torch.float32)
    Yb_true = torch.tensor(Yaug, dtype=torch.float32)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out_n, gate = model(Xb)
        out = out_n*sd_t+mu_t
        loss_mse = (((out_n-Yb_n)**2).mean(1)*w_t).mean()
        loss_hav = haversine_loss_torch(Yb_true, out)
        loss = 0.4*loss_mse + 0.6*(loss_hav/1000.0)
        loss.backward(); opt.step()

    model.train()  # MC-dropout active
    preds, gates_all = [], []
    Xte_t = torch.tensor(Fte, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(mc_samples):
            out_n, gate = model(Xte_t)
            preds.append(out_n.numpy()*sd+mu)
            gates_all.append(gate.numpy().mean(0))
    preds = np.stack(preds,0)
    return np.median(preds,0), preds.std(0), np.mean(gates_all,axis=0)

def mendelian_augment(X_geno, Y, deme_id, n_synth_per_deme=40, seed=0, extra_feats=None):
    """As before, but optionally also resamples an auxiliary per-individual feature matrix
    (e.g., phenotype) the same way, drawing from the same random parent pair per synthetic
    offspring (parent-average, i.e. mid-parent value plus small noise, appropriate for a
    polygenic quantitative trait)."""
    rng = np.random.default_rng(seed)
    demes = np.unique(deme_id)
    Xs, Ys, Es = [], [], []
    n_loci = X_geno.shape[1]
    for d in demes:
        mask = deme_id==d
        members = X_geno[mask]
        m = len(members)
        if m < 2:
            continue
        coord = Y[mask][0]
        p1i = rng.integers(0, m, size=n_synth_per_deme)
        p2i = rng.integers(0, m, size=n_synth_per_deme)
        p1 = members[p1i]; p2 = members[p2i]
        pick1 = rng.integers(0, 2, size=(n_synth_per_deme, n_loci))
        pick2 = rng.integers(0, 2, size=(n_synth_per_deme, n_loci))
        loc_idx = np.arange(n_loci)[None,:].repeat(n_synth_per_deme,0)
        a1 = p1[np.arange(n_synth_per_deme)[:,None].repeat(n_loci,1), loc_idx, pick1]
        a2 = p2[np.arange(n_synth_per_deme)[:,None].repeat(n_loci,1), loc_idx, pick2]
        child = np.stack([a1,a2], axis=-1)
        Xs.append(child)
        Ys.append(np.tile(coord, (n_synth_per_deme,1)))
        if extra_feats is not None:
            ef = extra_feats[mask]
            mid = 0.5*(ef[p1i] + ef[p2i])
            noise = rng.normal(0, np.nanstd(ef)*0.15+1e-6, size=mid.shape)
            Es.append(mid + noise)
    if len(Xs)==0:
        return X_geno[:0], Y[:0], (extra_feats[:0] if extra_feats is not None else None)
    Xout, Yout = np.concatenate(Xs,0), np.concatenate(Ys,0)
    Eout = np.concatenate(Es,0) if extra_feats is not None else None
    return Xout, Yout, Eout


class MultiModalGatedMLP(nn.Module):
    """Extends LocusGatedMLP with an optional auxiliary-feature branch (e.g. phenotype,
    or environmental covariates), fused with the locus-attention-gated genotype
    representation before the regression backbone. Setting genotype or phenotype input
    to a zero-width tensor effectively ablates that modality, allowing genotype-only,
    phenotype-only, and joint (multi-modal) variants to share one implementation."""
    def __init__(self, ranges, n_extra=0, hidden=96, depth=4, out_dim=2, dropout=0.25,
                 use_geno=True):
        super().__init__()
        self.ranges = ranges
        self.use_geno = use_geno and len(ranges) > 0
        self.n_extra = n_extra
        gate_hidden = 8
        if self.use_geno:
            self.locus_gate = nn.ModuleList([nn.Sequential(
                nn.Linear(2, gate_hidden), nn.Tanh(), nn.Linear(gate_hidden, 1)) for _ in ranges])
            geno_dim = ranges[-1][1]
        else:
            geno_dim = 0
        if n_extra > 0:
            self.extra_enc = nn.Sequential(nn.Linear(n_extra, 16), nn.ELU())
            extra_dim = 16
        else:
            extra_dim = 0
        in_dim = geno_dim + extra_dim
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ELU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)
    def forward(self, x_geno, x_extra=None):
        parts = []
        gate_w = None
        if self.use_geno:
            gates = []
            for i,(s,e) in enumerate(self.ranges):
                block = x_geno[:, s:e]
                summary = torch.stack([block.mean(1), block.max(1).values], dim=1)
                gates.append(self.locus_gate[i](summary))
            gates_cat = torch.cat(gates, dim=1)
            gate_w = torch.softmax(gates_cat, dim=1) * len(self.ranges)
            gated_blocks = [x_geno[:, s:e]*gate_w[:, i:i+1] for i,(s,e) in enumerate(self.ranges)]
            parts.append(torch.cat(gated_blocks, dim=1))
        if self.n_extra > 0:
            parts.append(self.extra_enc(x_extra))
        xg = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        h = self.backbone(xg)
        return self.head(h), gate_w


def multimodal_geo_predict(Xtr_geno, Ytr, Xte_geno, Dtr=None, Ptr=None, Pte=None,
                            mode='joint', epochs=400, augment=True, n_synth_per_deme=15,
                            mc_samples=15, hidden=96, depth=4):
    """Geographic-origin prediction with selectable input modality:
      mode='geno'  -> genotype only (equivalent to MicroGeoGate_predict)
      mode='pheno' -> phenotype (or other auxiliary feature matrix Ptr/Pte) only
      mode='joint' -> genotype + phenotype fused (multi-modal)
    Ptr/Pte should be 2D arrays (n_samples, n_features); NaNs are mean-imputed per feature.
    """
    use_geno = mode in ('geno','joint') and Xtr_geno is not None
    use_extra = mode in ('pheno','joint') and Ptr is not None

    if use_geno:
        _, ranges, enc = encode_genotypes_grouped(Xtr_geno)
    else:
        ranges, enc = [], None

    Xaug, Yaug, Paug = Xtr_geno, Ytr.copy(), (Ptr.copy() if use_extra else None)
    if augment and Dtr is not None and use_geno:
        Xs, Ys, Es = mendelian_augment(Xtr_geno, Ytr, Dtr, n_synth_per_deme=n_synth_per_deme,
                                        extra_feats=(Ptr if use_extra else None))
        if len(Xs):
            Xaug = np.concatenate([Xtr_geno, Xs], axis=0)
            Yaug = np.concatenate([Ytr, Ys], axis=0)
            if use_extra:
                Paug = np.concatenate([Ptr, Es], axis=0)
    elif use_extra and augment and Dtr is not None and not use_geno:
        # phenotype-only augmentation: mid-parent resampling without genotype
        _, Ys_dummy, Es = mendelian_augment(np.zeros((len(Ytr),1,2),dtype=np.int32), Ytr, Dtr,
                                             n_synth_per_deme=n_synth_per_deme, extra_feats=Ptr)
        if Es is not None and len(Es):
            Yaug = np.concatenate([Ytr, Ys_dummy], axis=0)
            Paug = np.concatenate([Ptr, Es], axis=0)

    if use_geno:
        Faug = enc.transform(Xaug); Fte = enc.transform(Xte_geno)
    else:
        Faug = np.zeros((len(Yaug),0), dtype=np.float32); Fte = np.zeros((len(Xte_geno if Xte_geno is not None else Pte),0), dtype=np.float32)

    n_extra = 0
    if use_extra:
        Paug = np.atleast_2d(Paug.T).T if Paug.ndim==1 else Paug
        Pte2 = np.atleast_2d(Pte.T).T if Pte.ndim==1 else Pte
        col_mean = np.nanmean(Paug, axis=0)
        Paug = np.where(np.isnan(Paug), col_mean, Paug)
        Pte2 = np.where(np.isnan(Pte2), col_mean, Pte2)
        p_mu, p_sd = Paug.mean(0), Paug.std(0)+1e-8
        Paug_n = (Paug-p_mu)/p_sd
        Pte_n = (Pte2-p_mu)/p_sd
        n_extra = Paug.shape[1]
    else:
        Paug_n = np.zeros((len(Yaug),0), dtype=np.float32)
        Pte_n = np.zeros((Fte.shape[0],0), dtype=np.float32)

    mu, sd = Yaug.mean(0), Yaug.std(0)+1e-8
    Yn = (Yaug-mu)/sd
    mu_t = torch.tensor(mu, dtype=torch.float32); sd_t = torch.tensor(sd, dtype=torch.float32)

    from sklearn.neighbors import NearestNeighbors
    k = min(6, len(Yaug)-1)
    nnb = NearestNeighbors(n_neighbors=k).fit(Yaug)
    dist,_ = nnb.kneighbors(Yaug)
    dens = 1.0/(dist.mean(1)+1e-3)
    w = (1.0/dens); w = w/w.mean()
    w_t = torch.tensor(w, dtype=torch.float32)

    model = MultiModalGatedMLP(ranges, n_extra=n_extra, hidden=hidden, depth=depth, use_geno=use_geno)
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    Xb = torch.tensor(Faug, dtype=torch.float32)
    Pb = torch.tensor(Paug_n, dtype=torch.float32)
    Yb_n = torch.tensor(Yn, dtype=torch.float32)
    Yb_true = torch.tensor(Yaug, dtype=torch.float32)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out_n, gate = model(Xb, Pb)
        out = out_n*sd_t+mu_t
        loss_mse = (((out_n-Yb_n)**2).mean(1)*w_t).mean()
        loss_hav = haversine_loss_torch(Yb_true, out)
        loss = 0.4*loss_mse + 0.6*(loss_hav/1000.0)
        loss.backward(); opt.step()

    model.train()
    preds, gates_all = [], []
    Xte_t = torch.tensor(Fte, dtype=torch.float32)
    Pte_t = torch.tensor(Pte_n, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(mc_samples):
            out_n, gate = model(Xte_t, Pte_t)
            preds.append(out_n.numpy()*sd+mu)
            if gate is not None:
                gates_all.append(gate.numpy().mean(0))
    preds = np.stack(preds,0)
    gate_mean = np.mean(gates_all,axis=0) if gates_all else None
    return np.median(preds,0), preds.std(0), gate_mean


# ---------------------------------------------------------------
# Alternative input encoding: population allele-frequency assignment
# profile (classical frequency-based assignment, e.g. Paetkau/GeneClass
# style likelihood), tested as an alternative to raw per-locus dosage.
# ---------------------------------------------------------------
def encode_allele_freq_profile(Xtr_geno, Dtr, X_query, eps=1e-3):
    """For each training locality, compute per-locus allele frequencies.
    Encode each queried individual (train or test) as its log-likelihood
    profile across all training localities: for locality k, sum over loci
    of log(freq of each of the individual's 2 alleles in locality k).
    Output: (n_query, n_localities) feature matrix -- a population
    frequency-based 'assignment profile' instead of raw genotype dosage."""
    demes = np.unique(Dtr)
    n_loci = Xtr_geno.shape[1]
    # per-locality, per-locus allele frequency dict
    freq_tables = []  # list over loci of dict: {allele: array(len(demes)) freq}
    all_alleles_per_locus = [np.unique(Xtr_geno[:, l, :]) for l in range(n_loci)]
    for l in range(n_loci):
        alleles = all_alleles_per_locus[l]
        amap = {a: i for i, a in enumerate(alleles)}
        table = np.full((len(demes), len(alleles)), eps)
        for di, d in enumerate(demes):
            sub = Xtr_geno[Dtr == d, l, :]
            n_tot = sub.size
            for a in sub.flatten():
                table[di, amap[a]] += 1.0
            table[di] /= (n_tot + eps*len(alleles))
        freq_tables.append((alleles, amap, table))

    n_q = X_query.shape[0]
    profile = np.zeros((n_q, len(demes)), dtype=np.float32)
    for l in range(n_loci):
        alleles, amap, table = freq_tables[l]  # table: (n_demes, n_alleles)
        logtable = np.log(table)  # (n_demes, n_alleles)
        col = X_query[:, l, :]
        for c in range(2):
            a = col[:, c]
            idx = np.array([amap.get(v, None) for v in a])
            for qi in range(n_q):
                if idx[qi] is None:
                    continue
                profile[qi] += logtable[:, idx[qi]]
    return profile, demes


def freqprofile_geo_predict(Xtr_geno, Ytr, Xte_geno, Dtr, epochs=350, hidden=64, depth=3,
                             mc_samples=15):
    """Genotype-to-geography prediction using the population allele-frequency
    assignment-profile encoding (instead of raw per-individual dosage)."""
    profile_tr, demes = encode_allele_freq_profile(Xtr_geno, Dtr, Xtr_geno)
    profile_te, _ = encode_allele_freq_profile(Xtr_geno, Dtr, Xte_geno)
    # standardize
    mu_f, sd_f = profile_tr.mean(0), profile_tr.std(0)+1e-6
    Ftr = (profile_tr-mu_f)/sd_f
    Fte = (profile_te-mu_f)/sd_f

    mu, sd = Ytr.mean(0), Ytr.std(0)+1e-8
    Yn = (Ytr-mu)/sd
    model = SimpleMLP(Ftr.shape[1], hidden=hidden, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    Xb = torch.tensor(Ftr, dtype=torch.float32)
    Yb_n = torch.tensor(Yn, dtype=torch.float32)
    Yb_true = torch.tensor(Ytr, dtype=torch.float32)
    mu_t = torch.tensor(mu, dtype=torch.float32); sd_t = torch.tensor(sd, dtype=torch.float32)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        pred_n = model(Xb)
        pred = pred_n*sd_t+mu_t
        loss = 0.4*((pred_n-Yb_n)**2).mean() + 0.6*(haversine_loss_torch(Yb_true, pred)/1000.0)
        loss.backward(); opt.step()
    model.train()
    preds = []
    Xte_t = torch.tensor(Fte, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(mc_samples):
            pred_n = model(Xte_t)
            preds.append(pred_n.numpy()*sd+mu)
    preds = np.stack(preds,0)
    return np.median(preds,0), preds.std(0)

def MicroGeoGate_extract_features(X_geno, Y, D, epochs=400, augment=True,
                                     n_synth_per_deme=15, hidden=96, depth=4):
    """Train MicroGeoGate on the full dataset and return the learned penultimate-layer
    ('extracted genetic feature') representation for every real (non-synthetic) individual,
    for downstream visualization (e.g. PCA colored by source locality)."""
    _, ranges, enc = encode_genotypes_grouped(X_geno)

    Xaug, Yaug = X_geno.copy(), Y.copy()
    if augment and D is not None:
        Xs, Ys, _ = mendelian_augment(X_geno, Y, D, n_synth_per_deme=n_synth_per_deme)
        if len(Xs):
            Xaug = np.concatenate([X_geno, Xs], axis=0)
            Yaug = np.concatenate([Y, Ys], axis=0)

    Faug = enc.transform(Xaug)
    Freal = enc.transform(X_geno)  # features for the real individuals only, for extraction

    mu, sd = Yaug.mean(0), Yaug.std(0)+1e-8
    Yn = (Yaug-mu)/sd
    mu_t = torch.tensor(mu, dtype=torch.float32); sd_t = torch.tensor(sd, dtype=torch.float32)

    from sklearn.neighbors import NearestNeighbors
    k = min(6, len(Yaug)-1)
    nnb = NearestNeighbors(n_neighbors=k).fit(Yaug)
    dist,_ = nnb.kneighbors(Yaug)
    dens = 1.0/(dist.mean(1)+1e-3)
    w = (1.0/dens); w = w/w.mean()
    w_t = torch.tensor(w, dtype=torch.float32)

    model = LocusGatedMLP(ranges, hidden=hidden, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    Xb = torch.tensor(Faug, dtype=torch.float32)
    Yb_n = torch.tensor(Yn, dtype=torch.float32)
    Yb_true = torch.tensor(Yaug, dtype=torch.float32)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out_n, gate = model(Xb)
        out = out_n*sd_t+mu_t
        loss_mse = (((out_n-Yb_n)**2).mean(1)*w_t).mean()
        loss_hav = haversine_loss_torch(Yb_true, out)
        loss = 0.4*loss_mse + 0.6*(loss_hav/1000.0)
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        _, _, hidden_feats = model(torch.tensor(Freal, dtype=torch.float32), return_hidden=True)
    return hidden_feats.numpy()


# ======================================================================
# Transformer variants (for architecture-comparison benchmark only).
# These were implemented to test whether cross-locus self-attention helps
# on sparse marker panels. They share the SAME dosage encoding, the SAME
# Mendelian augmentation, and the SAME combined loss as MicroGeoGate, so
# the only difference is the core aggregation mechanism.
# ======================================================================

class LocusTransformerV1(nn.Module):
    """Variant 1: full cross-locus self-attention. Each locus's dosage block
    is projected to a fixed-width token; a standard TransformerEncoder applies
    multi-head self-attention across the L locus-tokens; the pooled token
    representation feeds the same MLP head. Tests whether modelling explicit
    locus-locus interactions helps geographic prediction."""
    def __init__(self, ranges, d_model=32, nhead=4, nlayers=2, hidden=96, dropout=0.25):
        super().__init__()
        self.ranges = ranges
        self.proj = nn.ModuleList([nn.Linear(e - s, d_model) for (s, e) in ranges])
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=hidden, dropout=dropout,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(nn.Linear(d_model, hidden), nn.ELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        tokens = [self.proj[i](x[:, s:e]) for i, (s, e) in enumerate(self.ranges)]
        seq = torch.stack(tokens, dim=1)              # (batch, L, d_model)
        enc = self.encoder(seq)                       # (batch, L, d_model)
        pooled = enc.mean(dim=1)                      # mean-pool over loci
        return self.head(pooled)


class LocusTransformerV2(nn.Module):
    """Variant 2: lightweight attention pooling, NO cross-locus self-attention.
    Each locus token gets a scalar attention score; a softmax over loci gives
    attention-weighted pooling of the tokens before the MLP head. A cheaper
    'attention' design than V1 that still avoids the per-locus independent
    gate used by the final MicroGeoGate model."""
    def __init__(self, ranges, d_model=32, hidden=96, dropout=0.25):
        super().__init__()
        self.ranges = ranges
        self.proj = nn.ModuleList([nn.Linear(e - s, d_model) for (s, e) in ranges])
        self.attn = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.Linear(d_model, hidden), nn.ELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        tokens = [self.proj[i](x[:, s:e]) for i, (s, e) in enumerate(self.ranges)]
        seq = torch.stack(tokens, dim=1)              # (batch, L, d_model)
        score = self.attn(torch.tanh(seq))            # (batch, L, 1)
        w = torch.softmax(score, dim=1)               # attention over loci
        pooled = (seq * w).sum(dim=1)                 # (batch, d_model)
        return self.head(pooled)


def _transformer_predict(model_cls, Xtr_geno, Ytr, Xte_geno, Dtr=None, epochs=220,
                         augment=True, n_synth_per_deme=15, mc_samples=15, **model_kw):
    """Shared train/predict wrapper for the Transformer variants, matched to
    MicroGeoGate_predict: same encoding, same Mendelian augmentation, same
    density-weighted combined loss, same MC-dropout inference."""
    _, ranges, enc = encode_genotypes_grouped(Xtr_geno)
    Xaug, Yaug = Xtr_geno.copy(), Ytr.copy()
    if augment and Dtr is not None:
        Xs, Ys, _ = mendelian_augment(Xtr_geno, Ytr, Dtr, n_synth_per_deme=n_synth_per_deme)
        if len(Xs):
            Xaug = np.concatenate([Xtr_geno, Xs], axis=0)
            Yaug = np.concatenate([Ytr, Ys], axis=0)
    Faug = enc.transform(Xaug); Fte = enc.transform(Xte_geno)
    mu, sd = Yaug.mean(0), Yaug.std(0) + 1e-8
    Yn = (Yaug - mu) / sd
    mu_t = torch.tensor(mu, dtype=torch.float32); sd_t = torch.tensor(sd, dtype=torch.float32)
    from sklearn.neighbors import NearestNeighbors
    k = min(6, len(Yaug) - 1)
    dist, _ = NearestNeighbors(n_neighbors=k).fit(Yaug).kneighbors(Yaug)
    dens = 1.0 / (dist.mean(1) + 1e-3); w = (1.0 / dens); w = w / w.mean()
    w_t = torch.tensor(w, dtype=torch.float32)
    model = model_cls(ranges, **model_kw)
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    Xb = torch.tensor(Faug, dtype=torch.float32)
    Yb_n = torch.tensor(Yn, dtype=torch.float32)
    Yb_true = torch.tensor(Yaug, dtype=torch.float32)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out_n = model(Xb)
        out = out_n * sd_t + mu_t
        loss_mse = (((out_n - Yb_n) ** 2).mean(1) * w_t).mean()
        loss_hav = haversine_loss_torch(Yb_true, out)
        loss = 0.4 * loss_mse + 0.6 * (loss_hav / 1000.0)
        loss.backward(); opt.step()
    model.train()
    preds = []
    Xte_t = torch.tensor(Fte, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(mc_samples):
            preds.append(model(Xte_t).numpy() * sd + mu)
    preds = np.stack(preds, 0)
    return np.median(preds, 0)


def baseline_transformer_v1(Xtr_geno, Ytr, Xte_geno, Dtr=None, epochs=220):
    return _transformer_predict(LocusTransformerV1, Xtr_geno, Ytr, Xte_geno, Dtr=Dtr, epochs=epochs)


def baseline_transformer_v2(Xtr_geno, Ytr, Xte_geno, Dtr=None, epochs=220):
    return _transformer_predict(LocusTransformerV2, Xtr_geno, Ytr, Xte_geno, Dtr=Dtr, epochs=epochs)

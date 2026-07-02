"""
microgeoformer.model
=====================
Core neural network and encoding logic for MicroGeoFormer: a locus-attention
deep learning model for predicting geographic origin (latitude, longitude)
from multilocus genotype data (microsatellite or SNP).

This module implements a scikit-learn-style estimator (`MicroGeoFormer`)
with `.fit()`, `.predict()`, `.save()`, and `.load()` methods, so the model
can be used as a drop-in tool without needing to understand the internal
PyTorch implementation.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn

EARTH_RADIUS_KM = 6371.0088


# ---------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------

def haversine_km(y_true, y_pred):
    """Great-circle distance (km) between arrays of (lat, lon) pairs."""
    lat1, lon1 = np.radians(y_true[:, 0]), np.radians(y_true[:, 1])
    lat2, lon2 = np.radians(y_pred[:, 0]), np.radians(y_pred[:, 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _haversine_loss_torch(y_true, y_pred):
    lat1 = torch.deg2rad(y_true[:, 0]); lon1 = torch.deg2rad(y_true[:, 1])
    lat2 = torch.deg2rad(y_pred[:, 0]); lon2 = torch.deg2rad(y_pred[:, 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    a = torch.clamp(a, 1e-12, 1 - 1e-12)
    return (2 * EARTH_RADIUS_KM * torch.arcsin(torch.sqrt(a))).mean()


# ---------------------------------------------------------------------
# Genotype encoding
# ---------------------------------------------------------------------

class DosageEncoder:
    """Encodes diploid genotypes (n_individuals, n_loci, 2) as per-locus
    allele-dosage vectors (count of each observed allele: 0, 1, or 2)."""

    def __init__(self):
        self.locus_alleles = None

    def fit(self, X):
        self.locus_alleles = [np.unique(X[:, l, :]) for l in range(X.shape[1])]
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
                pos = np.clip(np.searchsorted(alleles, col), 0, na - 1)
                valid = alleles[pos] == col
                rows = np.nonzero(valid)[0]
                np.add.at(dos, (rows, pos[valid]), 1.0)
            feats.append(dos)
        return np.concatenate(feats, axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def ranges(self):
        """Returns (start, end) column ranges per locus in the encoded matrix."""
        ranges, pos = [], 0
        for alleles in self.locus_alleles:
            ranges.append((pos, pos + len(alleles)))
            pos += len(alleles)
        return ranges


def mendelian_augment(X_geno, Y, locality_id, n_synth_per_locality=20, seed=0, extra_feats=None):
    """Mendelian-resampling data augmentation: synthesizes additional training
    genotypes by resampling gamete pairs from individuals collected at the
    same reference locality. Recommended when per-locality sample sizes are
    small (fewer than ~15 individuals), the regime MicroGeoFormer targets."""
    rng = np.random.default_rng(seed)
    localities = np.unique(locality_id)
    Xs, Ys, Es = [], [], []
    n_loci = X_geno.shape[1]
    for loc in localities:
        mask = locality_id == loc
        members = X_geno[mask]
        m = len(members)
        if m < 2:
            continue
        coord = Y[mask][0]
        p1i = rng.integers(0, m, size=n_synth_per_locality)
        p2i = rng.integers(0, m, size=n_synth_per_locality)
        p1, p2 = members[p1i], members[p2i]
        pick1 = rng.integers(0, 2, size=(n_synth_per_locality, n_loci))
        pick2 = rng.integers(0, 2, size=(n_synth_per_locality, n_loci))
        loc_idx = np.arange(n_loci)[None, :].repeat(n_synth_per_locality, 0)
        row_idx = np.arange(n_synth_per_locality)[:, None].repeat(n_loci, 1)
        a1 = p1[row_idx, loc_idx, pick1]
        a2 = p2[row_idx, loc_idx, pick2]
        Xs.append(np.stack([a1, a2], axis=-1))
        Ys.append(np.tile(coord, (n_synth_per_locality, 1)))
        if extra_feats is not None:
            ef = extra_feats[mask]
            mid = 0.5 * (ef[p1i] + ef[p2i])
            noise = rng.normal(0, np.nanstd(ef) * 0.15 + 1e-6, size=mid.shape)
            Es.append(mid + noise)
    if not Xs:
        return X_geno[:0], Y[:0], (extra_feats[:0] if extra_feats is not None else None)
    Xout, Yout = np.concatenate(Xs, 0), np.concatenate(Ys, 0)
    Eout = np.concatenate(Es, 0) if extra_feats is not None else None
    return Xout, Yout, Eout


# ---------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------

class LocusGatedMLP(nn.Module):
    """Locus-attention-gated regression network. A per-locus attention gate
    scores each locus's geographic informativeness and rescales its dosage
    block before a residual MLP backbone predicts (latitude, longitude)."""

    def __init__(self, ranges, hidden=96, depth=4, out_dim=2, dropout=0.25):
        super().__init__()
        self.ranges = ranges
        gate_hidden = 8
        self.locus_gate = nn.ModuleList([
            nn.Sequential(nn.Linear(2, gate_hidden), nn.Tanh(), nn.Linear(gate_hidden, 1))
            for _ in ranges
        ])
        in_dim = ranges[-1][1]
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ELU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x, return_hidden=False):
        gates = []
        for i, (s, e) in enumerate(self.ranges):
            block = x[:, s:e]
            summary = torch.stack([block.mean(1), block.max(1).values], dim=1)
            gates.append(self.locus_gate[i](summary))
        gate_w = torch.softmax(torch.cat(gates, dim=1), dim=1) * len(self.ranges)
        gated = torch.cat([x[:, s:e] * gate_w[:, i:i + 1] for i, (s, e) in enumerate(self.ranges)], dim=1)
        h = self.backbone(gated)
        out = self.head(h)
        return (out, gate_w, h) if return_hidden else (out, gate_w)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

class MicroGeoFormer:
    """Locus-attention deep learning model for predicting geographic origin
    (latitude, longitude) from multilocus genotype data.

    Parameters
    ----------
    hidden : int, default 96
        Width of each fully connected layer in the regression backbone.
    depth : int, default 4
        Number of fully connected layers in the regression backbone.
    dropout : float, default 0.25
        Dropout probability, also used for Monte Carlo uncertainty at
        inference time.
    epochs : int, default 400
        Number of training epochs.
    lr : float, default 1.5e-3
        Adam learning rate.
    weight_decay : float, default 5e-4
        Adam weight decay.
    loss_weights : tuple, default (0.4, 0.6)
        Weights for (normalized MSE, Haversine distance / 1000) in the
        combined training loss.
    augment : bool, default True
        Whether to apply Mendelian-resampling augmentation using locality
        labels supplied to `fit()`. Recommended whenever per-locality
        sample sizes are small (the regime this model is designed for).
    n_synth_per_locality : int, default 20
        Number of synthetic individuals generated per reference locality
        when `augment=True`.
    mc_samples : int, default 15
        Number of Monte Carlo dropout forward passes averaged at inference
        to produce the point prediction and uncertainty estimate.

    Example
    -------
    >>> from microgeoformer import MicroGeoFormer
    >>> model = MicroGeoFormer(epochs=400)
    >>> model.fit(X_train, Y_train, locality_id=D_train)
    >>> Y_pred, Y_std = model.predict(X_test, return_std=True)
    """

    def __init__(self, hidden=96, depth=4, dropout=0.25, epochs=400, lr=1.5e-3,
                 weight_decay=5e-4, loss_weights=(0.4, 0.6), augment=True,
                 n_synth_per_locality=20, mc_samples=15, density_weighting=True,
                 random_state=0):
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        self.augment = augment
        self.n_synth_per_locality = n_synth_per_locality
        self.mc_samples = mc_samples
        self.density_weighting = density_weighting
        self.random_state = random_state
        self._encoder = None
        self._model = None
        self._mu = None
        self._sd = None

    def fit(self, X, Y, locality_id=None, verbose=False):
        """Train the model.

        Parameters
        ----------
        X : ndarray, shape (n_individuals, n_loci, 2)
            Diploid genotypes; missing data may be encoded with any
            consistent sentinel value (e.g. -999), which will be treated
            as its own allele state.
        Y : ndarray, shape (n_individuals, 2)
            True (latitude, longitude) for each individual.
        locality_id : ndarray, shape (n_individuals,), optional
            Integer or string label giving each individual's reference
            locality. Required if `augment=True` (the default).
        """
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        self._encoder = DosageEncoder().fit(X)
        ranges = self._encoder.ranges()

        X_aug, Y_aug = X, Y
        if self.augment:
            if locality_id is None:
                raise ValueError("locality_id is required when augment=True")
            Xs, Ys, _ = mendelian_augment(X, Y, locality_id,
                                           n_synth_per_locality=self.n_synth_per_locality,
                                           seed=self.random_state)
            if len(Xs):
                X_aug = np.concatenate([X, Xs], axis=0)
                Y_aug = np.concatenate([Y, Ys], axis=0)

        F_aug = self._encoder.transform(X_aug)
        self._mu, self._sd = Y_aug.mean(0), Y_aug.std(0) + 1e-8
        Y_norm = (Y_aug - self._mu) / self._sd

        if self.density_weighting:
            from sklearn.neighbors import NearestNeighbors
            k = min(6, len(Y_aug) - 1)
            dist, _ = NearestNeighbors(n_neighbors=k).fit(Y_aug).kneighbors(Y_aug)
            density = 1.0 / (dist.mean(1) + 1e-3)
            w = (1.0 / density); w = w / w.mean()
        else:
            w = np.ones(len(Y_aug))

        self._model = LocusGatedMLP(ranges, hidden=self.hidden, depth=self.depth, dropout=self.dropout)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        Xb = torch.tensor(F_aug, dtype=torch.float32)
        Yb_n = torch.tensor(Y_norm, dtype=torch.float32)
        Yb_true = torch.tensor(Y_aug, dtype=torch.float32)
        wb = torch.tensor(w, dtype=torch.float32)
        mu_t = torch.tensor(self._mu, dtype=torch.float32)
        sd_t = torch.tensor(self._sd, dtype=torch.float32)
        w_mse, w_hav = self.loss_weights

        for ep in range(self.epochs):
            self._model.train()
            opt.zero_grad()
            out_n, _ = self._model(Xb)
            out = out_n * sd_t + mu_t
            loss_mse = (((out_n - Yb_n) ** 2).mean(1) * wb).mean()
            loss_hav = _haversine_loss_torch(Yb_true, out)
            loss = w_mse * loss_mse + w_hav * (loss_hav / 1000.0)
            loss.backward()
            opt.step()
            if verbose and (ep + 1) % max(1, self.epochs // 10) == 0:
                print(f"epoch {ep+1}/{self.epochs}  loss={loss.item():.4f}")
        return self

    def predict(self, X, return_std=False, return_locus_importance=False):
        """Predict (latitude, longitude) for new individuals.

        Parameters
        ----------
        X : ndarray, shape (n_individuals, n_loci, 2)
            Genotypes to predict origin for (same locus set used in `fit`).
        return_std : bool, default False
            If True, also return the Monte Carlo dropout standard deviation
            per individual and coordinate, as a calibrated uncertainty
            estimate.
        return_locus_importance : bool, default False
            If True, also return the mean learned locus-attention weight
            per locus, averaged across the input individuals and MC passes.

        Returns
        -------
        Y_pred : ndarray, shape (n_individuals, 2)
        Y_std : ndarray, shape (n_individuals, 2), only if return_std=True
        locus_importance : ndarray, shape (n_loci,), only if
            return_locus_importance=True
        """
        if self._model is None:
            raise RuntimeError("Model has not been fit yet. Call .fit() first.")
        F = self._encoder.transform(X)
        Xt = torch.tensor(F, dtype=torch.float32)
        self._model.train()  # keep dropout active for MC sampling
        preds, gates = [], []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                out_n, gate_w = self._model(Xt)
                preds.append(out_n.numpy() * self._sd + self._mu)
                gates.append(gate_w.numpy().mean(0))
        preds = np.stack(preds, 0)
        Y_pred = np.median(preds, 0)
        out = [Y_pred]
        if return_std:
            out.append(preds.std(0))
        if return_locus_importance:
            out.append(np.mean(gates, axis=0))
        return out[0] if len(out) == 1 else tuple(out)

    def score(self, X, Y):
        """Median Haversine prediction error (km) on a held-out set."""
        Y_pred = self.predict(X)
        return float(np.median(haversine_km(Y, Y_pred)))

    def save(self, path):
        """Save the fitted model (weights + encoder + normalization) to disk."""
        state = {
            "config": dict(hidden=self.hidden, depth=self.depth, dropout=self.dropout,
                            epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay,
                            loss_weights=self.loss_weights, augment=self.augment,
                            n_synth_per_locality=self.n_synth_per_locality,
                            mc_samples=self.mc_samples, density_weighting=self.density_weighting,
                            random_state=self.random_state),
            "encoder_locus_alleles": self._encoder.locus_alleles,
            "mu": self._mu, "sd": self._sd,
            "model_state_dict": self._model.state_dict(),
            "ranges": self._encoder.ranges(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path):
        """Load a previously saved model."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(**state["config"])
        model._encoder = DosageEncoder()
        model._encoder.locus_alleles = state["encoder_locus_alleles"]
        model._mu, model._sd = state["mu"], state["sd"]
        model._model = LocusGatedMLP(state["ranges"], hidden=model.hidden,
                                      depth=model.depth, dropout=model.dropout)
        model._model.load_state_dict(state["model_state_dict"])
        return model

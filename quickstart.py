"""
quickstart.py
=============
Minimal, runnable example of the MicroGeoGate workflow, using synthetic
data generated on the fly so this script works out of the box with no
external data files. Replace the synthetic-data block with your own
genotype/coordinate arrays to use MicroGeoGate on real data.

Run with:
    python examples/quickstart.py
"""
import numpy as np
from sklearn.model_selection import train_test_split

from MicroGeoGate import MicroGeoGate, haversine_km


def make_synthetic_data(n_localities=12, n_per_locality=20, n_loci=8, seed=0):
    """Generates a toy dataset with the same shape conventions MicroGeoGate
    expects: genotypes (n_individuals, n_loci, 2), coordinates
    (n_individuals, 2), and a locality label per individual. Individuals
    from the same locality share noisy versions of a locality-specific
    allele-frequency profile, giving weak but real spatial genetic
    structure to recover -- illustrative only, not a substitute for the
    paper's population-genetic simulator.
    """
    rng = np.random.default_rng(seed)
    locality_coords = rng.uniform([35.0, -5.0], [45.0, 5.0], size=(n_localities, 2))
    locality_allele_freq = rng.uniform(0.2, 0.8, size=(n_localities, n_loci))

    X, Y, locality_id = [], [], []
    for loc in range(n_localities):
        for _ in range(n_per_locality):
            alleles = (rng.random((n_loci, 2)) < locality_allele_freq[loc][:, None]).astype(int) + 100
            X.append(alleles)
            Y.append(locality_coords[loc])
            locality_id.append(loc)
    return np.array(X), np.array(Y), np.array(locality_id)


def main():
    X, Y, locality_id = make_synthetic_data()

    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=0.25, stratify=locality_id, random_state=0
    )

    model = MicroGeoGate(epochs=300, mc_samples=10, random_state=0)
    model.fit(X[idx_train], Y[idx_train], locality_id=locality_id[idx_train], verbose=True)

    Y_pred, Y_std, locus_importance = model.predict(
        X[idx_test], return_std=True, return_locus_importance=True
    )

    errors_km = haversine_km(Y[idx_test], Y_pred)
    print("\nHeld-out prediction summary")
    print("----------------------------")
    print(f"n test individuals:      {len(idx_test)}")
    print(f"median error (km):       {np.median(errors_km):.1f}")
    print(f"mean uncertainty (deg):  {Y_std.mean(axis=0)}")
    print(f"locus importance:        {np.round(locus_importance, 3)}")

    model.save("MicroGeoGate_quickstart.pkl")
    print("\nSaved trained model to MicroGeoGate_quickstart.pkl")


if __name__ == "__main__":
    main()

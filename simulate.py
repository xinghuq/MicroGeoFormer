import numpy as np

def simulate_landscape(n_demes_x=5, n_demes_y=4, Ne=80, n_loci=15, n_gen=400,
                        mut_rate=5e-4, m_rate=0.03, allele_model='SMM',
                        lat_range=(38.5,40.5), lon_range=(-4.5,-2.0), seed=0,
                        max_allele_states=None):
    """
    Forward-time Wright-Fisher stepping-stone simulation on a 2D lattice of demes.
    allele_model: 'SMM' (stepwise mutation, microsatellite-like, integer allele size)
                  'SNP' (biallelic infinite-sites-like, 0/1)
    Returns: genotypes (n_demes, Ne, n_loci, 2), deme_coords (n_demes,2) [lat,lon], deme_grid_idx
    """
    rng = np.random.default_rng(seed)
    n_demes = n_demes_x * n_demes_y
    # grid coordinates -> lat/lon
    xs = np.linspace(lon_range[0], lon_range[1], n_demes_x)
    ys = np.linspace(lat_range[0], lat_range[1], n_demes_y)
    grid = np.array([[x,y] for y in ys for x in xs])  # lon, lat order internally
    deme_coords = grid[:, ::-1]  # lat, lon

    # build stepping-stone migration neighbor structure (4-neighbor on grid) with rate m_rate
    idx_grid = np.arange(n_demes).reshape(n_demes_y, n_demes_x)
    neighbors = {i: [] for i in range(n_demes)}
    for r in range(n_demes_y):
        for c in range(n_demes_x):
            i = idx_grid[r, c]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r+dr, c+dc
                if 0 <= rr < n_demes_y and 0 <= cc < n_demes_x:
                    neighbors[i].append(idx_grid[rr, cc])

    # init genotypes: start all populations fixed at allele 100 (SMM) or 0 (SNP)
    start_allele = 100 if allele_model == 'SMM' else 0
    geno = np.full((n_demes, Ne, n_loci, 2), start_allele, dtype=np.int32)

    for gen in range(n_gen):
        new_geno = np.empty_like(geno)
        # migration: build parent pool per deme = local (1-m) + migrants from neighbors (m, split evenly)
        for d in range(n_demes):
            nbrs = neighbors[d]
            if len(nbrs) == 0 or m_rate == 0:
                pool = geno[d]  # (Ne, loci, 2)
            else:
                n_mig = int(round(Ne * m_rate))
                n_local = Ne - n_mig
                local_idx = rng.integers(0, Ne, size=n_local)
                pool_parts = [geno[d][local_idx]]
                per_nbr = max(1, n_mig // len(nbrs))
                for nb in nbrs:
                    mig_idx = rng.integers(0, Ne, size=per_nbr)
                    pool_parts.append(geno[nb][mig_idx])
                pool = np.concatenate(pool_parts, axis=0)
            pool_size = pool.shape[0]
            # random mating: draw Ne offspring, each gets 1 random allele from 2 random parents per locus
            p1 = rng.integers(0, pool_size, size=(Ne,1)).repeat(n_loci, axis=1)
            p2 = rng.integers(0, pool_size, size=(Ne,1)).repeat(n_loci, axis=1)
            loci_idx = np.arange(n_loci)[None,:].repeat(Ne,0)
            a1 = pool[p1, loci_idx, rng.integers(0,2,size=(Ne,n_loci))]
            a2 = pool[p2, loci_idx, rng.integers(0,2,size=(Ne,n_loci))]
            off = np.stack([a1, a2], axis=-1)  # (Ne, loci, 2)
            # mutation
            mut_mask = rng.random(off.shape) < mut_rate
            if allele_model == 'SMM':
                steps = rng.choice([-1,1], size=off.shape)
                off = np.where(mut_mask, off + steps, off)
                off = np.clip(off, 60, 160)
            else:  # SNP biallelic flip
                off = np.where(mut_mask, 1 - off, off)
            new_geno[d] = off
        geno = new_geno
    return geno, deme_coords

def simulate_phenotype(geno, deme_coords, qtl_loci=(0,1,2), h2=0.4, env_scale=1.0,
                        noise_scale=1.0, seed=0):
    """Simulate a single polygenic quantitative trait (e.g., body size) for every
    individual in `geno` (n_demes, Ne, n_loci, 2):
      trait = additive_genetic_value (sum of standardized allele sizes at `qtl_loci`)
            + environmental_deviation (deterministic smooth function of deme geographic
              position -- a simple synthetic 'climate' gradient across the simulated
              landscape, in lieu of real bioclimatic data)
            + residual noise
    Returns trait array shaped (n_demes, Ne) and the per-deme environmental covariate
    (n_demes,) used to generate it, so genotype-only / phenotype-only / joint
    genotype+phenotype geo-prediction can be compared under a known, fully-controlled
    genetic and environmental architecture.
    """
    rng = np.random.default_rng(seed)
    n_demes, Ne, n_loci, _ = geno.shape
    qtl_loci = [l for l in qtl_loci if l < n_loci]
    additive = geno[:, :, qtl_loci, :].astype(np.float64).sum(axis=(2,3))  # (n_demes, Ne)
    additive = (additive - additive.mean()) / (additive.std() + 1e-8)

    lat = deme_coords[:, 0]
    lon = deme_coords[:, 1]
    env_covariate = 0.7*(lat - lat.mean())/ (lat.std()+1e-8) + 0.3*(lon - lon.mean())/(lon.std()+1e-8)
    env_dev = np.tile(env_covariate[:, None], (1, Ne))

    noise = rng.normal(0, 1, size=(n_demes, Ne))
    h2 = np.clip(h2, 0, 1)
    trait = np.sqrt(h2)*additive + np.sqrt(1-h2)*env_scale*env_dev
    trait = trait + noise*noise_scale*0.3
    return trait, env_covariate

if __name__ == "__main__":
    geno, coords = simulate_landscape(n_gen=50)
    print(geno.shape, coords.shape)
    print(coords[:5])

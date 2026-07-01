import pandas as pd, numpy as np

COORDS = {  # lat, lon  (converted from Table 1; longitude sign corrected to West)
 'Saladar de Ocaña': (39.985445, -3.630508),
 'Saladar de Huerta': (39.838697, -3.617103),
 'Laguna de Longar': (39.700548, -3.321046),
 'Laguna de La Albardiosa': (39.658024, -3.288700),
 'Laguna Larga': (39.609088, -3.317164),
 'Laguna de Tírez': (39.546603, -3.354411),
 'Laguna de Palomares': (39.535906, -3.172344),
 'Laguna de Los Carros': (39.472016, -3.262528),
 'Laguna de Las Yeguas': (39.418396, -3.281576),
 'Laguna de Salicor': (39.470083, -3.173809),
 'Laguna de Alcahozo': (39.391585, -2.875947),
 'Saladar de El Pedernoso': (39.491164, -2.767518),
}
# Normalize accented Locality strings that appear in the xls (may differ in accents/spelling)
def match_locality(name):
    name = str(name).strip()
    key_map = {
        'Saladar de Ocaña':'Saladar de Ocaña','Saladar de Ocana':'Saladar de Ocaña','Saladar de Ocaña ':'Saladar de Ocaña',
        'Saladar de Huerta':'Saladar de Huerta',
        'Laguna de Longar':'Laguna de Longar',
        'Laguna de La Albardiosa':'Laguna de La Albardiosa','Laguna de la Albardiosa':'Laguna de La Albardiosa',
        'Laguna Larga':'Laguna Larga',
        'Laguna de Tírez':'Laguna de Tírez','Laguna de Tirez':'Laguna de Tírez',
        'Laguna de Palomares':'Laguna de Palomares',
        'Laguna de Los Carros':'Laguna de Los Carros','Laguna de los Carros':'Laguna de Los Carros',
        'Laguna de Las Yeguas':'Laguna de Las Yeguas','Laguna de las Yeguas':'Laguna de Las Yeguas',
        'Laguna de Salicor':'Laguna de Salicor',
        'Laguna de Alcahozo':'Laguna de Alcahozo',
        'Saladar de El Pedernoso':'Saladar de El Pedernoso','Saladar de el Pedernoso':'Saladar de El Pedernoso',
    }
    return key_map.get(name, name)

def load_species(sheet):
    df = pd.read_excel('/mnt/user-data/uploads/PhenotypicGenotypicData.xls', sheet_name=sheet)
    loc_cols = [c for c in df.columns if c not in ['ID','Species','Locality','Sex','Femur Length']]
    n_loci = len(loc_cols)//2
    genos = []
    for i in range(len(df)):
        row = []
        for l in range(n_loci):
            a1 = df.iloc[i][loc_cols[2*l]]
            a2 = df.iloc[i][loc_cols[2*l+1]]
            def conv(v):
                try:
                    return int(float(v))
                except Exception:
                    return -999
            row.append([conv(a1), conv(a2)])
        genos.append(row)
    X = np.array(genos, dtype=np.int32)
    locs_raw = df['Locality'].values
    locs = [match_locality(l) for l in locs_raw]
    coords = []
    valid_mask = []
    for l in locs:
        if l in COORDS:
            coords.append(COORDS[l]); valid_mask.append(True)
        else:
            coords.append((np.nan,np.nan)); valid_mask.append(False)
    Y = np.array(coords, dtype=np.float64)
    valid_mask = np.array(valid_mask)
    return X[valid_mask], Y[valid_mask], np.array(locs)[valid_mask], df['ID'].values[valid_mask], n_loci

if __name__ == "__main__":
    for sheet in ['Mioscirtus_wagneri','Ramburiella_hispanica','Calliptamus_barbarus','Calliptamus_italicus','Oedaleus_decorus']:
        X,Y,locs,ids,n_loci = load_species(sheet)
        print(sheet, X.shape, "n_loci=",n_loci, "n_localities=",len(set(locs.tolist())), "unmatched:", (Y[:,0]!=Y[:,0]).sum())

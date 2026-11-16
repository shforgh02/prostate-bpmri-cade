# X2: localisation of evaluable cancers by subgroup (top-1 and top-3 candidates, 10 mm), each against its own permutation
# chance, next to the radiologist geometric hit rate. Subgroups: suspicion score, PSA, PSA density, prostate volume, age, grade group,
# number of csPCa cores, number of targets, targeted-biopsy result (systematic-only cancers), radiologist hit/miss, field strength, vendor.
#   python analysis/extended/X2_subgroups.py > outputs/extended/X2.txt
import sys
sys.path.insert(0, 'analysis/extended')
from xcommon import *

T, P = exam_table(), perm_table()
ev = T[T.evaluable == 1].set_index('Session_ID')
v = ev.vol.quantile([1 / 3, 2 / 3]).to_numpy()
S = ev.score
G = {
    'Suspicion score': [('1-2', S.isin([1, 2])), ('3', S == 3), ('4', S == 4), ('5', S == 5), ('missing (-1 or 0)', ~S.isin([1, 2, 3, 4, 5]))],
    'PSA, ng/mL': [('< 4', ev.psa < 4), ('4-10', (ev.psa >= 4) & (ev.psa < 10)), ('>= 10', ev.psa >= 10)],
    'PSA density, ng/mL/mL': [('< 0.10', ev.psad < 0.10), ('0.10-0.149', (ev.psad >= 0.10) & (ev.psad < 0.15)),
                              ('0.15-0.199', (ev.psad >= 0.15) & (ev.psad < 0.20)), ('>= 0.20', ev.psad >= 0.20)],
    'Prostate volume (tertiles)': [(f'< {v[0]:.0f} mL', ev.vol < v[0]), (f'{v[0]:.0f}-{v[1]:.0f} mL', (ev.vol >= v[0]) & (ev.vol < v[1])), (f'>= {v[1]:.0f} mL', ev.vol >= v[1])],
    'Age, years': [('< 60', ev.age < 60), ('60-69', (ev.age >= 60) & (ev.age < 70)), ('>= 70', ev.age >= 70)],
    'Grade group (highest)': [('2', ev.grade == 2), ('3', ev.grade == 3), ('4-5', ev.grade >= 4)],
    'csPCa-positive cores': [('1', ev.n_pos_cores == 1), ('2-3', ev.n_pos_cores.between(2, 3)), ('>= 4', ev.n_pos_cores >= 4)],
    'Radiologist targets': [('1', ev.n_targets == 1), ('>= 2', ev.n_targets >= 2)],
    'Targeted biopsy': [('targeted cores positive (C = 1)', ev.C == 1), ('systematic-only cancer (C = 0)', ev.C == 0)],
    'Radiologist target (10 mm)': [('hit', ev.rad_dist <= 10), ('missed', ~(ev.rad_dist <= 10))],
    'Field strength': [('3 T', ev.field == 3), ('1.5 T', ev.field == 1.5)],
    'Vendor': [('Siemens', ev.vendor.str.upper().str.contains('SIEMENS')), ('GE or Philips', ~ev.vendor.str.upper().str.contains('SIEMENS'))],
}
rows = []
for k in (3, 1):
    print(f'\n=== top-{k} candidates, 10 mm | each cell: observed % (chance %) ; p vs chance')
    PK = {m: P[(P.model == m) & (P.k == k)].set_index('Session_ID') for m in MODELS}
    for grp, subs in G.items():
        print(f'-- {grp}')
        for lab, mask in subs:
            ids = mask[mask].index
            if len(ids) == 0:
                continue
            rh = int((ev.loc[ids, 'rad_dist'] <= 10).sum())
            cells = []
            for m in MODELS:
                d = PK[m].loc[ids]
                o = int((d.dist <= 10).sum())
                e, p = pbin_tail(o, d['perm_10.0'].to_numpy())
                rows.append(dict(k=k, group=grp, subgroup=lab, n=len(ids), radiologist_hit=rh, model=NAME[m], observed=o, chance=round(e, 1), p=p))
                cells.append(f'{NAME[m]} {100 * o / len(ids):5.1f}% ({100 * e / len(ids):4.1f}%) {fmt_p(p)}')
            print(f'   {lab:32s} n={len(ids):3d} | radiologist {100 * rh / len(ids):5.1f}% | ' + ' | '.join(cells))
pd.DataFrame(rows).to_csv(R + 'X2_subgroups.csv', index=False)
print('\nNotes: chance = between-patient permutation (same number of candidates, mapped through the prostate bounding box). '
      'Subgroup p-values are not adjusted for multiplicity. No radiologist-or-model union is tested.')

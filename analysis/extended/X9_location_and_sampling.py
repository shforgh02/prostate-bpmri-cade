# X9: localisation (top-3, 10 mm) by tumour location and by sampling intensity, against permutation chance, with the
# radiologist hit rate. Location = midpoint of the highest-grade csPCa core (ties: longest core), normalised within the prostate bounding box
# (LPS: x left, y posterior, z superior): craniocaudal thirds (apex / mid / base), anterior vs posterior half, far-lateral (> 65% of the
# half-width from the midline, as in the earlier manuscript) vs central. Sampling intensity: number of cores per examination (median split).
#   python analysis/extended/X9_location_and_sampling.py > outputs/extended/X9.txt
import sys
sys.path.insert(0, 'analysis/extended')
from xcommon import *

T, P = exam_table(), perm_table()
ms = master()
cores = cores_757()
pos = cores[cores.Is_Valid_csPCa_Trajectory == 1].copy()
pos['len'] = np.sqrt(sum((pos[CC[i]] - pos[CC[i + 3]]) ** 2 for i in range(3)))
pos = pos.sort_values(['ISUP_Grade', 'len'], ascending=False).drop_duplicates('Session_ID_Matched')
ev = T[T.evaluable == 1].set_index('Session_ID')
loc = {}
for _, r in pos.iterrows():
    s = r.Session_ID_Matched
    if s not in ev.index:
        continue
    mid = (r[CC[:3]].to_numpy(float) + r[CC[3:]].to_numpy(float)) / 2
    mn, sp = bbox(ms, s)
    u = (mid - mn) / sp
    loc[s] = dict(x=u[0], y=u[1], z=u[2])
L = pd.DataFrame(loc).T.reindex(ev.index)
med = ev.n_cores.median()
G = {'Craniocaudal (normalised z)': [('apex third', L.z < 1 / 3), ('mid third', (L.z >= 1 / 3) & (L.z < 2 / 3)), ('base third', L.z >= 2 / 3)],
     'Anteroposterior (normalised y)': [('anterior half', L.y < 0.5), ('posterior half', L.y >= 0.5)],
     'Mediolateral': [('far-lateral (> 65% from midline)', (L.x - 0.5).abs() * 2 > 0.65), ('central / paramedian', (L.x - 0.5).abs() * 2 <= 0.65)],
     'Cores per examination': [(f'<= {med:.0f}', ev.n_cores <= med), (f'> {med:.0f}', ev.n_cores > med)]}
print(f'{len(ev)} evaluable cancers; location available for {int(L.z.notna().sum())} | top-3, 10 mm: observed % (chance %) p vs chance')
rows = []
PK = {m: P[(P.model == m) & (P.k == 3)].set_index('Session_ID') for m in MODELS}
for grp, subs in G.items():
    print(f'-- {grp}')
    for lab, mask in subs:
        ids = mask[mask.fillna(False).astype(bool)].index
        rh = int((ev.loc[ids, 'rad_dist'] <= 10).sum())
        cells = []
        for m in MODELS:
            d = PK[m].loc[ids]
            o = int((d.dist <= 10).sum()); e, p = pbin_tail(o, d['perm_10.0'].to_numpy())
            rows.append(dict(group=grp, subgroup=lab, n=len(ids), radiologist_hit=rh, model=NAME[m], observed=o, chance=round(e, 1), p=p))
            cells.append(f'{NAME[m]} {100 * o / len(ids):5.1f}% ({100 * e / len(ids):4.1f}%) {fmt_p(p)}')
        print(f'   {lab:34s} n={len(ids):3d} | radiologist {100 * rh / len(ids):5.1f}% | ' + ' | '.join(cells))
pd.DataFrame(rows).to_csv(R + 'X9_location_sampling.csv', index=False)
print('\nNote: the bounding-box position approximates anatomy (no zonal segmentation is available).')

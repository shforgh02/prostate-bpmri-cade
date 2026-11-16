# X11: description of the evaluable cancers that none of the three models localized (top-3 candidates, 10 mm),
# compared with the cancers that all three localized. Uses the stored per-examination tables of X0 (no new model output).
# Location = midpoint of the highest-grade csPCa core (ties: longest core), normalized within the prostate bounding box, as in X9.
#   python analysis/extended/X11_cancers_missed_by_all_models.py > outputs/extended/X11.txt
import os, sys
sys.path.insert(0, 'analysis/extended')
from xcommon import *

XO = 'outputs/extended/'
os.makedirs(XO, exist_ok=True)
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
    if s in ev.index:
        mid = (r[CC[:3]].to_numpy(float) + r[CC[3:]].to_numpy(float)) / 2
        mn, sp = bbox(ms, s)
        u = (mid - mn) / sp
        loc[s] = dict(x=u[0], y=u[1], z=u[2])
L = pd.DataFrame(loc).T.reindex(ev.index)
D3 = P[P.k == 3].pivot(index='Session_ID', columns='model', values='dist').reindex(ev.index)
hit = D3[MODELS] <= 10
ev['n_hit'] = hit.sum(1)
assert len(ev) == 387 and int((ev.n_hit == 3).sum()) == 302 and int((ev.n_hit == 0).sum()) == 21
CH = [('grade group 2', ev.grade == 2), ('grade group 3-5', ev.grade >= 3), ('one csPCa-positive core', ev.n_pos_cores == 1),
      ('suspicion score 3', ev.score == 3), ('suspicion score 4-5', ev.score.isin([4, 5])),
      ('apex third', L.z < 1 / 3), ('posterior half', L.y >= 0.5), ('far-lateral', (L.x - 0.5).abs() * 2 > 0.65),
      ('localized by a radiologist target (10 mm)', ev.rad_dist <= 10)]
rows = []
print('Evaluable cancers by the number of models that localized them (top-3 candidates, 10 mm)')
for lab, grp in (('missed by all three models', ev.n_hit == 0), ('localized by one or two models', ev.n_hit.isin([1, 2])), ('localized by all three models', ev.n_hit == 3)):
    n = int(grp.sum())
    cells = []
    for c, m in CH:
        k = int((m & grp).sum())
        rows.append(dict(group=lab, n=n, characteristic=c, k=k, pct=round(100 * k / n, 1)))
        cells.append(f'{c} {k} ({100 * k / n:.0f}%)')
    print(f'\n{lab}: n = {n}\n  ' + '\n  '.join(cells))
pd.DataFrame(rows).to_csv(XO + 'X11_missed_by_all_models.csv', index=False)

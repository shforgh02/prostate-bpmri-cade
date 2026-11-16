# Shared loaders for the A/B/C scripts. Run everything from the repository root.
import argparse, numpy as np, pandas as pd
SD = 'results/tables/'; OUT = 'outputs/'
CC = ['Bx Base X (MRI Coord)', 'Bx Base Y (MRI Coord)', 'Bx Base Z (MRI Coord)', 'Bx Tip X (MRI Coord)', 'Bx Tip Y (MRI Coord)', 'Bx Tip Z (MRI Coord)']


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', help='csv with Session_ID column to drop')
    ap.add_argument('--candidates', default=SD + 'CANDIDATES_ANALYSIS_READY.csv')
    ap.add_argument('--tag', default='all800')
    ap.add_argument('--nperm', type=int, default=2000)
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--offdist', type=float, default=10.0)
    return ap.parse_args()


def load(a):
    """Returns exams, top-3 dynamic-RGA candidates, cores, session master (indexed), list of model names."""
    ep = pd.read_csv(SD + 'EXAMINATION_LEVEL_ENDPOINTS.csv')
    if a.exclude:
        ep = ep[~ep.Session_ID.isin(pd.read_csv(a.exclude).Session_ID)].reset_index(drop=True)
    cand = pd.read_csv(a.candidates, low_memory=False)
    cand = cand[(cand.Extractor == 'Official_RGA_Dynamic') & (cand.Is_Canonical == 1) & (cand.Candidate_Rank <= a.topk) & cand.Session_ID.isin(ep.Session_ID)]
    cores = pd.read_csv(SD + 'CORES_REFERENCE.csv', low_memory=False)
    cores = cores[cores.Session_ID_Matched.isin(ep.Session_ID)]
    ms = pd.read_csv('sources/Phase3_Patient_Session_Level_Master.csv', low_memory=False).set_index('Session_ID')
    return ep, cand, cores, ms, sorted(cand.Model.unique())


def p2s(P, A, B):
    """[n points, m segments] point-to-segment distances in mm."""
    AB = B - A
    L2 = (AB ** 2).sum(1)
    L2[L2 == 0] = 1e-9
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    return np.sqrt(((P[:, None, :] - (A[None] + t[..., None] * AB[None])) ** 2).sum(2))


def pbin_tail(obs, p):
    """Exact Poisson-binomial: expected count and P(X >= obs)."""
    d = np.array([1.0])
    for x in p:
        d = np.convolve(d, [1 - x, x])
    return float(np.sum(p)), float(d[obs:].sum())


def cand_xyz(cand, model):
    c = cand[cand.Model == model]
    return {s: g[['CoM_LPS_X_mm', 'CoM_LPS_Y_mm', 'CoM_LPS_Z_mm']].to_numpy(float) for s, g in c.groupby('Session_ID')}


def bbox(ms, s):
    mn = ms.loc[s, ['Prostate_LPS_Min_X', 'Prostate_LPS_Min_Y', 'Prostate_LPS_Min_Z']].to_numpy(float)
    sp = ms.loc[s, ['Prostate_LPS_Span_X', 'Prostate_LPS_Span_Y', 'Prostate_LPS_Span_Z']].to_numpy(float)
    return mn, sp


_TS = None


def target_units(sid, lbl):
    """Radiologist targets of one examination, one per public target surface (linked by B6c_link_target_scores.py).
    Returns a list of dicts: tno (target number), mask (voxels of all mask components of that surface), score (suspicion score of the target,
    highest value if biopsy sessions disagree), conflict (1 if they disagree). Components without a surface (1-8 voxel rasterization fragments) are dropped."""
    global _TS
    if _TS is None:
        t = pd.read_csv(OUT + 'TARGET_SCORES.csv')
        _TS = {k: g for k, g in t[t.target_score.notna()].groupby('Session_ID')}
    g = _TS.get(sid)
    if g is None:
        return []
    units = []
    for tno, q in g.groupby(g.target_nos.astype(float).astype(int)):
        units.append(dict(tno=int(tno), mask=np.isin(lbl, q.target.to_numpy()), score=int(q.target_score.max()), conflict=int(q.score_conflict.max())))
    return units


def gee_fit(y, X, groups, term='agree'):
    """Logistic GEE clustered by patient. Exchangeable working correlation; if that fit does not converge to a finite estimate
    (happens in small strata such as score 5 targets), the independence working correlation is used (robust standard errors in both)."""
    import statsmodels.api as sm
    r = None
    for cs in (sm.cov_struct.Exchangeable(), sm.cov_struct.Independence()):
        try:
            r = sm.GEE(y, X, groups=groups, family=sm.families.Binomial(), cov_struct=cs).fit()
            b, se = float(r.params[term]), float(r.bse[term])
            if bool(getattr(r, 'converged', True)) and np.isfinite(b) and np.isfinite(se) and abs(b) < 10 and 0 < se < 5:
                r.working = type(cs).__name__
                return r
        except Exception:
            pass
    if r is not None:
        r.working = 'failed'
    return r

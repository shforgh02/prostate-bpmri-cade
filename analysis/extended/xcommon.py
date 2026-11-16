# Shared helpers for the further analyses X1-X9.
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'analysis')
from _common import SD, CC, p2s, pbin_tail, bbox  # noqa: F401

R = 'outputs/extended/'
O = 'outputs/'
MODELS = ['3D_SegResNet', 'nnUNet_official', 'ProViCNet']
NAME = {'3D_SegResNet': 'SegResNet', 'nnUNet_official': 'nnU-Net', 'ProViCNet': 'ProViCNet'}
TOLS = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0)


def exam_table():
    return pd.read_csv(R + 'X0_exam_table.csv')


def perm_table():
    return pd.read_csv(R + 'X0_perm_by_k.csv')


def excluded():
    return set(pd.read_csv(O + 'EXCLUDED_SESSIONS_v2.csv').Session_ID)


def cores_757():
    c = pd.read_csv(SD + 'CORES_REFERENCE.csv', low_memory=False)
    keep = set(exam_table().Session_ID)
    return c[c.Session_ID_Matched.isin(keep)]


def master():
    return pd.read_csv('sources/Phase3_Patient_Session_Level_Master.csv', low_memory=False).set_index('Session_ID')


def candidates(topk=5):
    c = pd.read_csv(O + 'CANDIDATES_3MODELS.csv', low_memory=False)
    keep = set(exam_table().Session_ID)
    return c[(c.Extractor == 'Official_RGA_Dynamic') & (c.Is_Canonical == 1) & (c.Candidate_Rank <= topk) & c.Session_ID.isin(keep)]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n, d=1):
    return f'{k}/{n} ({100 * k / n:.{d}f}%)' if n else f'{k}/0'


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return ''
    return 'p < 0.001' if round(p, 3) < 0.001 else (f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}')


def offtarget_cores(cores, ms, offdist=10.0):
    """Systematic cores with a valid trajectory, flagged near/off the radiologist target exactly as in A2."""
    cv = cores[cores.Has_Valid_Physical_Trajectory == 1]
    out = []
    for s, g in cv.groupby('Session_ID_Matched'):
        sy, tg = g[g.Is_Systematic_Core == 1], g[g.Is_Targeted_Core == 1]
        if len(sy) == 0:
            continue
        A, B = sy[CC[:3]].to_numpy(float), sy[CC[3:]].to_numpy(float)
        near = np.zeros(len(sy), bool)
        if len(tg):
            TA, TB = tg[CC[:3]].to_numpy(float), tg[CC[3:]].to_numpy(float)
            near |= p2s(np.concatenate([TA + (TB - TA) * f for f in np.linspace(0, 1, 9)]), A, B).min(0) <= offdist
        h = ms.loc[s, ['Human_Rad_CoM_LPS_X_mm', 'Human_Rad_CoM_LPS_Y_mm', 'Human_Rad_CoM_LPS_Z_mm']].to_numpy(float)
        if np.isfinite(h).all():
            near |= p2s(h[None], A, B)[0] <= offdist
        out.append(dict(Session_ID=s, A=A, B=B, y=sy.Is_csPCa.to_numpy(int), near=near))
    return out


def gee_or(df, ycol, xcols, group):
    import statsmodels.api as sm
    X = sm.add_constant(df[xcols].astype(float))
    r = sm.GEE(df[ycol].astype(float).to_numpy(), X, groups=df[group].to_numpy(), family=sm.families.Binomial(),
               cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = r.conf_int().loc[xcols[0]]
    return float(np.exp(r.params[xcols[0]])), float(np.exp(ci[0])), float(np.exp(ci[1])), float(r.pvalues[xcols[0]])

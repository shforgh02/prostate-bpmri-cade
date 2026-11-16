# Later biopsy sessions.
# Question: in men without csPCa at the first tracked biopsy session, is the model output on the baseline MRI related to csPCa found at
# a later tracked session that was registered to the same MRI examination?
# Cohort: analysed men (757) with cores from more than one session date, and no csPCa-positive core at the first session date.
# Outcome: csPCa in a core of a later session date ("later csPCa").
#   A. Per man: later csPCa by model agreement with a radiologist target at baseline (a top-3 candidate centre within 10 mm of a target centre),
#      logistic regression adjusted for the examination suspicion score and log PSA density; AUC of the highest candidate probability.
#   B. Per target: targets sampled at the first session without csPCa (cores within 10 mm of the target centre) and sampled again later;
#      later csPCa at the target by agreement (GEE clustered by man, adjusted for target score and log PSA density).
#   C. Per location: in men with later csPCa, a top-3 baseline candidate within 10 mm of a later csPCa-positive core, against the
#      between-patient permutation chance model of script A1 (donor candidate sets from all analysed examinations with the same number of candidates).
# Session dates: study dates of the public DICOM headers (shifted per patient by the archive; intervals preserved).
#   python analysis/F1_followup_biopsy.py   -> outputs/F1_followup_results.csv, F1_followup_men.csv, F1_followup_targets.csv
import os, sys, warnings
import numpy as np, pandas as pd, nibabel as nib
import statsmodels.api as sm
from scipy.ndimage import label
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SD, OUT, CC, p2s, pbin_tail, cand_xyz, bbox, target_units, gee_fit

ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
SESS = os.path.join(ROOT, 'MRI_US_Biopsy', 'Script', 'output', 'standardized_ucla_sessions_new')
ex = set(pd.read_csv(OUT + 'EXCLUDED_SESSIONS_v2.csv').Session_ID)
ep = pd.read_csv(SD + 'EXAMINATION_LEVEL_ENDPOINTS.csv')
ep = ep[~ep.Session_ID.isin(ex)].reset_index(drop=True)
cand = pd.read_csv(OUT + 'CANDIDATES_3MODELS.csv', low_memory=False)
cand = cand[(cand.Extractor == 'Official_RGA_Dynamic') & (cand.Is_Canonical == 1) & cand.Session_ID.isin(ep.Session_ID)]
top3 = cand[cand.Candidate_Rank <= 3]
MODELS = [('3D_SegResNet', 'SegResNet'), ('nnUNet_official', 'nnU-Net'), ('ProViCNet', 'ProViCNet')]
ms = pd.read_csv('sources/Phase3_Patient_Session_Level_Master.csv', low_memory=False).set_index('Session_ID')
md = pd.read_csv(os.path.join(ROOT, 'MRI_US_Biopsy', 'manifest-1694710246744', 'metadata.csv'))
md.columns = [c.strip() for c in md.columns]
date = md.set_index('Series UID')['Study Date']
c = pd.read_csv(SD + 'CORES_REFERENCE.csv', low_memory=False)
c = c[c.Session_ID_Matched.isin(ep.Session_ID)].copy()
c['d'] = pd.to_datetime(c['Series Instance UID (US)'].astype(str).map(date))
c['first'] = c.groupby('Session_ID_Matched').d.transform('min')
c['later'] = (c.d > c['first']).astype(int)
g = c.groupby('Session_ID_Matched')
S = pd.DataFrame({'base_cs': c[c.later == 0].groupby('Session_ID_Matched').Is_csPCa.max(), 'has_later': g.later.max()})
S['later_cs'] = c[c.later == 1].groupby('Session_ID_Matched').Is_csPCa.max().reindex(S.index)
S['base_gg'] = c[c.later == 0].groupby('Session_ID_Matched').ISUP_Grade.max().reindex(S.index).fillna(0)
F = S[(S.has_later == 1) & (S.base_cs == 0)].copy()
F['y'] = F.later_cs.astype(int)
ld = c[(c.later == 1) & c.Session_ID_Matched.isin(F.index)].drop_duplicates(['Session_ID_Matched', 'd'])
gap = (ld.d - ld['first']).dt.days
print(f'men with later sessions {int((S.has_later == 1).sum())} | without csPCa at the first session {len(F)} | later csPCa {int(F.y.sum())} ({F.y.mean():.1%}) | '
      f'baseline benign {int((F.base_gg == 0).sum())}, grade group 1 {int((F.base_gg == 1).sum())} | days first to later session: median {gap.median():.0f} (IQR {gap.quantile(.25):.0f}-{gap.quantile(.75):.0f})')
out = []


def wil(k, n):
    lo, hi = proportion_confint(k, n, method='wilson') if n else (np.nan, np.nan)
    return f'{k}/{n} = {100 * k / max(n, 1):.1f}% [{100 * lo:.1f}-{100 * hi:.1f}]'


def add(analysis, group, k, n, **kw):
    lo, hi = proportion_confint(k, n, method='wilson') if n else (np.nan, np.nan)
    out.append(dict(analysis=analysis, group=group, k=int(k), n=int(n), rate=k / n if n else np.nan, lo=lo, hi=hi, **kw))


# ---------------------------------------------------------------- targets of the follow-up cohort
cv = c[c.Has_Valid_Physical_Trajectory == 1]
CX = {m: cand_xyz(top3, m) for m, _ in MODELS}
rows = []
for s in F.index:
    f = os.path.join(SESS, s, 'target_1_mask.nii.gz')
    if not os.path.exists(f):
        continue
    nii = nib.load(f)
    lbl, _ = label(np.asarray(nii.dataobj) > 0)
    cs_ = cv[cv.Session_ID_Matched == s]
    for u in target_units(s, lbl):
        com = np.argwhere(u['mask']).mean(0)
        lps = (nii.affine[:3, :3] @ com + nii.affine[:3, 3]) * np.array([-1.0, -1.0, 1.0])
        r = dict(Session_ID=s, Patient_ID=ms.loc[s, 'Patient_ID'] if 'Patient_ID' in ms.columns else s.rsplit('_', 1)[0], target=u['tno'], score=u['score'])
        for lab, sub in (('first', cs_[cs_.later == 0]), ('later', cs_[cs_.later == 1])):
            if len(sub):
                dist = p2s(lps[None], sub[CC[:3]].to_numpy(float), sub[CC[3:]].to_numpy(float))[0]
                near = dist <= 10
                r[f'{lab}_cores'] = int(near.sum()); r[f'{lab}_cs'] = int((sub.Is_csPCa.to_numpy(int)[near] == 1).any()) if near.any() else 0
            else:
                r[f'{lab}_cores'] = 0; r[f'{lab}_cs'] = 0
        for m, _ in MODELS:
            cc_ = CX[m].get(s)
            r['agree_' + m] = int(cc_ is not None and np.sqrt(((cc_ - lps[None]) ** 2).sum(1)).min() <= 10)
        rows.append(r)
T = pd.DataFrame(rows)
T['n_agree'] = T[[f'agree_{m}' for m, _ in MODELS]].sum(1)
T['all3'] = (T.n_agree == 3).astype(int)
T = T.merge(ms[['PSA', 'MRI_Prostate_Vol_cc']].reset_index(), on='Session_ID', how='left')
T['lpsad'] = np.log(T.PSA / T.MRI_Prostate_Vol_cc)
T.to_csv(OUT + 'F1_followup_targets.csv', index=False)

# ---------------------------------------------------------------- A. per man
E = F.join(ep.set_index('Session_ID')[['Exact_Exam_UCLA_Score']]).join(ms[['PSA', 'MRI_Prostate_Vol_cc']])
E['lpsad'] = np.log(E.PSA / E.MRI_Prostate_Vol_cc)
tm = T.groupby('Session_ID').agg(n_targets=('target', 'size'), n_agree=('n_agree', 'max'), **{f'agree_{m}': (f'agree_{m}', 'max') for m, _ in MODELS})
E = E.join(tm)
for col in ['n_targets', 'n_agree'] + [f'agree_{m}' for m, _ in MODELS]:
    E[col] = E[col].fillna(0).astype(int)
E['all3'] = (E.n_agree == 3).astype(int)
for m, _ in MODELS:
    E['prob_' + m] = cand[(cand.Model == m) & (cand.Candidate_Rank == 1)].set_index('Session_ID').Max_Prob.reindex(E.index).fillna(0.0)
E.reset_index().rename(columns={'Session_ID_Matched': 'Session_ID'}).to_csv(OUT + 'F1_followup_men.csv', index=False)
print('\nA. per man (n = %d, later csPCa %d)' % (len(E), int(E.y.sum())))
add('A', 'all men', E.y.sum(), len(E))
for lab, msk in [('all three models agree with a target', E.all3 == 1), ('fewer than three models agree', E.all3 == 0)] + \
                [(f'{k} models agree', E.n_agree == k) for k in range(4)] + \
                [x for m, nm in MODELS for x in ((f'{nm} agrees with a target', E[f'agree_{m}'] == 1), (f'{nm} does not agree', E[f'agree_{m}'] == 0))] + \
                [('highest suspicion score 3', E.Exact_Exam_UCLA_Score == 3), ('highest suspicion score 4-5', E.Exact_Exam_UCLA_Score.isin([4, 5])),
                 ('baseline grade group 1', E.base_gg == 1), ('baseline benign', E.base_gg == 0)]:
    q = E[msk]
    print(f'  {lab:42s} later csPCa {wil(int(q.y.sum()), len(q))}')
    add('A', lab, q.y.sum(), len(q))
_, pf = fisher_exact(pd.crosstab(E.all3, E.y).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy())
print(f'  Fisher exact test, all three agree vs fewer: p = {pf:.4f}')
out.append(dict(analysis='A', group='Fisher exact p, all three agree vs fewer', p=pf))
K = E[E.Exact_Exam_UCLA_Score.isin([1, 2, 3, 4, 5]) & np.isfinite(E.lpsad)]
for var, lab in [('all3', 'all three models agree (adjusted)'), ('n_agree', 'per additional agreeing model (adjusted)')] + [(f'agree_{m}', f'{nm} agrees (adjusted)') for m, nm in MODELS]:
    X = sm.add_constant(pd.DataFrame({'x': K[var].to_numpy(float), 'score': K.Exact_Exam_UCLA_Score.to_numpy(float), 'lpsad': K.lpsad.to_numpy(float)}))
    try:
        r = sm.Logit(K.y.to_numpy(), X).fit(disp=0)
        b, se = r.params['x'], r.bse['x']
        print(f'  {lab:42s} n={len(K)} OR {np.exp(b):.2f} [{np.exp(b - 1.96 * se):.2f}-{np.exp(b + 1.96 * se):.2f}] p={r.pvalues["x"]:.4f}')
        out.append(dict(analysis='A', group='OR ' + lab, n=len(K), OR=np.exp(b), or_lo=np.exp(b - 1.96 * se), or_hi=np.exp(b + 1.96 * se), p=r.pvalues['x']))
    except Exception as e_:
        print('  model failed:', lab, str(e_)[:80])
rng = np.random.default_rng(20261115)


def auc_ci(y, x):
    a = roc_auc_score(y, x); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            bs.append(roc_auc_score(y[i], x[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


for lab, x in [('suspicion score', K.Exact_Exam_UCLA_Score), ('PSA density', K.lpsad)] + [(nm + ' highest candidate probability', K['prob_' + m]) for m, nm in MODELS] + [('number of agreeing models', K.n_agree)]:
    a, lo, hi = auc_ci(K.y.to_numpy(), x.to_numpy(float))
    print(f'  AUC for later csPCa, {lab:40s} {a:.3f} [{lo:.3f}-{hi:.3f}]')
    out.append(dict(analysis='A', group='AUC ' + lab, n=len(K), auc=a, auc_lo=lo, auc_hi=hi))

# ---------------------------------------------------------------- B. per target
B_ = T[(T.first_cores > 0) & (T.first_cs == 0) & (T.later_cores > 0)].copy()
B_['y'] = B_.later_cs
print(f'\nB. per target: targets of these men {len(T)} | sampled at the first session without csPCa and sampled again later {len(B_)} in {B_.Session_ID.nunique()} men | later csPCa at the target {int(B_.y.sum())}')
add('B', 'all targets', B_.y.sum(), len(B_))
for lab, msk in [('all three models agree', B_.all3 == 1), ('fewer than three models agree', B_.all3 == 0)] + [(f'{k} models agree', B_.n_agree == k) for k in range(4)] + \
                [x for m, nm in MODELS for x in ((f'{nm} agrees', B_[f'agree_{m}'] == 1), (f'{nm} does not agree', B_[f'agree_{m}'] == 0))] + \
                [('target score 3', B_.score == 3), ('target score 3, all three agree', (B_.score == 3) & (B_.all3 == 1)), ('target score 3, fewer agree', (B_.score == 3) & (B_.all3 == 0)),
                 ('target score 4-5', B_.score.isin([4, 5]))]:
    q = B_[msk]
    print(f'  {lab:42s} later csPCa at the target {wil(int(q.y.sum()), len(q))}')
    add('B', lab, q.y.sum(), len(q))
G = B_[B_.score.isin([3, 4, 5]) & np.isfinite(B_.lpsad)]
for var, lab in [('all3', 'all three models agree (adjusted)'), ('n_agree', 'per additional agreeing model (adjusted)')] + [(f'agree_{m}', f'{nm} agrees (adjusted)') for m, nm in MODELS]:
    X = sm.add_constant(pd.DataFrame({'agree': G[var].to_numpy(float), 's4': (G.score == 4).to_numpy(float), 's5': (G.score == 5).to_numpy(float), 'lpsad': G.lpsad.to_numpy(float)}))
    try:
        r = gee_fit(G.y.to_numpy(), X, G.Session_ID.to_numpy())
        b, se = float(r.params['agree']), float(r.bse['agree'])
        print(f'  {lab:42s} n={len(G)} OR {np.exp(b):.2f} [{np.exp(b - 1.96 * se):.2f}-{np.exp(b + 1.96 * se):.2f}] p={float(r.pvalues["agree"]):.4f} ({r.working})')
        out.append(dict(analysis='B', group='OR ' + lab, n=len(G), OR=np.exp(b), or_lo=np.exp(b - 1.96 * se), or_hi=np.exp(b + 1.96 * se), p=float(r.pvalues['agree'])))
    except Exception as e_:
        print('  model failed:', lab, str(e_)[:80])

# ---------------------------------------------------------------- C. per location (men with later csPCa)
up = F.index[F.y == 1]
pos = cv[(cv.later == 1) & (cv.Is_csPCa == 1) & cv.Session_ID_Matched.isin(up)]
SEG = {s: (q[CC[:3]].to_numpy(float), q[CC[3:]].to_numpy(float)) for s, q in pos.groupby('Session_ID_Matched')}
BB = {s: bbox(ms, s) for s in ep.Session_ID}
TC = {}
for s in up:
    f = os.path.join(SESS, s, 'target_1_mask.nii.gz')
    cen = []
    if os.path.exists(f):
        nii = nib.load(f)
        lbl, _ = label(np.asarray(nii.dataobj) > 0)
        for u in target_units(s, lbl):
            com = np.argwhere(u['mask']).mean(0)
            cen.append((nii.affine[:3, :3] @ com + nii.affine[:3, 3]) * np.array([-1.0, -1.0, 1.0]))
    TC[s] = np.array(cen) if cen else np.zeros((0, 3))
ev = [s for s in up if s in SEG]
rad = {s: bool(len(TC[s]) and p2s(TC[s], *SEG[s]).min() <= 10) for s in ev}
print(f'\nC. per location: men with later csPCa {len(up)} | with a tracked later csPCa-positive core {len(ev)} | radiologist target within 10 mm of a later positive core {wil(sum(rad.values()), len(ev))}')
add('C', 'radiologist target', sum(rad.values()), len(ev))
for m, nm in MODELS:
    C_ = cand_xyz(top3, m)
    NORM = {s: (C_[s] - BB[s][0]) / BB[s][1] for s in C_}
    obs, pr = [], []
    for s in ev:
        A, Bq = SEG[s]
        mn, sp = BB[s]
        k = len(C_.get(s, []))
        obs.append(int(k > 0 and p2s(C_[s], A, Bq).min() <= 10))
        donors = [v for j, v in NORM.items() if j != s and len(v) == k] if k else []
        pr.append(float(np.mean([p2s(mn + v * sp, A, Bq).min() <= 10 for v in donors])) if donors else 0.0)
    obs, pr = np.array(obs), np.array(pr)
    for lab, msk in (('all', np.ones(len(ev), bool)), ('radiologist target within 10 mm', np.array([rad[s] for s in ev])), ('radiologist target not within 10 mm', ~np.array([rad[s] for s in ev]))):
        o = int(obs[msk].sum()); e, p = pbin_tail(o, pr[msk])
        print(f'  {nm:10s} {lab:36s} baseline top-3 candidate within 10 mm of a later positive core {wil(o, int(msk.sum()))} | chance {e:.1f} | p={p:.4f}')
        add('C', f'{nm}: {lab}', o, int(msk.sum()), expected=e, p=p)
pd.DataFrame(out).to_csv(OUT + 'F1_followup_results.csv', index=False)

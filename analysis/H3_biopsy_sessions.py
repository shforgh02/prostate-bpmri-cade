# Biopsy sessions per analysed MRI examination, from the study dates of the public DICOM headers (dates are shifted per patient by the
# archive, intervals are preserved). Cores of all tracked sessions registered to the analysed MRI are part of the reference standard;
# this script counts men with more than one session date and writes the exclusion list for the single-session sensitivity analysis.
#   python analysis/H3_biopsy_sessions.py
#   -> outputs/BIOPSY_SESSIONS.csv (per examination), outputs/BIOPSY_SESSIONS_summary.csv, outputs/EXCLUDED_SESSIONS_v2_plus_multidate.csv
import os
import numpy as np, pandas as pd

O = 'outputs/'
ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
md = pd.read_csv(os.path.join(ROOT, 'MRI_US_Biopsy', 'manifest-1694710246744', 'metadata.csv'))
md.columns = [c.strip() for c in md.columns]
date = md.set_index('Series UID')['Study Date']
ex = set(pd.read_csv(O + 'EXCLUDED_SESSIONS_v2.csv').Session_ID)
ep = pd.read_csv('results/tables/EXAMINATION_LEVEL_ENDPOINTS.csv')
ep = ep[~ep.Session_ID.isin(ex)]
c = pd.read_csv('results/tables/CORES_REFERENCE.csv', low_memory=False)
c = c[c.Session_ID_Matched.isin(ep.Session_ID)].copy()
c['us_date'] = pd.to_datetime(c['Series Instance UID (US)'].astype(str).map(date), errors='coerce')
c['mr_date'] = pd.to_datetime(c['MRI_UID'].astype(str).map(date), errors='coerce')
assert c.us_date.notna().all() and c.mr_date.notna().all()
g = c.groupby('Session_ID_Matched')
S = pd.DataFrame({'n_cores': g.size(), 'n_session_dates': g.us_date.nunique(), 'first_session': g.us_date.min(), 'mri': g.mr_date.first()})
S['days_mri_to_first_session'] = (S.first_session - S.mri).dt.days
c['first'] = c.Session_ID_Matched.map(S.first_session)
later = c[c.us_date > c['first']]
S['n_cores_later_sessions'] = later.groupby('Session_ID_Matched').size().reindex(S.index).fillna(0).astype(int)
ld = later.drop_duplicates(['Session_ID_Matched', 'us_date'])
gap = (ld.us_date - ld['first']).dt.days
S.reset_index().rename(columns={'Session_ID_Matched': 'Session_ID'}).to_csv(O + 'BIOPSY_SESSIONS.csv', index=False)
multi = S.index[S.n_session_dates > 1]
pd.DataFrame({'Session_ID': sorted(ex | set(multi))}).to_csv(O + 'EXCLUDED_SESSIONS_v2_plus_multidate.csv', index=False)
q = lambda v, p: float(np.percentile(v, p))
summ = dict(men=len(S), men_multi=len(multi), men_single=int((S.n_session_dates == 1).sum()), cores=len(c), cores_later=len(later), cspca_cores=int(c.Is_csPCa.sum()),
            cspca_cores_later=int(later.Is_csPCa.sum()), mri_to_first_median=float(S.days_mri_to_first_session.median()), mri_to_first_q1=q(S.days_mri_to_first_session, 25),
            mri_to_first_q3=q(S.days_mri_to_first_session, 75), later_gap_median=float(gap.median()), later_gap_q1=q(gap, 25), later_gap_q3=q(gap, 75),
            mri_after_first_session=int((S.days_mri_to_first_session < 0).sum()))
pd.DataFrame([summ]).to_csv(O + 'BIOPSY_SESSIONS_summary.csv', index=False)
print(summ)
print('session dates per man:', S.n_session_dates.value_counts().sort_index().to_dict())

# Table 1: characteristics of the analyzed examinations. Descriptive only; no new analysis.
#   python analysis/D3_table1.py   -> tables/Table1_characteristics.csv (+ appended to tables/TABLES.md)
import os, re
import numpy as np, pandas as pd

SD = 'results/tables/'
O, TAB = 'outputs/', 'tables/'

ep = pd.read_csv(SD + 'EXAMINATION_LEVEL_ENDPOINTS.csv')
ep = ep[~ep.Session_ID.isin(pd.read_csv(O + 'EXCLUDED_SESSIONS_v2.csv').Session_ID)].copy()
ms = pd.read_csv('sources/Phase3_Patient_Session_Level_Master.csv', low_memory=False).set_index('Session_ID')
cores = pd.read_csv(SD + 'CORES_REFERENCE.csv', low_memory=False)
cores = cores[cores.Session_ID_Matched.isin(ep.Session_ID)]
tg = pd.read_csv(O + 'B6_targets_final_qc.csv')
sc = pd.read_csv('outputs/scanner_coil_by_session.csv').set_index('Session_ID')
man = pd.read_csv('results/tables/ANALYSIS_MANIFEST.csv', nrows=1)
N = len(ep)
rows = []


def add(section, item, value):
    rows.append(dict(Characteristic=section, Item=item, Value=value))


def n_pct(k, d=N):
    return f'{int(k)} ({100 * k / d:.1f}%)'


def med_iqr(x, fmt='{:.1f}'):
    x = pd.Series(x, dtype=float).dropna()
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f'{fmt.format(q2)} [{fmt.format(q1)}–{fmt.format(q3)}] (n = {len(x)})'


add('Men (one examination each)', '', str(ep.Patient_ID.nunique()))
AGE_F = 'outputs/extended/X0_dicom_header.csv'   # PatientAge from the DICOM header
if os.path.exists(AGE_F):
    age = pd.read_csv(AGE_F).set_index('Session_ID').reindex(ep.Session_ID).age
    add('Age, years (DICOM header)', 'median [IQR]', med_iqr(age, '{:.0f}'))
else:
    add('Age, years', '', 'TO FILL (no age source)')

psa = ms.loc[ep.Session_ID, 'PSA'].to_numpy(float)
vol = ms.loc[ep.Session_ID, 'MRI_Prostate_Vol_cc'].to_numpy(float)
add('PSA, ng/mL', 'median [IQR]', med_iqr(psa))
add('Prostate volume on MRI, mL', 'median [IQR]', med_iqr(vol, '{:.0f}'))
psad = np.where((vol > 0) & np.isfinite(psa), psa / vol, np.nan)
add('PSA density, ng/mL²', 'median [IQR]', med_iqr(psad, '{:.2f}'))

s = ep.Exact_Exam_UCLA_Score
add('UCLA suspicion score (examination)', '1–2', n_pct(s.isin([1, 2]).sum()))
add('UCLA suspicion score (examination)', '3', n_pct((s == 3).sum()))
add('UCLA suspicion score (examination)', '4–5', n_pct(s.isin([4, 5]).sum()))
add('UCLA suspicion score (examination)', 'not recorded', n_pct((~s.isin([1, 2, 3, 4, 5])).sum()))

g = ms.loc[ep.Session_ID, 'Sess_Max_ISUP']
for lab, m in (('benign', g == 0), ('1', g == 1), ('2', g == 2), ('3', g == 3), ('4–5', g.isin([4, 5]))):
    add('Highest grade group on biopsy', lab, n_pct(m.sum()))
other = (~g.isin([0, 1, 2, 3, 4, 5])).sum()
if other:
    add('Highest grade group on biopsy', 'other / missing', n_pct(other))

cs = int(ep.Is_csPCa.sum())
evl = int(((ep.Is_csPCa == 1) & (ep.Has_Valid_3D_Trajectory == 1)).sum())
add('csPCa (grade group ≥ 2)', '', n_pct(cs))
add('Evaluable cancers (csPCa with a recorded core position)', '', n_pct(evl))

per = cores.groupby('Session_ID_Matched').agg(all=('Is_csPCa', 'size'), targeted=('Is_Targeted_Core', 'sum'), systematic=('Is_Systematic_Core', 'sum'))
per = per.reindex(ep.Session_ID).fillna(0)
add('Cores per examination', f'all (total {len(cores):,})', med_iqr(per['all'], '{:.0f}'))
add('Cores per examination', f'targeted (total {int(cores.Is_Targeted_Core.sum()):,})', med_iqr(per['targeted'], '{:.0f}'))
add('Cores per examination', f'systematic (total {int(cores.Is_Systematic_Core.sum()):,})', med_iqr(per['systematic'], '{:.0f}'))

tg = tg[tg.Session_ID.isin(ep.Session_ID)]
sampled = tg.path.isin(['csPCa', 'GG1', 'benign'])
add('Radiologist targets', 'total', str(len(tg)))
add('Radiologist targets', 'sampled (cores within 10 mm of the target center)', str(int(sampled.sum())))
add('Radiologist targets', 'csPCa-positive', str(int((tg.path == 'csPCa').sum())))

f = sc.loc[ep.Session_ID, 't2_field']
v = sc.loc[ep.Session_ID, 't2_manufacturer'].astype(str).str.upper()
vend = np.select([v.str.contains('SIEMENS'), v.str.contains('GE '), v.str.contains('PHILIPS')], ['Siemens', 'GE', 'Philips'], 'other')
add('Field strength', '3.0 T', n_pct((f == 3.0).sum()))
add('Field strength', '1.5 T', n_pct((f == 1.5).sum()))
for lab in ('Siemens', 'GE', 'Philips', 'other'):
    k = int((vend == lab).sum())
    if k or lab != 'other':
        add('Vendor', lab, n_pct(k))

T1 = pd.DataFrame(rows)
# consistency checks against the reported counts
chk = {'csPCa = 411': cs == 411, 'evaluable = 387': evl == 387, 'targets 955/880/402': (len(tg), int(sampled.sum()), int((tg.path == 'csPCa').sum())) == (955, 880, 402),
       '29 at 1.5 T': int((f == 1.5).sum()) == 29, '8 on GE or Philips': int(np.isin(vend, ['GE', 'Philips']).sum()) == 8, 'N = 757': N == 757}
for k, ok in chk.items():
    print(('PASS ' if ok else 'FAIL ') + k)
T1.to_csv(TAB + 'Table1_characteristics.csv', index=False)
md = open(TAB + 'TABLES.md', encoding='utf-8').read()
md = re.sub(r'\n## Table 1\. .*?(?=\n## |\Z)', '', md, flags=re.S)       # replace an earlier copy if rerun
md += ('\n## Table 1. Characteristics of the 757 examinations\n\n' + T1.to_markdown(index=False) +
       '\n\nIQR, interquartile range. PSA density = PSA / MRI prostate volume. Grade group = highest ISUP grade group on any core of the examination.'
       '\n\n<!-- Table1_characteristics.csv -->\n')
open(TAB + 'TABLES.md', 'w', encoding='utf-8').write(md)
print(T1.to_string(index=False))

# Usage: python analysis/H1_highb_series_audit.py <MRI_US_Biopsy dir> <out csv>   (reads DICOM headers; about 10 minutes)
# Read-only. Needs pandas, numpy, scipy, scikit-learn, SimpleITK .
"""Audit of the UCLA (TCIA Prostate-MRI-US-Biopsy) series picked by the earlier pipeline.

Usage: python -I ucla_audit.py <MRI_US_Biopsy dir> <out_csv> [max_sessions]
Read-only. Reproduces the series picker of 01_phase1_preprocessing_and_standardization.py
(lines 316-381), then reads DICOM headers of the picked diffusion series.
"""
import sys, os, re, zipfile, collections, time, glob
import xml.etree.ElementTree as ET
import pandas as pd, numpy as np
import SimpleITK as sitk

ROOT = sys.argv[1]; OUT = sys.argv[2]
MAXS = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
MAN = os.path.join(ROOT, "manifest-1694710246744")
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_xlsx(path):
    """Minimal .xlsx reader (first sheet), no extra packages."""
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            ss.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
    rows = []
    for r in ET.fromstring(z.read(sheet)).iter("{%s}row" % NS["m"]):
        d = {}
        for c in r.findall("m:c", NS):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find("m:v", NS); t = c.get("t")
            if t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter("{%s}t" % NS["m"]))
            elif v is None:
                val = None
            elif t == "s":
                val = ss[int(v.text)]
            else:
                val = v.text
            d[col] = val
        rows.append(d)
    hdr = rows[0]
    cols = sorted(hdr, key=lambda k: (len(k), k))
    return pd.DataFrame([{hdr[c]: r.get(c) for c in cols} for r in rows[1:]])


meta = pd.read_csv(os.path.join(MAN, "metadata.csv"))
meta.columns = [c.strip() for c in meta.columns]
print("series in manifest:", len(meta))   # the Modality column is incomplete in this manifest, so it is not used

bx = read_xlsx(os.path.join(ROOT, "TCIA-Biopsy-Data_2020-07-14.xlsx"))
print("\nbiopsy rows:", len(bx), "| columns:", list(bx.columns))
bx = bx[bx["Bx Tip X (MRI Coord)"].notna()]
sessions = collections.OrderedDict()
for _, row in bx.iterrows():
    pid = str(row["Patient Number"]).strip(); uid = str(row.get("Series Instance UID (MRI)", "")).strip()
    if not uid or uid in ("nan", "None"):
        continue
    sessions.setdefault(f"{pid}_{uid[-5:]}", (pid, uid))
print("candidate MRI-biopsy sessions:", len(sessions))

EXCL = "ADC|Apparent|Localizer|Screen|T2|TSE|SPC|AX|Sag|Cor|Survey|Recon"
PATS = [r"1400", r"1500", r"CALCBVAL", r"TRACEW", r"DWI", r"diff"]
rec = []; drop = collections.Counter()
for sid, (pid, uid) in sessions.items():
    t2 = meta[meta["Series UID"] == uid]
    if t2.empty:
        drop["T2 series not in manifest"] += 1; continue
    t2r = t2.iloc[0]
    st = meta[meta["Study UID"] == t2r["Study UID"]]
    adc = st[st["Series Description"].str.contains("ADC|Apparent", case=False, na=False)]
    if adc.empty:
        drop["no ADC series in study"] += 1; continue
    cand = st[~st["Series Description"].str.contains(EXCL, case=False, na=False)]
    if cand.empty:
        drop["no diffusion candidate"] += 1; continue
    pick, how = None, "FALLBACK first leftover"
    for p in PATS:
        m = cand[cand["Series Description"].str.contains(p, case=False, na=False)]
        if not m.empty:
            pick, how = m.iloc[0], p; break
    if pick is None:
        pick = cand.iloc[0]
    # what a stricter picker would have had available in the same study
    alldesc = list(st["Series Description"].fillna(""))
    rec.append(dict(session=sid, t2_desc=t2r["Series Description"], t2_n=t2r["Number of Images"],
                    adc_desc=adc.iloc[0]["Series Description"], adc_n=adc.iloc[0]["Number of Images"], n_adc_series=len(adc),
                    dwi_desc=pick["Series Description"], dwi_n=pick["Number of Images"], matched=how, n_candidates=len(cand),
                    dwi_loc=pick["File Location"], adc_loc=adc.iloc[0]["File Location"],
                    study_has_calc=any(re.search("CALC", d, re.I) for d in alldesc),
                    excluded_diffusion=[d for d in alldesc if re.search("diff|dwi|trace|ep2d", d, re.I) and re.search(EXCL, d, re.I) and not re.search("ADC|Apparent", d, re.I)]))
R = pd.DataFrame(rec)
print("dropped:", dict(drop), "| sessions kept by the picker:", len(R))
print("\nwhich pattern picked the 'high-b' series:"); print(R.matched.value_counts().to_string())
print("\npicked 'high-b' series descriptions (digits -> #), top 25:")
print(R.dwi_desc.str.replace(r"\d+", "#", regex=True).value_counts().head(25).to_string())
print("\nADC series descriptions picked, top 8:"); print(R.adc_desc.str.replace(r"\d+", "#", regex=True).value_counts().head(8).to_string())
print("\nimages per series: T2 median", R.t2_n.median(), "| ADC median", R.adc_n.median(), "| picked DWI median", R.dwi_n.median())
print("picked DWI images / ADC images ratio:"); print((R.dwi_n / R.adc_n).round(1).value_counts().head(8).to_string())
ex = R[R.excluded_diffusion.map(len) > 0]
print("\nsessions where a diffusion-looking series was EXCLUDED by the name filter:", len(ex))
print(pd.Series([d for l in ex.excluded_diffusion for d in l]).str.replace(r"\d+", "#", regex=True).value_counts().head(10).to_string())

# ---- DICOM headers of the picked diffusion series ----
rd = sitk.ImageFileReader(); rd.LoadPrivateTagsOn()
def tags(fp):
    rd.SetFileName(fp); rd.ReadImageInformation()
    g = lambda k: rd.GetMetaData(k).strip() if rd.HasMetaDataKey(k) else ""
    return dict(seq=g("0018|0024"), bval=g("0018|9087"), bval_siemens=g("0019|100c"), itype=g("0008|0008"), ipp=g("0020|0032"), desc=g("0008|103e"))
def bfrom(seq):
    m = re.search(r"b(\d+)", seq or ""); return int(m.group(1)) if m else None

t0 = time.time(); hrows = []
for i, r in R.head(MAXS).iterrows():
    folder = os.path.join(MAN, r.dwi_loc.lstrip(".\\").replace("\\", os.sep))
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    o = dict(session=r.session, dwi_desc=r.dwi_desc, n_files=len(files))
    try:
        T = [tags(f) for f in files]
        bs = []
        for t in T:
            b = t["bval"] or t["bval_siemens"]
            b = int(float(b)) if re.fullmatch(r"[\d.]+", b or "") else bfrom(t["seq"])
            bs.append(b)
        o["bvals"] = "/".join(str(b) for b in sorted(set(bs), key=lambda x: (x is None, x)))
        o["n_bvals"] = len(set(bs))
        o["n_positions"] = len(set(t["ipp"] for t in T))
        o["seqnames"] = "/".join(sorted(set(t["seq"] for t in T)))[:80]
        o["calc"] = any("CALC" in t["itype"].upper() or "calc" in t["seq"].lower() for t in T)
        o["itype"] = sorted(set(t["itype"] for t in T))[0][:60] if T else ""
    except Exception as e:
        o["error"] = str(e)[:80]
    hrows.append(o)
    if (i + 1) % 100 == 0:
        print(f"  headers {i+1}/{min(len(R), MAXS)}  {time.time()-t0:.0f}s", flush=True)
H = pd.DataFrame(hrows)
R.merge(H, on=["session", "dwi_desc"], how="left").drop(columns=["dwi_loc", "adc_loc"]).to_csv(OUT, index=False)
print("\nheader audit sessions:", len(H), "| errors:", int(H.get("error", pd.Series(dtype=object)).notna().sum()), f"| {time.time()-t0:.0f}s")
print("\nb-values present in the picked 'high-b' series:"); print(H.bvals.value_counts(dropna=False).head(20).to_string())
print("\nnumber of distinct b-values per picked series:"); print(H.n_bvals.value_counts(dropna=False).to_string())
print("\nfiles per slice position (>1 means several images stacked per slice):"); print((H.n_files / H.n_positions).round(1).value_counts().head(8).to_string())
print("\ncomputed (CALC) image:"); print(H.calc.value_counts(dropna=False).to_string())
print("\nsequence names, top 12:"); print(H.seqnames.value_counts().head(12).to_string())


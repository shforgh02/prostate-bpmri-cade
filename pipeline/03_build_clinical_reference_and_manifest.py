import os
import sys
import shutil
import numpy as np
import pandas as pd

PROD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get(
    "PROSTATE_DATA_ROOT",
    os.path.abspath(os.path.join(PROD_DIR, "..", ".."))
)
AUDIT_DIR = os.path.join(PROD_DIR, "results", "tables")
SUBMISSION_SOURCE_DIR = os.path.join(PROD_DIR, "results", "source_data")
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_SOURCE_DIR, exist_ok=True)

def find_first_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    return path_list[0]

UCLA_BIOPSY_XLSX = find_first_existing([
    os.path.join(DATA_ROOT, "MRI_US_Biopsy", "TCIA-Biopsy-Data_2020-07-14.xlsx"),
])
UCLA_TARGET_XLSX = find_first_existing([
    os.path.join(DATA_ROOT, "MRI_US_Biopsy", "Target-Data_2019-12-05-2.xlsx"),
])
UCLA_METADATA_CSV = find_first_existing([
    os.path.join(DATA_ROOT, "MRI_US_Biopsy", "manifest-1694710246744", "metadata.csv"),
])
UCLA_MASTER_CSV = find_first_existing([
    os.path.join(DATA_ROOT, "MRI_US_Biopsy", "Script", "output", "ucla_master_clinical_cohort_new.csv"),
])
SEG_PROBMAP_DIR = find_first_existing([
    os.path.join(PROD_DIR, "results", "segresnet_probmaps"),
    os.path.join(DATA_ROOT, "Final Script", "antigravity", "results", "segresnet_probmaps"),
])
NNU_PRED_DIR = find_first_existing([
    os.path.join(DATA_ROOT, "MRI_US_Biopsy", "Script", "nnUNet_predictions", "predictions"),
])

def build_clinical_reference_and_manifest():
    print("=" * 80)
    print("BUILDING CLINICAL REFERENCE, COHORT MANIFEST & LABEL RECONCILIATION")
    print("=" * 80)

    # 1. Load raw biopsy spreadsheet
    df_bx = pd.read_excel(UCLA_BIOPSY_XLSX)
    df_bx["Original_Row_ID"] = df_bx.index
    df_bx["Patient Number"] = df_bx["Patient Number"].astype(str).str.strip()
    df_bx["Series Instance UID (MRI)"] = df_bx["Series Instance UID (MRI)"].astype(str).str.strip()
    df_bx["Series Instance UID (US)"] = df_bx["Series Instance UID (US)"].astype(str).str.strip()
    df_bx["Core Label"] = df_bx["Core Label"].astype(str).str.strip()

    # Identify clean MRI UID (handle string 'nan')
    df_bx["Has_Linked_MRI"] = (df_bx["Series Instance UID (MRI)"] != "nan") & (df_bx["Series Instance UID (MRI)"] != "")
    df_bx["MRI_UID"] = np.where(df_bx["Has_Linked_MRI"], df_bx["Series Instance UID (MRI)"], "")
    df_bx["MRI_UID_Suffix5"] = np.where(df_bx["Has_Linked_MRI"], df_bx["MRI_UID"].str[-5:], "")
    df_bx["Session_ID_Matched"] = np.where(
        df_bx["Has_Linked_MRI"],
        df_bx["Patient Number"] + "_" + df_bx["MRI_UID_Suffix5"],
        df_bx["Patient Number"] + "_TRUS_ONLY"
    )

    # 2. Classify Pathology independently of coordinates
    def classify_core_pathology(row):
        p_g = row["Primary Gleason"]
        s_g = row["Secondary Gleason"]
        c_len = row["Cancer Length (mm)"]
        c_pct = row["% Cancer in Core"]

        has_pg = pd.notna(p_g) and p_g > 0
        has_sg = pd.notna(s_g) and s_g > 0
        has_len = pd.notna(c_len) and c_len > 0
        has_pct = pd.notna(c_pct) and c_pct > 0

        if has_pg and has_sg:
            pg = int(p_g)
            sg = int(s_g)
            g_sum = pg + sg
            if pg == 3 and sg == 3:
                isup = 1
            elif pg == 3 and sg == 4:
                isup = 2
            elif pg == 4 and sg == 3:
                isup = 3
            elif g_sum == 8:
                isup = 4
            elif g_sum >= 9:
                isup = 5
            else:
                isup = 1 if g_sum <= 6 else 2

            if isup >= 2:
                return "csPCa", isup, 1, 0, 0
            else:
                return "GG1_indolent", 1, 0, 1, 0
        elif has_len or has_pct:
            # Cancer recorded, but Gleason grade missing
            return "cancer_missing_gleason", -1, 0, 0, 1
        else:
            # Benign / Negative
            return "benign", 0, 0, 0, 0

    path_results = [classify_core_pathology(r) for _, r in df_bx.iterrows()]
    df_bx["Pathology_Category"] = [p[0] for p in path_results]
    df_bx["ISUP_Grade"] = [p[1] for p in path_results]
    df_bx["Is_csPCa"] = [p[2] for p in path_results]
    df_bx["Is_GG1_Indolent"] = [p[3] for p in path_results]
    df_bx["Is_Cancer_Missing_Gleason"] = [p[4] for p in path_results]

    # 3. Classify Core Type
    def classify_core_type(lbl):
        l_upper = str(lbl).upper().strip()
        if "TARGET" in l_upper or "PRIOR POSITIVE" in l_upper:
            return "targeted"
        elif any(k in l_upper for k in ["APEX", "MID", "BASE", "LATERAL"]):
            return "systematic"
        elif "EXTRA" in l_upper:
            return "other_extra"
        else:
            return "unknown"

    df_bx["Core_Type"] = df_bx["Core Label"].apply(classify_core_type)
    df_bx["Is_Targeted_Core"] = (df_bx["Core_Type"] == "targeted").astype(int)
    df_bx["Is_Systematic_Core"] = (df_bx["Core_Type"] == "systematic").astype(int)

    # 4. Coordinate flags and verification
    coord_cols = [
        "Bx Base X (MRI Coord)", "Bx Base Y (MRI Coord)", "Bx Base Z (MRI Coord)",
        "Bx Tip X (MRI Coord)", "Bx Tip Y (MRI Coord)", "Bx Tip Z (MRI Coord)"
    ]
    for c in coord_cols:
        df_bx[c] = pd.to_numeric(df_bx[c], errors="coerce")

    df_bx["Coord_Completeness_AllFinite"] = df_bx[coord_cols].notna().all(axis=1).astype(int)
    df_bx["Coord_Sentinel_Neg1000"] = (df_bx[coord_cols] == -1000.0).any(axis=1).astype(int)

    # Needle segment length in physical MRI coordinates
    bases = df_bx[["Bx Base X (MRI Coord)", "Bx Base Y (MRI Coord)", "Bx Base Z (MRI Coord)"]].values
    tips = df_bx[["Bx Tip X (MRI Coord)", "Bx Tip Y (MRI Coord)", "Bx Tip Z (MRI Coord)"]].values
    vecs = tips - bases
    lengths = np.linalg.norm(vecs, axis=1)
    df_bx["Needle_Length_mm"] = np.where(df_bx["Coord_Completeness_AllFinite"] == 1, lengths, np.nan)
    df_bx["Has_Valid_Physical_Trajectory"] = (
        (df_bx["Coord_Completeness_AllFinite"] == 1) &
        (df_bx["Coord_Sentinel_Neg1000"] == 0) &
        (df_bx["Needle_Length_mm"] > 0.1) &
        (df_bx["Has_Linked_MRI"] == True)
    ).astype(int)

    df_bx["Is_Valid_csPCa_Trajectory"] = (
        (df_bx["Is_csPCa"] == 1) &
        (df_bx["Has_Valid_Physical_Trajectory"] == 1)
    ).astype(int)

    # 5. Check inference coverage
    master_df = pd.read_csv(UCLA_MASTER_CSV)
    master_df["Session_ID"] = master_df["Session_ID"].astype(str).str.strip()
    master_sids = set(master_df["Session_ID"])

    # Check existence of segresnet probmap and nnunet prediction
    has_seg_probmap = set()
    for sid in master_sids:
        if os.path.exists(os.path.join(SEG_PROBMAP_DIR, f"{sid}.npz")):
            has_seg_probmap.add(sid)

    has_nnu_pred = set()
    for sid in master_sids:
        if os.path.exists(os.path.join(NNU_PRED_DIR, f"UCLA_{sid}.npz")):
            has_nnu_pred.add(sid)

    df_bx["In_Standardized_Master"] = df_bx["Session_ID_Matched"].isin(master_sids).astype(int)
    df_bx["Has_SegResNet_Inference"] = df_bx["Session_ID_Matched"].isin(has_seg_probmap).astype(int)
    df_bx["Has_nnUNet_Inference"] = df_bx["Session_ID_Matched"].isin(has_nnu_pred).astype(int)

    # Export CORES_REFERENCE.csv
    cores_ref_cols = [
        "Original_Row_ID", "Patient Number", "MRI_UID", "Series Instance UID (US)",
        "Session_ID_Matched", "Core Label", "Core_Type", "Pathology_Category",
        "ISUP_Grade", "Is_csPCa", "Is_GG1_Indolent", "Is_Cancer_Missing_Gleason",
        "Is_Targeted_Core", "Is_Systematic_Core", "Coord_Completeness_AllFinite",
        "Coord_Sentinel_Neg1000", "Needle_Length_mm", "Has_Valid_Physical_Trajectory",
        "Is_Valid_csPCa_Trajectory", "Bx Base X (MRI Coord)", "Bx Base Y (MRI Coord)",
        "Bx Base Z (MRI Coord)", "Bx Tip X (MRI Coord)", "Bx Tip Y (MRI Coord)",
        "Bx Tip Z (MRI Coord)", "In_Standardized_Master", "Has_SegResNet_Inference",
        "Has_nnUNet_Inference"
    ]
    cores_ref_csv = os.path.join(AUDIT_DIR, "CORES_REFERENCE.csv")
    df_bx[cores_ref_cols].to_csv(cores_ref_csv, index=False)
    print(f"Saved CORES_REFERENCE.csv ({len(df_bx)} cores) -> {cores_ref_csv}")

    # 6. Parse MRI Examination Metadata & Canonical Selection Rule
    meta_df = pd.read_csv(UCLA_METADATA_CSV)
    meta_df.columns = [c.strip() for c in meta_df.columns]
    meta_df["StudyDate_dt"] = pd.to_datetime(meta_df["Study Date"], errors="coerce")
    uid_to_date = meta_df.groupby("Series UID")["StudyDate_dt"].min().to_dict()
    uid_to_subj = meta_df.groupby("Series UID")["Subject ID"].first().to_dict()

    # Link Master sessions to exact MRI series UID and dates
    # Session_ID is Patient_ID + "_" + Suffix5
    # Build dictionary from biopsy table for MRI UID
    sid_to_mri_uid = df_bx[df_bx["Has_Linked_MRI"]].groupby("Session_ID_Matched")["MRI_UID"].first().to_dict()

    master_df["MRI_Series_UID"] = master_df["Session_ID"].map(sid_to_mri_uid).fillna("")
    
    # Fill any missing MRI_Series_UID by searching metadata with Subject ID and suffix
    for idx, r in master_df.iterrows():
        if not r["MRI_Series_UID"]:
            pid = r["Patient_ID"]
            suf = r["Session_ID"].split("_")[-1]
            matches = meta_df[(meta_df["Subject ID"] == pid) & (meta_df["Series UID"].str.endswith(suf))]
            if not matches.empty:
                master_df.at[idx, "MRI_Series_UID"] = matches.iloc[0]["Series UID"]

    master_df["MRI_Study_Date_dt"] = master_df["MRI_Series_UID"].map(uid_to_date)
    master_df["MRI_Study_Date"] = master_df["MRI_Study_Date_dt"].dt.strftime("%Y-%m-%d").fillna("Missing_Date")
    master_df["Has_Valid_Date"] = master_df["MRI_Study_Date_dt"].notna().astype(int)
    master_df["Manifest_Row_Index"] = np.arange(len(master_df))

    # Stable deterministic canonical selection rule:
    # 1. Earliest MRI Study Date (pd.to_datetime)
    # 2. Tie-break: lowest Manifest_Row_Index
    # 3. Tie-break: Session_ID alphanumeric
    canon_sorted = master_df.sort_values(
        ["Patient_ID", "MRI_Study_Date_dt", "Manifest_Row_Index", "Session_ID"],
        na_position="last"
    )
    canon_first = canon_sorted.groupby("Patient_ID", as_index=False).first()
    canonical_sids = set(canon_first["Session_ID"])

    master_df["Is_Canonical_Patient_Session"] = master_df["Session_ID"].isin(canonical_sids).astype(int)
    master_df["Has_SegResNet_Inference"] = master_df["Session_ID"].isin(has_seg_probmap).astype(int)
    master_df["Has_nnUNet_Inference"] = master_df["Session_ID"].isin(has_nnu_pred).astype(int)
    master_df["Is_Paired_Complete_Inference"] = (
        (master_df["Has_SegResNet_Inference"] == 1) & (master_df["Has_nnUNet_Inference"] == 1)
    ).astype(int)

    # 7. Aggregate Clinical Reference per MRI Examination independently of coordinates
    # For every master session, compute:
    # - Total cores linked to this MRI
    # - Cores by type: targeted, systematic, other
    # - Pathology from ALL linked cores (independent of coordinate validity)
    # - csPCa status, highest ISUP grade, targeted csPCa hit, systematic csPCa hit
    # - Valid-trajectory csPCa status (cores with valid 3D physical coordinates)
    # - Sentinel-only csPCa flag
    
    sess_records = []
    reconcil_records = []

    # Target spreadsheet UCLA scores: Link by exact seriesInstanceUID_MR!
    df_tgt = pd.read_excel(UCLA_TARGET_XLSX).dropna(subset=["Patient ID"])
    df_tgt["seriesInstanceUID_MR"] = df_tgt["seriesInstanceUID_MR"].astype(str).str.strip()
    df_tgt["Patient ID"] = df_tgt["Patient ID"].astype(str).str.strip()
    
    # Exact exam-level max UCLA score
    exam_ucla_scores = df_tgt.groupby("seriesInstanceUID_MR")["UCLA Score (Similar to PIRADS v2)"].max().to_dict()
    # Patient-level max UCLA score (for audit comparison)
    pat_max_ucla_scores = df_tgt.groupby("Patient ID")["UCLA Score (Similar to PIRADS v2)"].max().to_dict()

    for idx, r in master_df.iterrows():
        sid = r["Session_ID"]
        pid = r["Patient_ID"]
        mri_uid = r["MRI_Series_UID"]
        is_canon = r["Is_Canonical_Patient_Session"]

        # ALL cores linked to this exact MRI examination (independent of coordinates!)
        mri_cores = df_bx[df_bx["Session_ID_Matched"] == sid]
        # Cores with valid trajectory
        mri_cores_valid = mri_cores[mri_cores["Has_Valid_Physical_Trajectory"] == 1]

        # ALL cores for this patient across all sessions (for patient-level reference)
        pat_cores = df_bx[df_bx["Patient Number"] == pid]
        pat_cores_valid = pat_cores[pat_cores["Has_Valid_Physical_Trajectory"] == 1]

        # Session-level metrics (MRI exam reference)
        n_cores_total = len(mri_cores)
        n_cores_targ = int((mri_cores["Is_Targeted_Core"] == 1).sum())
        n_cores_sys = int((mri_cores["Is_Systematic_Core"] == 1).sum())
        n_cores_other = int((mri_cores["Core_Type"] == "other_extra").sum())

        # Raw pathology independent of coords
        sess_is_cspca = int((mri_cores["Is_csPCa"] == 1).any())
        sess_max_isup = int(mri_cores["ISUP_Grade"].max()) if n_cores_total > 0 else 0
        sess_has_targ_cspca = int(((mri_cores["Is_Targeted_Core"] == 1) & (mri_cores["Is_csPCa"] == 1)).any())
        sess_has_sys_cspca = int(((mri_cores["Is_Systematic_Core"] == 1) & (mri_cores["Is_csPCa"] == 1)).any())
        
        # Valid physical trajectory metrics
        sess_has_valid_pos_core = int((mri_cores_valid["Is_csPCa"] == 1).any())
        sess_pos_cores_all_sentinel = int(sess_is_cspca == 1 and sess_has_valid_pos_core == 0)

        # Systematic-only csPCa definition:
        # STRICT: Has systematic csPCa AND has NO targeted csPCa
        # Distinguish:
        # (a) targeted cores performed and negative vs
        # (b) no targeted cores performed (targeted absent)
        if sess_is_cspca == 1:
            if sess_has_targ_cspca == 1:
                sys_only_category = "Targeted_Positive"
            elif n_cores_targ > 0:
                sys_only_category = "Systematic_Only_Targeted_Negative"
            else:
                sys_only_category = "Systematic_Only_Targeted_Absent"
        else:
            sys_only_category = "Non_csPCa"

        # UCLA Score linkage
        # Exact exam match
        exact_score = exam_ucla_scores.get(mri_uid, np.nan)
        pat_score = pat_max_ucla_scores.get(pid, np.nan)
        score_link_match = (exact_score == pat_score) if (pd.notna(exact_score) and pd.notna(pat_score)) else False

        # Old logic simulation (from bx_nonnull where Has_NonNull_Coords == 1)
        # In old code: bx_nonnull was used for s_all!
        # Check if any csPCa cores were missed by old logic:
        old_mri_cores = mri_cores[mri_cores["Coord_Completeness_AllFinite"] == 1]
        old_sess_is_cspca = int((old_mri_cores["Is_csPCa"] == 1).any())
        
        reconcil_records.append({
            "Session_ID": sid,
            "Patient_ID": pid,
            "MRI_Series_UID": mri_uid,
            "Is_Canonical": is_canon,
            "Old_Sess_Is_csPCa": old_sess_is_cspca,
            "New_Sess_Is_csPCa_Independent": sess_is_cspca,
            "Discrepancy_Sess_csPCa": int(old_sess_is_cspca != sess_is_cspca),
            "Sess_Max_ISUP": sess_max_isup,
            "Sess_Has_Valid_Pos_Core": sess_has_valid_pos_core,
            "Sess_Pos_Cores_All_Sentinel": sess_pos_cores_all_sentinel,
            "Total_Cores_Linked": n_cores_total,
            "Targeted_Cores_Linked": n_cores_targ,
            "Systematic_Cores_Linked": n_cores_sys,
            "Targeted_csPCa_Hit": sess_has_targ_cspca,
            "Systematic_csPCa_Hit": sess_has_sys_cspca,
            "Systematic_Only_Category": sys_only_category,
            "Exact_Exam_UCLA_Score": exact_score if pd.notna(exact_score) else -1,
            "Patient_Max_UCLA_Score": pat_score if pd.notna(pat_score) else -1,
            "UCLA_Score_Linkage_Discrepancy": int(exact_score != pat_score and pd.notna(exact_score) and pd.notna(pat_score)),
            "Has_SegResNet": r["Has_SegResNet_Inference"],
            "Has_nnUNet": r["Has_nnUNet_Inference"],
        })

    df_reconcil = pd.DataFrame(reconcil_records)
    reconcil_csv = os.path.join(AUDIT_DIR, "CLINICAL_LABEL_RECONCILIATION.csv")
    df_reconcil.to_csv(reconcil_csv, index=False)
    print(f"Saved CLINICAL_LABEL_RECONCILIATION.csv ({len(df_reconcil)} sessions) -> {reconcil_csv}")

    # Summary of label reconciliation
    discrep = df_reconcil["Discrepancy_Sess_csPCa"].sum()
    print(f"  -> csPCa label discrepancies between old and independent definition: {discrep}")
    ucla_discrep = df_reconcil["UCLA_Score_Linkage_Discrepancy"].sum()
    print(f"  -> UCLA score discrepancies between patient-max and exam-matched: {ucla_discrep}")

    # 8. Export ANALYSIS_MANIFEST.csv
    manifest_cols = [
        "Session_ID", "Patient_ID", "MRI_Series_UID", "MRI_Study_Date", "Has_Valid_Date",
        "Manifest_Row_Index", "Is_Canonical_Patient_Session", "Has_SegResNet_Inference",
        "Has_nnUNet_Inference", "Is_Paired_Complete_Inference"
    ]
    manifest_df = master_df[manifest_cols].merge(
        df_reconcil[[
            "Session_ID", "New_Sess_Is_csPCa_Independent", "Sess_Max_ISUP",
            "Sess_Has_Valid_Pos_Core", "Sess_Pos_Cores_All_Sentinel",
            "Total_Cores_Linked", "Targeted_Cores_Linked", "Systematic_Cores_Linked",
            "Targeted_csPCa_Hit", "Systematic_csPCa_Hit", "Systematic_Only_Category",
            "Exact_Exam_UCLA_Score", "Patient_Max_UCLA_Score"
        ]],
        on="Session_ID"
    )
    manifest_csv = os.path.join(AUDIT_DIR, "ANALYSIS_MANIFEST.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Saved ANALYSIS_MANIFEST.csv ({len(manifest_df)} sessions) -> {manifest_csv}")

    # Cohort counts breakdown
    canon_df = manifest_df[manifest_df["Is_Canonical_Patient_Session"] == 1]
    paired_df = canon_df[canon_df["Is_Paired_Complete_Inference"] == 1]
    print("\nCanonical 1-Session-Per-Patient Cohort Summary:")
    print(f"  Total canonical patients/sessions: {len(canon_df)}")
    print(f"  Paired with complete SegResNet & nnU-Net: {len(paired_df)}")
    print(f"  Canonical csPCa (independent): {canon_df['New_Sess_Is_csPCa_Independent'].sum()}")
    print(f"  Canonical csPCa valid trajectory: {canon_df['Sess_Has_Valid_Pos_Core'].sum()}")
    print(f"  Canonical csPCa all sentinel -1000: {canon_df['Sess_Pos_Cores_All_Sentinel'].sum()}")
    print(f"  Systematic-only csPCa breakdown in canonical:")
    print(canon_df["Systematic_Only_Category"].value_counts())

    # Synchronize manifest to submission source_data
    shutil.copy2(manifest_csv, os.path.join(SUBMISSION_SOURCE_DIR, "ANALYSIS_MANIFEST.csv"))
    generate_table1()


def generate_table1():
    print("\n" + "=" * 80)
    print("GENERATING TABLE 1: BASELINE DEMOGRAPHIC & CLINICAL CHARACTERISTICS")
    print("=" * 80)

    manifest = pd.read_csv(os.path.join(AUDIT_DIR, "ANALYSIS_MANIFEST.csv"))
    ucla_master = pd.read_csv(UCLA_MASTER_CSV)

    m = manifest.merge(
        ucla_master[["Session_ID", "PSA", "MRI_Prostate_Vol_cc"]],
        on="Session_ID",
        how="left"
    )
    m["PSA_Density"] = m["PSA"] / m["MRI_Prostate_Vol_cc"]

    df_canon = m[m["Is_Canonical_Patient_Session"] == 1].copy()
    df_paired = df_canon[df_canon["Is_Paired_Complete_Inference"] == 1].copy()

    rows = []

    def add_row(category, characteristic, c_canon_str, c_paired_str):
        rows.append({
            "Category": category,
            "Characteristic": characteristic,
            "Canonical_Cohort_N809": c_canon_str,
            "Paired_Cohort_N800": c_paired_str
        })

    add_row("Study Population", "Total unique patients / examinations, N", "809", "800")

    for label, col, unit in [
        ("Serum PSA (ng/mL)", "PSA", "ng/mL"),
        ("Prostate Gland Volume (cc)", "MRI_Prostate_Vol_cc", "cc"),
        ("PSA Density (ng/mL²)", "PSA_Density", "ng/mL²")
    ]:
        c_vals = df_canon[col].dropna()
        p_vals = df_paired[col].dropna()

        c_med = c_vals.median()
        c_q25, c_q75 = c_vals.quantile(0.25), c_vals.quantile(0.75)
        c_min, c_max = c_vals.min(), c_vals.max()
        c_n_valid = len(c_vals)

        p_med = p_vals.median()
        p_q25, p_q75 = p_vals.quantile(0.25), p_vals.quantile(0.75)
        p_min, p_max = p_vals.min(), p_vals.max()
        p_n_valid = len(p_vals)

        if col == "PSA_Density":
            c_str = f"{c_med:.3f} [{c_q25:.3f}–{c_q75:.3f}] (range {c_min:.3f}–{c_max:.3f})"
            p_str = f"{p_med:.3f} [{p_q25:.3f}–{p_q75:.3f}] (range {p_min:.3f}–{p_max:.3f})"
        else:
            c_str = f"{c_med:.1f} [{c_q25:.1f}–{c_q75:.1f}] (range {c_min:.1f}–{c_max:.1f})"
            p_str = f"{p_med:.1f} [{p_q25:.1f}–{p_q75:.1f}] (range {p_min:.1f}–{p_max:.1f})"

        add_row("Laboratory / Morphometry", f"{label}, Median [IQR]", c_str, p_str)
        add_row("Laboratory / Morphometry", f"{label}, Non-missing data available", f"{c_n_valid} ({(c_n_valid/len(df_canon)*100):.1f}%)", f"{p_n_valid} ({(p_n_valid/len(df_paired)*100):.1f}%)")

    s_canon = df_canon["Exact_Exam_UCLA_Score"].value_counts()
    s_paired = df_paired["Exact_Exam_UCLA_Score"].value_counts()

    c_low = s_canon.get(1, 0) + s_canon.get(2, 0)
    p_low = s_paired.get(1, 0) + s_paired.get(2, 0)
    add_row("MRI Suspicion (UCLA Score)", "MRI Low Suspicion (UCLA Scores 1–2), n (%)", f"{c_low} ({(c_low/len(df_canon)*100):.1f}%)", f"{p_low} ({(p_low/len(df_paired)*100):.1f}%)")

    c_0 = s_canon.get(0, 0)
    p_0 = s_paired.get(0, 0)
    add_row("MRI Suspicion (UCLA Score)", "MRI Score 0, n (%)", f"{c_0} ({(c_0/len(df_canon)*100):.1f}%)", f"{p_0} ({(p_0/len(df_paired)*100):.1f}%)")

    c_eq = s_canon.get(3, 0)
    p_eq = s_paired.get(3, 0)
    add_row("MRI Suspicion (UCLA Score)", "MRI Equivocal Suspicion (UCLA Score 3), n (%)", f"{c_eq} ({(c_eq/len(df_canon)*100):.1f}%)", f"{p_eq} ({(p_eq/len(df_paired)*100):.1f}%)")

    c_hi = s_canon.get(4, 0) + s_canon.get(5, 0)
    p_hi = s_paired.get(4, 0) + s_paired.get(5, 0)
    add_row("MRI Suspicion (UCLA Score)", "MRI High Suspicion (UCLA Scores 4–5), n (%)", f"{c_hi} ({(c_hi/len(df_canon)*100):.1f}%)", f"{p_hi} ({(p_hi/len(df_paired)*100):.1f}%)")

    c_miss = s_canon.get(-1, 0)
    p_miss = s_paired.get(-1, 0)
    add_row("MRI Suspicion (UCLA Score)", "MRI Score Missing / Unassigned, n (%)", f"{c_miss} ({(c_miss/len(df_canon)*100):.1f}%)", f"{p_miss} ({(p_miss/len(df_paired)*100):.1f}%)")

    i_canon = df_canon["Sess_Max_ISUP"].value_counts()
    i_paired = df_paired["Sess_Max_ISUP"].value_counts()

    c_ben = i_canon.get(0, 0)
    p_ben = i_paired.get(0, 0)
    add_row("Biopsy Histopathology (ISUP)", "Benign / Negative for Cancer, n (%)", f"{c_ben} ({(c_ben/len(df_canon)*100):.1f}%)", f"{p_ben} ({(p_ben/len(df_paired)*100):.1f}%)")

    c_gg1 = i_canon.get(1, 0)
    p_gg1 = i_paired.get(1, 0)
    add_row("Biopsy Histopathology (ISUP)", "ISUP Grade Group 1 (Gleason 3+3=6), n (%)", f"{c_gg1} ({(c_gg1/len(df_canon)*100):.1f}%)", f"{p_gg1} ({(p_gg1/len(df_paired)*100):.1f}%)")

    c_cspca = int(df_canon["New_Sess_Is_csPCa_Independent"].sum())
    p_cspca = int(df_paired["New_Sess_Is_csPCa_Independent"].sum())
    add_row("Biopsy Histopathology (ISUP)", "Clinically Significant PCa (csPCa, ISUP ≥ 2), n (%)", f"{c_cspca} ({(c_cspca/len(df_canon)*100):.1f}%)", f"{p_cspca} ({(p_cspca/len(df_paired)*100):.1f}%)")

    c_gg2 = i_canon.get(2, 0)
    p_gg2 = i_paired.get(2, 0)
    add_row("Biopsy Histopathology (ISUP)", "  ISUP Grade Group 2 (Gleason 3+4=7), n (%)", f"{c_gg2} ({(c_gg2/len(df_canon)*100):.1f}%)", f"{p_gg2} ({(p_gg2/len(df_paired)*100):.1f}%)")

    c_gg3 = i_canon.get(3, 0)
    p_gg3 = i_paired.get(3, 0)
    add_row("Biopsy Histopathology (ISUP)", "  ISUP Grade Group 3 (Gleason 4+3=7), n (%)", f"{c_gg3} ({(c_gg3/len(df_canon)*100):.1f}%)", f"{p_gg3} ({(p_gg3/len(df_paired)*100):.1f}%)")

    c_gg45 = i_canon.get(4, 0) + i_canon.get(5, 0)
    p_gg45 = i_paired.get(4, 0) + i_paired.get(5, 0)
    add_row("Biopsy Histopathology (ISUP)", "  ISUP Grade Group 4–5 (Gleason ≥ 8), n (%)", f"{c_gg45} ({(c_gg45/len(df_canon)*100):.1f}%)", f"{p_gg45} ({(p_gg45/len(df_paired)*100):.1f}%)")

    for col_c, name_c in [
        ("Total_Cores_Linked", "Total Biopsy Cores Sampled per Examination"),
        ("Targeted_Cores_Linked", "Recorded Targeted/Tracking Cores Sampled per Examination"),
        ("Systematic_Cores_Linked", "Systematic Template Cores Sampled per Examination")
    ]:
        c_v = df_canon[col_c]
        p_v = df_paired[col_c]
        c_str = f"{c_v.median():.1f} [{c_v.quantile(0.25):.1f}–{c_v.quantile(0.75):.1f}] (total = {int(c_v.sum())})"
        p_str = f"{p_v.median():.1f} [{p_v.quantile(0.25):.1f}–{p_v.quantile(0.75):.1f}] (total = {int(p_v.sum())})"
        add_row("Biopsy Core Burden", f"{name_c}, Median [IQR]", c_str, p_str)

    c_has_targ = int((df_canon["Targeted_Cores_Linked"] > 0).sum())
    p_has_targ = int((df_paired["Targeted_Cores_Linked"] > 0).sum())
    add_row("Biopsy Core Burden", "Examinations with ≥ 1 Targeted Core Linked, n (%)", f"{c_has_targ} ({(c_has_targ/len(df_canon)*100):.1f}%)", f"{p_has_targ} ({(p_has_targ/len(df_paired)*100):.1f}%)")

    c_has_sys = int((df_canon["Systematic_Cores_Linked"] > 0).sum())
    p_has_sys = int((df_paired["Systematic_Cores_Linked"] > 0).sum())
    add_row("Biopsy Core Burden", "Examinations with ≥ 1 Systematic Core Linked, n (%)", f"{c_has_sys} ({(c_has_sys/len(df_canon)*100):.1f}%)", f"{p_has_sys} ({(p_has_sys/len(df_paired)*100):.1f}%)")

    cores_df = pd.read_csv(os.path.join(AUDIT_DIR, "CORES_REFERENCE.csv"))
    canon_sids = set(df_canon["Session_ID"])
    paired_sids = set(df_paired["Session_ID"])

    c_cores = cores_df[cores_df["Session_ID_Matched"].isin(canon_sids)]
    p_cores = cores_df[cores_df["Session_ID_Matched"].isin(paired_sids)]

    c_val_cores = int(c_cores["Has_Valid_Physical_Trajectory"].sum())
    p_val_cores = int(p_cores["Has_Valid_Physical_Trajectory"].sum())
    add_row("Biopsy Core Burden", "Total Linked Cores with Valid 3D Trajectory, n (%)", f"{c_val_cores} ({(c_val_cores/len(c_cores)*100):.1f}%)", f"{p_val_cores} ({(p_val_cores/len(p_cores)*100):.1f}%)")

    valid_core_sids = set(cores_df[cores_df["Has_Valid_Physical_Trajectory"] == 1]["Session_ID_Matched"])
    c_val_all = len(canon_sids.intersection(valid_core_sids))
    p_val_all = len(paired_sids.intersection(valid_core_sids))
    add_row("Spatial Evaluation Eligibility", "All Examinations with ≥ 1 Valid 3D Trajectory Core, n (%)", f"{c_val_all} ({(c_val_all/len(df_canon)*100):.1f}%)", f"{p_val_all} ({(p_val_all/len(df_paired)*100):.1f}%)")

    c_val_cs = int(df_canon[df_canon["New_Sess_Is_csPCa_Independent"] == 1]["Sess_Has_Valid_Pos_Core"].sum())
    p_val_cs = int(df_paired[df_paired["New_Sess_Is_csPCa_Independent"] == 1]["Sess_Has_Valid_Pos_Core"].sum())
    add_row("Spatial Evaluation Eligibility", "csPCa Examinations with Valid 3D Positive Trajectory, n (%)", f"{c_val_cs} ({(c_val_cs/c_cspca*100):.1f}%)", f"{p_val_cs} ({(p_val_cs/p_cspca*100):.1f}%)")

    df_t1 = pd.DataFrame(rows)
    out_csv = os.path.join(AUDIT_DIR, "TABLE1_BASELINE_CHARACTERISTICS.csv")
    df_t1.to_csv(out_csv, index=False)
    shutil.copy2(out_csv, os.path.join(SUBMISSION_SOURCE_DIR, "TABLE1_BASELINE_CHARACTERISTICS.csv"))
    print(f"Saved Table 1 -> {out_csv} ({len(df_t1)} rows)")


if __name__ == "__main__":
    build_clinical_reference_and_manifest()

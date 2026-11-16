import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

PROD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DIR = os.path.join(PROD_DIR, "results", "tables")
SOURCES_DIR = os.path.join(PROD_DIR, "sources")
BACKUP_DIR = os.path.join(AUDIT_DIR, "original_tables_backup")
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(SOURCES_DIR, exist_ok=True)

CANDIDATES_CSV = os.path.join(AUDIT_DIR, "CANDIDATES_ANALYSIS_READY.csv")
MANIFEST_CSV = os.path.join(AUDIT_DIR, "ANALYSIS_MANIFEST.csv")
CORES_CSV = os.path.join(AUDIT_DIR, "CORES_REFERENCE.csv")
MASTER_CSV = (
    os.path.join(SOURCES_DIR, "Phase3_Patient_Session_Level_Master.csv")
    if os.path.exists(os.path.join(SOURCES_DIR, "Phase3_Patient_Session_Level_Master.csv"))
    else os.path.join(BACKUP_DIR, "Phase3_Patient_Session_Level_Master.csv")
)

N_BOOT = 2000
N_MC_NULL = 10000
RNG_SEED = 42

def wilson_score_interval(hits, total, conf=0.95):
    """Calculates exact Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0
    p = hits / total
    z = 1.959963984540054
    denom = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * total)) / total)) / denom
    ci_low = max(0.0, center - half) * 100.0
    ci_high = min(1.0, center + half) * 100.0
    return round(ci_low, 2), round(ci_high, 2)

def fmt_ci(hits, total):
    low, high = wilson_score_interval(hits, total)
    return f"[{low:.2f}, {high:.2f}]"

def run_burden_matched_and_statistical_analysis():
    t0 = time.time()
    print("=" * 90)
    print("COMPREHENSIVE STATISTICAL ANALYSIS & CLINICAL AUDIT PIPELINE (V3.1 VERIFIED)")
    print("=" * 90)

    if not os.path.exists(CANDIDATES_CSV):
        raise FileNotFoundError(f"Missing {CANDIDATES_CSV}")
    if not os.path.exists(MANIFEST_CSV):
        raise FileNotFoundError(f"Missing {MANIFEST_CSV}")
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(f"Missing {MASTER_CSV}")

    # 1. Load Data
    cands_df = pd.read_csv(CANDIDATES_CSV)
    manifest_df = pd.read_csv(MANIFEST_CSV)
    cores_df = pd.read_csv(CORES_CSV)
    master_df = pd.read_csv(MASTER_CSV)

    meta_cols = [
        "Session_ID", "Prostate_Mask_Vol_cc", "Human_Rad_Mask_Vol_cc",
        "Human_Rad_Mask_Num_Components", "Human_Sess_CoM_Dist_csPCa_mm",
        "Human_Sess_Surface3D_Dist_csPCa_mm", "Human_Sess_DirectPuncture_0mm_csPCa",
        "q_undil_sess_2.5", "q_undil_sess_5.0", "q_undil_sess_7.5", "q_undil_sess_10.0"
    ]
    df_merged = manifest_df.merge(master_df[meta_cols], on="Session_ID", how="left")

    df_canon = df_merged[df_merged["Is_Canonical_Patient_Session"] == 1].copy().reset_index(drop=True)
    df_paired = df_canon[df_canon["Is_Paired_Complete_Inference"] == 1].copy().reset_index(drop=True)

    n_canon = len(df_canon)
    n_paired = len(df_paired)
    print(f"Canonical Cohort: N = {n_canon} unique patients/examinations")
    print(f"Paired Complete-Inference Cohort: N = {n_paired} unique patients/examinations")

    canon_cs_all = df_canon[df_canon["New_Sess_Is_csPCa_Independent"] == 1]
    canon_cs_val = canon_cs_all[canon_cs_all["Sess_Has_Valid_Pos_Core"] == 1]
    paired_cs_all = df_paired[df_paired["New_Sess_Is_csPCa_Independent"] == 1]
    paired_cs_val = df_paired[(df_paired["New_Sess_Is_csPCa_Independent"] == 1) & (df_paired["Sess_Has_Valid_Pos_Core"] == 1)]

    print(f"Canonical csPCa: Total = {len(canon_cs_all)}, Trajectory-Evaluable = {len(canon_cs_val)}, Sentinel-Only = {len(canon_cs_all)-len(canon_cs_val)}")
    print(f"Paired csPCa: Total = {len(paired_cs_all)}, Trajectory-Evaluable = {len(paired_cs_val)}, Sentinel-Only = {len(paired_cs_all)-len(paired_cs_val)}")

    paired_sids = set(df_paired["Session_ID"])
    cands_paired = cands_df[cands_df["Session_ID"].isin(paired_sids)].copy()

    models = ["3D_SegResNet", "Pretrained_nnUNet"]
    extractors = ["Official_RGA_Dynamic", "Shared_Fixed_CC_0.10", "Custom_Dynamic_Cube"]

    # ==============================================================================
    # 1. EXAMINATION-LEVEL DISCRIMINATION (ROC-AUC & AVERAGE PRECISION)
    # ==============================================================================
    print("\n" + "-" * 75)
    print("ANALYSIS 1: EXAMINATION-LEVEL DISCRIMINATION (ROC-AUC & AVERAGE PRECISION)")
    print("-" * 75)

    discrim_rows = []
    y_true = df_paired["New_Sess_Is_csPCa_Independent"].values.astype(int)

    rng_disc = np.random.default_rng(RNG_SEED)
    boot_indices_disc = rng_disc.integers(0, n_paired, size=(N_BOOT, n_paired))

    for ext in extractors:
        ext_cands = cands_paired[cands_paired["Extractor"] == ext]
        seg_cands = ext_cands[ext_cands["Model"] == "3D_SegResNet"]
        nnu_cands = ext_cands[ext_cands["Model"] == "Pretrained_nnUNet"]

        map_seg = seg_cands.groupby("Session_ID")["Max_Prob"].max().to_dict()
        map_nnu = nnu_cands.groupby("Session_ID")["Max_Prob"].max().to_dict()

        y_score_seg = np.array([map_seg.get(sid, 0.0) for sid in df_paired["Session_ID"]], dtype=np.float32)
        y_score_nnu = np.array([map_nnu.get(sid, 0.0) for sid in df_paired["Session_ID"]], dtype=np.float32)

        auc_seg = roc_auc_score(y_true, y_score_seg)
        auc_nnu = roc_auc_score(y_true, y_score_nnu)
        ap_seg = average_precision_score(y_true, y_score_seg)
        ap_nnu = average_precision_score(y_true, y_score_nnu)

        delta_auc = auc_seg - auc_nnu
        delta_ap = ap_seg - ap_nnu

        boot_aucs_seg, boot_aucs_nnu = [], []
        boot_aps_seg, boot_aps_nnu = [], []
        boot_delta_aucs, boot_delta_aps = [], []

        for b in range(N_BOOT):
            idx = boot_indices_disc[b]
            yt_b = y_true[idx]
            if len(np.unique(yt_b)) < 2:
                continue
            as_b = roc_auc_score(yt_b, y_score_seg[idx])
            an_b = roc_auc_score(yt_b, y_score_nnu[idx])
            ps_b = average_precision_score(yt_b, y_score_seg[idx])
            pn_b = average_precision_score(yt_b, y_score_nnu[idx])

            boot_aucs_seg.append(as_b)
            boot_aucs_nnu.append(an_b)
            boot_aps_seg.append(ps_b)
            boot_aps_nnu.append(pn_b)
            boot_delta_aucs.append(as_b - an_b)
            boot_delta_aps.append(ps_b - pn_b)

        ci_seg_auc = (np.percentile(boot_aucs_seg, 2.5), np.percentile(boot_aucs_seg, 97.5))
        ci_nnu_auc = (np.percentile(boot_aucs_nnu, 2.5), np.percentile(boot_aucs_nnu, 97.5))
        ci_seg_ap = (np.percentile(boot_aps_seg, 2.5), np.percentile(boot_aps_seg, 97.5))
        ci_nnu_ap = (np.percentile(boot_aps_nnu, 2.5), np.percentile(boot_aps_nnu, 97.5))

        ci_delta_auc = (np.percentile(boot_delta_aucs, 2.5), np.percentile(boot_delta_aucs, 97.5))
        ci_delta_ap = (np.percentile(boot_delta_aps, 2.5), np.percentile(boot_delta_aps, 97.5))

        p_val_auc = float(2.0 * min(np.mean(np.array(boot_delta_aucs) <= 0), np.mean(np.array(boot_delta_aucs) >= 0)))
        p_val_ap = float(2.0 * min(np.mean(np.array(boot_delta_aps) <= 0), np.mean(np.array(boot_delta_aps) >= 0)))

        discrim_rows.append({
            "Extractor": ext,
            "Model": "3D_SegResNet",
            "N_Cohort": n_paired,
            "N_csPCa": len(paired_cs_all),
            "ROC_AUC": round(auc_seg, 4),
            "ROC_AUC_95CI": f"[{ci_seg_auc[0]:.4f}, {ci_seg_auc[1]:.4f}]",
            "Average_Precision_AP": round(ap_seg, 4),
            "Average_Precision_95CI": f"[{ci_seg_ap[0]:.4f}, {ci_seg_ap[1]:.4f}]",
            "Delta_ROC_AUC": round(delta_auc, 4),
            "Delta_ROC_AUC_95CI": f"[{ci_delta_auc[0]:.4f}, {ci_delta_auc[1]:.4f}]",
            "Delta_ROC_AUC_P_Value": round(p_val_auc, 4),
            "Delta_AP": round(delta_ap, 4),
            "Delta_AP_95CI": f"[{ci_delta_ap[0]:.4f}, {ci_delta_ap[1]:.4f}]",
            "Delta_AP_P_Value": round(p_val_ap, 4),
        })

        discrim_rows.append({
            "Extractor": ext,
            "Model": "Pretrained_nnUNet",
            "N_Cohort": n_paired,
            "N_csPCa": len(paired_cs_all),
            "ROC_AUC": round(auc_nnu, 4),
            "ROC_AUC_95CI": f"[{ci_nnu_auc[0]:.4f}, {ci_nnu_auc[1]:.4f}]",
            "Average_Precision_AP": round(ap_nnu, 4),
            "Average_Precision_95CI": f"[{ci_nnu_ap[0]:.4f}, {ci_nnu_ap[1]:.4f}]",
            "Delta_ROC_AUC": round(-delta_auc, 4),
            "Delta_ROC_AUC_95CI": f"[{-ci_delta_auc[1]:.4f}, {-ci_delta_auc[0]:.4f}]",
            "Delta_ROC_AUC_P_Value": round(p_val_auc, 4),
            "Delta_AP": round(-delta_ap, 4),
            "Delta_AP_95CI": f"[{-ci_delta_ap[1]:.4f}, {-ci_delta_ap[0]:.4f}]",
            "Delta_AP_P_Value": round(p_val_ap, 4),
        })

        print(f"[{ext}] Seg: AUROC = {auc_seg:.4f} {discrim_rows[-2]['ROC_AUC_95CI']}, AP = {ap_seg:.4f} {discrim_rows[-2]['Average_Precision_95CI']}")
        print(f"[{ext}] nnU: AUROC = {auc_nnu:.4f} {discrim_rows[-1]['ROC_AUC_95CI']}, AP = {ap_nnu:.4f} {discrim_rows[-1]['Average_Precision_95CI']}")
        print(f"[{ext}] Delta AUROC = {delta_auc:+.4f} [{ci_delta_auc[0]:.4f}, {ci_delta_auc[1]:.4f}] (p = {p_val_auc:.4f})")

    df_discrim = pd.DataFrame(discrim_rows)
    df_discrim.to_csv(os.path.join(AUDIT_DIR, "EXAMINATION_DISCRIMINATION_RESULTS.csv"), index=False)

    # Export examination IDs, labels, and scores for external audit
    df_scores = pd.DataFrame({
        "Session_ID": df_paired["Session_ID"],
        "Patient_ID": df_paired["Patient_ID"],
        "Ground_Truth_csPCa": y_true,
    })
    for ext in extractors:
        ext_cands = cands_paired[cands_paired["Extractor"] == ext]
        for mod in models:
            mod_cands = ext_cands[ext_cands["Model"] == mod]
            max_map = mod_cands.groupby("Session_ID")["Max_Prob"].max().to_dict()
            col_name = f"{mod}_{ext}_Score"
            df_scores[col_name] = [round(max_map.get(sid, 0.0), 4) for sid in df_paired["Session_ID"]]
    df_scores.to_csv(os.path.join(AUDIT_DIR, "EXAMINATION_DISCRIMINATION_PATIENT_SCORES.csv"), index=False)
    print(f"Saved EXAMINATION_DISCRIMINATION_PATIENT_SCORES.csv ({len(df_scores)} rows)")

    # ==============================================================================
    # 2. STRICT PHYSICAL OCCUPANCY-BASED SPATIAL NULL BENCHMARK
    # ==============================================================================
    print("\n" + "-" * 75)
    print("ANALYSIS 2: PHYSICAL OCCUPANCY-BASED SPATIAL NULL BENCHMARK")
    print("-" * 75)

    tolerances = [2.5, 5.0, 7.5, 10.0]

    seg_rga_cands = cands_paired[(cands_paired["Extractor"] == "Official_RGA_Dynamic") & (cands_paired["Model"] == "3D_SegResNet")]
    m_top1_map = seg_rga_cands[seg_rga_cands["Candidate_Rank"] == 1].groupby("Session_ID").size().to_dict()
    m_top3_map = seg_rga_cands[seg_rga_cands["Candidate_Rank"] <= 3].groupby("Session_ID").size().to_dict()
    sort_cols_b = ["Max_Prob", "Candidate_Rank", "Volume_cc"]
    asc_order_b = [False, True, False]
    if "Candidate_UID" in seg_rga_cands.columns:
        sort_cols_b.append("Candidate_UID")
        asc_order_b.append(True)
    top800_cands = seg_rga_cands.sort_values(by=sort_cols_b, ascending=asc_order_b).head(n_paired)
    m_b100_map = top800_cands.groupby("Session_ID").size().to_dict()

    # Per-exam table
    eval_cs_df = paired_cs_val.copy()
    eval_cs_df["m_Top1"] = [m_top1_map.get(sid, 0) for sid in eval_cs_df["Session_ID"]]
    eval_cs_df["m_Top3"] = [m_top3_map.get(sid, 0) for sid in eval_cs_df["Session_ID"]]
    eval_cs_df["m_B100"] = [m_b100_map.get(sid, 0) for sid in eval_cs_df["Session_ID"]]

    null_exam_records = []
    for _, r in eval_cs_df.iterrows():
        null_exam_records.append({
            "Session_ID": r["Session_ID"],
            "Patient_ID": r["Patient_ID"],
            "Prostate_Mask_Vol_cc": round(r["Prostate_Mask_Vol_cc"], 2),
            "q_2.5mm": round(r["q_undil_sess_2.5"], 4),
            "q_5.0mm": round(r["q_undil_sess_5.0"], 4),
            "q_7.5mm": round(r["q_undil_sess_7.5"], 4),
            "q_10.0mm": round(r["q_undil_sess_10.0"], 4),
            "Candidate_Allocation_Top1": r["m_Top1"],
            "Candidate_Allocation_Top3": r["m_Top3"],
            "Candidate_Allocation_B100": r["m_B100"],
            "Analytic_Null_Prob_10mm_Top1": round(1.0 - (1.0 - r["q_undil_sess_10.0"])**r["m_Top1"], 4),
            "Analytic_Null_Prob_10mm_Top3": round(1.0 - (1.0 - r["q_undil_sess_10.0"])**r["m_Top3"], 4),
            "Analytic_Null_Prob_10mm_B100": round(1.0 - (1.0 - r["q_undil_sess_10.0"])**r["m_B100"], 4),
        })
    df_null_exam = pd.DataFrame(null_exam_records)
    df_null_exam.to_csv(os.path.join(AUDIT_DIR, "PER_EXAM_NULL_OCCUPANCY_AND_ALLOCATION.csv"), index=False)
    print(f"Saved PER_EXAM_NULL_OCCUPANCY_AND_ALLOCATION.csv ({len(df_null_exam)} sessions)")

    null_summary_rows = []
    rng_mc = np.random.default_rng(RNG_SEED)
    n_cs_val = len(paired_cs_val)

    alloc_configs = [
        ("Top-1 (m_i <= 1)", eval_cs_df["m_Top1"].values),
        ("Top-3 (m_i <= 3)", eval_cs_df["m_Top3"].values),
        ("Matched Burden B=1.00", eval_cs_df["m_B100"].values),
    ]

    for tol in tolerances:
        q_vals = eval_cs_df[f"q_undil_sess_{tol}"].values
        mean_q = float(np.mean(q_vals)) * 100.0

        for alloc_label, m_vec in alloc_configs:
            p_null_per_exam = 1.0 - np.power(1.0 - q_vals, m_vec)
            exp_null_sens = float(np.mean(p_null_per_exam)) * 100.0

            # Observed SegResNet sensitivity
            if "Top-1" in alloc_label:
                obs_hits = seg_rga_cands[(seg_rga_cands["Candidate_Rank"] == 1) & (seg_rga_cands["CoM_Dist_csPCa_mm"] <= tol)]["Session_ID"].nunique()
            elif "Top-3" in alloc_label:
                obs_hits = seg_rga_cands[(seg_rga_cands["Candidate_Rank"] <= 3) & (seg_rga_cands["CoM_Dist_csPCa_mm"] <= tol)]["Session_ID"].nunique()
            else:
                obs_hits = top800_cands[top800_cands["CoM_Dist_csPCa_mm"] <= tol]["Session_ID"].nunique()

            obs_sens = (obs_hits / n_cs_val) * 100.0
            excess_sens = obs_sens - exp_null_sens

            # Monte Carlo simulation
            sim_draws = rng_mc.uniform(0.0, 1.0, size=(N_MC_NULL, n_cs_val))
            sim_hits_per_rep = np.sum(sim_draws < p_null_per_exam[None, :], axis=1)
            exceedances = int(np.sum(sim_hits_per_rep >= obs_hits))
            p_mc = (exceedances + 1.0) / (N_MC_NULL + 1.0)

            null_summary_rows.append({
                "Tolerance_mm": tol,
                "Candidate_Allocation": alloc_label,
                "N_Evaluated_csPCa": n_cs_val,
                "Mean_Prostate_Voxel_Occupancy_Pct": round(mean_q, 2),
                "Expected_Analytic_Null_Sens_Pct": round(exp_null_sens, 2),
                "Observed_SegResNet_Hits": obs_hits,
                "Observed_SegResNet_Sens_Pct": round(obs_sens, 2),
                "Excess_Sens_Over_Null_Pct": round(excess_sens, 2),
                "Monte_Carlo_Replicates": N_MC_NULL,
                "RNG_Seed": RNG_SEED,
                "Monte_Carlo_Exceedance_Count": exceedances,
                "Monte_Carlo_Empirical_P_Value": p_mc,
                "P_Value_Formatted": f"{p_mc:.4f}" if p_mc >= 0.001 else "< 0.0001",
            })
            print(f"Tol={tol:4.1f}mm [{alloc_label}]: Mean q={mean_q:.2f}% | Exp Null={exp_null_sens:.2f}% | Obs={obs_sens:.2f}% (Excess: {excess_sens:+.2f}%, Exceedances: {exceedances}/{N_MC_NULL}, p={null_summary_rows[-1]['P_Value_Formatted']})")

    df_null_summary = pd.DataFrame(null_summary_rows)
    df_null_summary.to_csv(os.path.join(AUDIT_DIR, "NULL_MODEL_RESULTS.csv"), index=False)

    # ==============================================================================
    # 3. ACHIEVED BURDEN-MATCHED SPATIAL CORRESPONDENCE (EXACT INSTANCE-LEVEL BOOTSTRAP)
    # ==============================================================================
    print("\n" + "-" * 75)
    print("ANALYSIS 3: ACHIEVED BURDEN-MATCHED SPATIAL CORRESPONDENCE (INSTANCE-LEVEL BOOTSTRAP)")
    print("-" * 75)

    burdens = [0.50, 0.75, 1.00, 1.50]
    bm_rows = []
    primary_tol = 10.0

    sid_to_idx = {sid: i for i, sid in enumerate(df_paired["Session_ID"])}
    cs_val_int_mask = np.array([1 if sid in set(paired_cs_val["Session_ID"]) else 0 for sid in df_paired["Session_ID"]], dtype=int)
    n_cs_val_denom = int(np.sum(cs_val_int_mask))

    def get_tier(isup):
        if isup >= 2:
            return "csPCa"
        elif isup == 1:
            return "ISUP_1"
        else:
            return "Benign"

    exam_tier_by_idx = np.array([get_tier(row["Sess_Max_ISUP"]) for _, row in df_paired.iterrows()])

    rng_bm = np.random.default_rng(RNG_SEED)
    boot_indices_bm = rng_bm.integers(0, n_paired, size=(N_BOOT, n_paired))

    for ext in extractors:
        ext_cands = cands_paired[cands_paired["Extractor"] == ext]

        seg_cands_sub = ext_cands[ext_cands["Model"] == "3D_SegResNet"]
        nnu_cands_sub = ext_cands[ext_cands["Model"] == "Pretrained_nnUNet"]

        # Deterministic sorting (Max_Prob DESC, Candidate_Rank ASC, Volume_cc DESC)
        sort_cols = ["Max_Prob", "Candidate_Rank", "Volume_cc"]
        asc_order = [False, True, False]
        if "Candidate_UID" in ext_cands.columns:
            sort_cols.append("Candidate_UID")
            asc_order.append(True)

        seg_cands_sub = seg_cands_sub.sort_values(by=sort_cols, ascending=asc_order).reset_index(drop=True)
        nnu_cands_sub = nnu_cands_sub.sort_values(by=sort_cols, ascending=asc_order).reset_index(drop=True)

        seg_probs = seg_cands_sub["Max_Prob"].values.astype(np.float32)
        seg_dists = seg_cands_sub["CoM_Dist_csPCa_mm"].fillna(999.0).values.astype(np.float32)
        seg_s_idx = np.array([sid_to_idx[sid] for sid in seg_cands_sub["Session_ID"]], dtype=int)
        seg_gg1 = seg_cands_sub["CoM_Dist_GG1_mm"].fillna(999.0).values.astype(np.float32)
        seg_ben = seg_cands_sub["CoM_Dist_Benign_mm"].fillna(999.0).values.astype(np.float32)

        nnu_probs = nnu_cands_sub["Max_Prob"].values.astype(np.float32)
        nnu_dists = nnu_cands_sub["CoM_Dist_csPCa_mm"].fillna(999.0).values.astype(np.float32)
        nnu_s_idx = np.array([sid_to_idx[sid] for sid in nnu_cands_sub["Session_ID"]], dtype=int)
        nnu_gg1 = nnu_cands_sub["CoM_Dist_GG1_mm"].fillna(999.0).values.astype(np.float32)
        nnu_ben = nnu_cands_sub["CoM_Dist_Benign_mm"].fillna(999.0).values.astype(np.float32)

        # Pre-group candidates by session for lightning-fast exact instance replication
        seg_cands_by_s = [[] for _ in range(n_paired)]
        for c_idx, s in enumerate(seg_s_idx):
            seg_cands_by_s[s].append(c_idx)
        seg_cands_by_s = [np.array(c, dtype=int) for c in seg_cands_by_s]
        seg_c_counts = np.array([len(c) for c in seg_cands_by_s], dtype=int)

        nnu_cands_by_s = [[] for _ in range(n_paired)]
        for c_idx, s in enumerate(nnu_s_idx):
            nnu_cands_by_s[s].append(c_idx)
        nnu_cands_by_s = [np.array(c, dtype=int) for c in nnu_cands_by_s]
        nnu_c_counts = np.array([len(c) for c in nnu_cands_by_s], dtype=int)

        def fast_eval_cands(s_idx_arr, probs, dists, gg1, ben, K):
            n_tot = len(probs)
            k_act = min(K, n_tot)
            if k_act == 0:
                return 0.0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0

            # Deterministic selection: arrays are already pre-sorted in descending priority order
            top_k_i = np.arange(k_act)
            eff_thresh = float(probs[k_act - 1]) if k_act > 0 else 0.0

            sel_s = s_idx_arr[top_k_i]
            sel_d = dists[top_k_i]
            sel_gg = gg1[top_k_i]
            sel_bn = ben[top_k_i]

            hit_mask = (sel_d <= primary_tol)
            hit_sids = np.unique(sel_s[hit_mask])
            n_hits = int(np.sum(cs_val_int_mask[hit_sids]))
            sens = (n_hits / n_cs_val_denom * 100.0) if n_cs_val_denom > 0 else 0.0

            # True allocation by examination pathology (mutually exclusive across patients, summing exactly to k_act)
            sel_tiers = exam_tier_by_idx[sel_s]
            n_exam_cs = int(np.sum(sel_tiers == "csPCa"))
            n_exam_gg = int(np.sum(sel_tiers == "ISUP_1"))
            n_exam_bn = int(np.sum(sel_tiers == "Benign"))

            # Proximity counts to core types
            n_cs_c = int(np.sum(hit_mask))
            n_gg_c = int(np.sum(sel_gg <= primary_tol))
            n_bn_c = int(np.sum(sel_bn <= primary_tol))
            return sens, n_hits, n_exam_cs, n_exam_gg, n_exam_bn, n_cs_c, n_gg_c, n_bn_c, eff_thresh, k_act

        for B in burdens:
            K_req = int(np.round(B * n_paired))

            # Point estimates
            seg_s_val, seg_h_val, seg_exam_cs, seg_exam_gg, seg_exam_bn, seg_cs_c, seg_gg1_c, seg_ben_c, seg_thresh, seg_k_act = fast_eval_cands(seg_s_idx, seg_probs, seg_dists, seg_gg1, seg_ben, K_req)
            nnu_s_val, nnu_h_val, nnu_exam_cs, nnu_exam_gg, nnu_exam_bn, nnu_cs_c, nnu_gg1_c, nnu_ben_c, nnu_thresh, nnu_k_act = fast_eval_cands(nnu_s_idx, nnu_probs, nnu_dists, nnu_gg1, nnu_ben, K_req)

            seg_b_achieved = round(seg_k_act / n_paired, 3)
            nnu_b_achieved = round(nnu_k_act / n_paired, 3)

            is_strictly_matched = (seg_k_act == K_req and nnu_k_act == K_req)
            if is_strictly_matched:
                feas_status = "Fully_Matched"
            elif nnu_k_act < K_req:
                feas_status = "nnUNet_Pool_Exhausted_Capped"
            else:
                feas_status = "SegResNet_Pool_Exhausted_Capped"

            # 2,000 Paired Cluster Bootstrap with explicit instance tracking
            boot_seg = np.zeros(N_BOOT, dtype=np.float32)
            boot_nnu = np.zeros(N_BOOT, dtype=np.float32)
            boot_deltas = np.zeros(N_BOOT, dtype=np.float32)
            boot_feas = np.zeros(N_BOOT, dtype=np.int32)

            for b in range(N_BOOT):
                b_draw = boot_indices_bm[b]
                eval_insts = np.where(cs_val_int_mask[b_draw])[0]
                n_eval_insts = len(eval_insts)

                # 1. Resample SegResNet candidate occurrences per instance
                seg_counts_b = seg_c_counts[b_draw]
                tot_seg_b = int(np.sum(seg_counts_b))
                k_seg = min(K_req, tot_seg_b)

                if k_seg > 0:
                    inst_ids_seg = np.repeat(np.arange(n_paired), seg_counts_b)
                    cand_idx_seg = np.concatenate([seg_cands_by_s[s] for s in b_draw])
                    if k_seg < tot_seg_b:
                        top_seg_i = np.argsort(-seg_probs[cand_idx_seg], kind="mergesort")[:k_seg]
                    else:
                        top_seg_i = np.arange(tot_seg_b)

                    hit_mask_seg = (seg_dists[cand_idx_seg][top_seg_i] <= primary_tol)
                    hit_inst_seg = np.unique(inst_ids_seg[top_seg_i][hit_mask_seg])
                    n_hits_seg = int(np.isin(eval_insts, hit_inst_seg).sum())
                    b_sens_seg = (n_hits_seg / n_eval_insts * 100.0) if n_eval_insts > 0 else 0.0
                else:
                    b_sens_seg = 0.0

                # 2. Resample nnUNet candidate occurrences per instance
                nnu_counts_b = nnu_c_counts[b_draw]
                tot_nnu_b = int(np.sum(nnu_counts_b))
                k_nnu = min(K_req, tot_nnu_b)

                if k_nnu > 0:
                    inst_ids_nnu = np.repeat(np.arange(n_paired), nnu_counts_b)
                    cand_idx_nnu = np.concatenate([nnu_cands_by_s[s] for s in b_draw])
                    if k_nnu < tot_nnu_b:
                        top_nnu_i = np.argsort(-nnu_probs[cand_idx_nnu], kind="mergesort")[:k_nnu]
                    else:
                        top_nnu_i = np.arange(tot_nnu_b)

                    hit_mask_nnu = (nnu_dists[cand_idx_nnu][top_nnu_i] <= primary_tol)
                    hit_inst_nnu = np.unique(inst_ids_nnu[top_nnu_i][hit_mask_nnu])
                    n_hits_nnu = int(np.isin(eval_insts, hit_inst_nnu).sum())
                    b_sens_nnu = (n_hits_nnu / n_eval_insts * 100.0) if n_eval_insts > 0 else 0.0
                else:
                    b_sens_nnu = 0.0

                boot_seg[b] = b_sens_seg
                boot_nnu[b] = b_sens_nnu
                boot_deltas[b] = b_sens_seg - b_sens_nnu
                boot_feas[b] = int(tot_seg_b >= K_req and tot_nnu_b >= K_req)

            ci_seg = (np.percentile(boot_seg, 2.5), np.percentile(boot_seg, 97.5))
            ci_nnu = (np.percentile(boot_nnu, 2.5), np.percentile(boot_nnu, 97.5))
            ci_del = (np.percentile(boot_deltas, 2.5), np.percentile(boot_deltas, 97.5))
            feas_pass_rate = float(np.mean(boot_feas))

            p_val = 2.0 * min(np.mean(boot_deltas <= 0.0), np.mean(boot_deltas >= 0.0))
            p_val = min(1.0, max(1.0 / (N_BOOT + 1), p_val))

            bm_rows.append({
                "Extractor": ext,
                "Nominal_Burden_B": B,
                "Requested_Budget_K": K_req,
                "SegResNet_Available_Candidates": len(seg_cands_sub),
                "SegResNet_Selected_Candidates": seg_k_act,
                "SegResNet_Achieved_Burden": seg_b_achieved,
                "SegResNet_Effective_Score_Threshold": round(seg_thresh, 4),
                "nnUNet_Available_Candidates": len(nnu_cands_sub),
                "nnUNet_Selected_Candidates": nnu_k_act,
                "nnUNet_Achieved_Burden": nnu_b_achieved,
                "nnUNet_Effective_Score_Threshold": round(nnu_thresh, 4),
                "Is_Strictly_Matched": is_strictly_matched,
                "Feasibility_Status": feas_status,
                "Bootstrap_Feasibility_Pass_Rate": round(feas_pass_rate, 3),
                "Tolerance_mm": primary_tol,
                "N_Evaluated_ValidTraj_csPCa": n_cs_val_denom,
                "SegResNet_Sens_ValidTraj_Pct": round(seg_s_val, 2),
                "SegResNet_Sens_ValidTraj_95CI": f"[{ci_seg[0]:.2f}, {ci_seg[1]:.2f}]",
                "nnUNet_Sens_ValidTraj_Pct": round(nnu_s_val, 2),
                "nnUNet_Sens_ValidTraj_95CI": f"[{ci_nnu[0]:.2f}, {ci_nnu[1]:.2f}]",
                "Delta_Sens_ValidTraj_Pct": round(seg_s_val - nnu_s_val, 2),
                "Delta_Sens_ValidTraj_95CI": f"[{ci_del[0]:.2f}, {ci_del[1]:.2f}]",
                "Bootstrap_P_Value_TwoSided": round(p_val, 4),
                "P_Value_Formatted": f"{p_val:.4f}" if p_val >= 0.001 else "< 0.001",
                # Candidate allocation by examination pathology (summing exactly to K)
                "SegResNet_csPCa_Cands": seg_exam_cs,
                "SegResNet_GG1_Cands": seg_exam_gg,
                "SegResNet_Benign_Cands": seg_exam_bn,
                "nnUNet_csPCa_Cands": nnu_exam_cs,
                "nnUNet_GG1_Cands": nnu_exam_gg,
                "nnUNet_Benign_Cands": nnu_exam_bn,
                "SegResNet_Exam_csPCa_Cands": seg_exam_cs,
                "SegResNet_Exam_GG1_Cands": seg_exam_gg,
                "SegResNet_Exam_Benign_Cands": seg_exam_bn,
                "nnUNet_Exam_csPCa_Cands": nnu_exam_cs,
                "nnUNet_Exam_GG1_Cands": nnu_exam_gg,
                "nnUNet_Exam_Benign_Cands": nnu_exam_bn,
                # Explicit proximity hit counts
                "SegResNet_Proximity_csPCa_Hits": seg_cs_c,
                "SegResNet_Proximity_GG1_Hits": seg_gg1_c,
                "SegResNet_Proximity_Benign_Hits": seg_ben_c,
                "nnUNet_Proximity_csPCa_Hits": nnu_cs_c,
                "nnUNet_Proximity_GG1_Hits": nnu_gg1_c,
                "nnUNet_Proximity_Benign_Hits": nnu_ben_c,
            })
            print(f"[{ext}] B={B:.2f} (K={K_req}): Status={feas_status} (BootFeas={feas_pass_rate:.1%}) | Seg={seg_s_val:.2f}% vs nnU={nnu_s_val:.2f}% | Delta={seg_s_val-nnu_s_val:+.2f}% {bm_rows[-1]['Delta_Sens_ValidTraj_95CI']} (p={bm_rows[-1]['P_Value_Formatted']})")

    df_bm = pd.DataFrame(bm_rows)
    df_bm.to_csv(os.path.join(AUDIT_DIR, "PRIMARY_BURDEN_MATCHED_RESULTS.csv"), index=False)

    # ==============================================================================
    # 4. SPATIAL TOLERANCE SWEEPS (SEPARATING CENTROID VS APPROXIMATE COMPONENT-VOLUME)
    # ==============================================================================
    print("\n" + "-" * 75)
    print("ANALYSIS 4: SPATIAL TOLERANCE SWEEPS (CENTROID VS COMPONENT-VOLUME)")
    print("-" * 75)

    tol_sweep_rows = []

    for ext in ["Official_RGA_Dynamic", "Shared_Fixed_CC_0.10"]:
        ext_cands = cands_paired[cands_paired["Extractor"] == ext]
        seg_cands = ext_cands[ext_cands["Model"] == "3D_SegResNet"]
        nnu_cands = ext_cands[ext_cands["Model"] == "Pretrained_nnUNet"]

        seg1 = seg_cands[seg_cands["Candidate_Rank"] == 1]
        seg3 = seg_cands[seg_cands["Candidate_Rank"] <= 3]
        nnu1 = nnu_cands[nnu_cands["Candidate_Rank"] == 1]
        nnu3 = nnu_cands[nnu_cands["Candidate_Rank"] <= 3]

        # 4A. Centroid Distance Series (2.5, 5.0, 7.5, 10.0 mm)
        for tol in [2.5, 5.0, 7.5, 10.0]:
            h_hits = int(np.sum(paired_cs_val["Human_Sess_CoM_Dist_csPCa_mm"].fillna(999.0) <= tol))
            h_sens = (h_hits / n_cs_val_denom) * 100.0

            s_top1_com = (seg1[seg1["CoM_Dist_csPCa_mm"] <= tol]["Session_ID"].nunique() / n_cs_val_denom * 100.0)
            s_top3_com = (seg3[seg3["CoM_Dist_csPCa_mm"] <= tol]["Session_ID"].nunique() / n_cs_val_denom * 100.0)
            n_top1_com = (nnu1[nnu1["CoM_Dist_csPCa_mm"] <= tol]["Session_ID"].nunique() / n_cs_val_denom * 100.0)
            n_top3_com = (nnu3[nnu3["CoM_Dist_csPCa_mm"] <= tol]["Session_ID"].nunique() / n_cs_val_denom * 100.0)

            # Exact geometric union with human target geometry (H OR AI)
            h_hit_sids = set(paired_cs_val[paired_cs_val["Human_Sess_CoM_Dist_csPCa_mm"].fillna(999.0) <= tol]["Session_ID"])
            s1_hit_sids = set(seg1[seg1["CoM_Dist_csPCa_mm"] <= tol]["Session_ID"].unique())
            s3_hit_sids = set(seg3[seg3["CoM_Dist_csPCa_mm"] <= tol]["Session_ID"].unique())
            geom_union_top1 = (len(h_hit_sids.union(s1_hit_sids)) / n_cs_val_denom) * 100.0
            geom_union_top3 = (len(h_hit_sids.union(s3_hit_sids)) / n_cs_val_denom) * 100.0

            tol_sweep_rows.append({
                "Extractor": ext,
                "Distance_Metric": "Session_Center_of_Mass_CoM",
                "Tolerance_mm": tol,
                "N_Evaluated_ValidTraj_csPCa": n_cs_val_denom,
                "Human_Target_Sens_Pct": round(h_sens, 2),
                "SegResNet_Top1_Sens_Pct": round(s_top1_com, 2),
                "SegResNet_Top3_Sens_Pct": round(s_top3_com, 2),
                "nnUNet_Top1_Sens_Pct": round(n_top1_com, 2),
                "nnUNet_Top3_Sens_Pct": round(n_top3_com, 2),
                "Delta_Top1_Sens_Pct": round(s_top1_com - n_top1_com, 2),
                "Geometric_Union_Top1_Sens_Pct": round(geom_union_top1, 2),
                "Geometric_Union_Top3_Sens_Pct": round(geom_union_top3, 2),
            })

        # 4B. Approximate Component-Volume Distance Series (0.0 direct traversal, 2.5, 5.0, 7.5, 10.0 mm)
        for tol in [0.0, 2.5, 5.0, 7.5, 10.0]:
            if tol == 0.0:
                h_hits = int(np.sum(paired_cs_val["Human_Sess_DirectPuncture_0mm_csPCa"] == 1))
                h_hit_sids = set(paired_cs_val[paired_cs_val["Human_Sess_DirectPuncture_0mm_csPCa"] == 1]["Session_ID"])
            else:
                h_hits = int(np.sum(paired_cs_val["Human_Sess_Surface3D_Dist_csPCa_mm"].fillna(999.0) <= tol))
                h_hit_sids = set(paired_cs_val[paired_cs_val["Human_Sess_Surface3D_Dist_csPCa_mm"].fillna(999.0) <= tol]["Session_ID"])
            h_sens = (h_hits / n_cs_val_denom) * 100.0

            s1_hit_sids = set(seg1[seg1["Surf3D_Dist_csPCa_mm"] <= tol]["Session_ID"].unique())
            s3_hit_sids = set(seg3[seg3["Surf3D_Dist_csPCa_mm"] <= tol]["Session_ID"].unique())
            geom_union_top1 = (len(h_hit_sids.union(s1_hit_sids)) / n_cs_val_denom) * 100.0
            geom_union_top3 = (len(h_hit_sids.union(s3_hit_sids)) / n_cs_val_denom) * 100.0

            s_top1_vol = (seg1[seg1["Surf3D_Dist_csPCa_mm"] <= tol]["Session_ID"].nunique() / n_cs_val_denom * 100.0)
            s_top3_vol = (seg3[seg3["Surf3D_Dist_csPCa_mm"] <= tol]["Session_ID"].nunique() / n_cs_val_denom * 100.0)
            n_top1_vol = (nnu1[nnu1["Surf3D_Dist_csPCa_mm"] <= tol]["Session_ID"].nunique() / n_cs_val_denom * 100.0)
            n_top3_vol = (nnu3[nnu3["Surf3D_Dist_csPCa_mm"] <= tol]["Session_ID"].nunique() / n_cs_val_denom * 100.0)

            tol_sweep_rows.append({
                "Extractor": ext,
                "Distance_Metric": "Approximate_Component_Volume_Distance",
                "Tolerance_mm": tol,
                "N_Evaluated_ValidTraj_csPCa": n_cs_val_denom,
                "Human_Target_Sens_Pct": round(h_sens, 2),
                "SegResNet_Top1_Sens_Pct": round(s_top1_vol, 2),
                "SegResNet_Top3_Sens_Pct": round(s_top3_vol, 2),
                "nnUNet_Top1_Sens_Pct": round(n_top1_vol, 2),
                "nnUNet_Top3_Sens_Pct": round(n_top3_vol, 2),
                "Delta_Top1_Sens_Pct": round(s_top1_vol - n_top1_vol, 2),
                "Geometric_Union_Top1_Sens_Pct": round(geom_union_top1, 2),
                "Geometric_Union_Top3_Sens_Pct": round(geom_union_top3, 2),
            })

    df_tol_sweep = pd.DataFrame(tol_sweep_rows)
    df_tol_sweep.to_csv(os.path.join(AUDIT_DIR, "TOLERANCE_SWEEP_RESULTS.csv"), index=False)
    print(f"Saved TOLERANCE_SWEEP_RESULTS.csv ({len(df_tol_sweep)} rows)")

    # ==============================================================================
    # 5. RECONCILED CLINICAL SUSPICION TIER EVALUATION (CANONICAL & PAIRED)
    # ==============================================================================
    print("\n" + "-" * 75)
    print("ANALYSIS 5: CLINICAL SUSPICION TIER BREAKDOWN")
    print("-" * 75)

    def generate_tier_table(df_cohort, cohort_name):
        tier_defs = [
            ("Tier1_Low_Score_0_to_2", df_cohort[df_cohort["Exact_Exam_UCLA_Score"].isin([0, 1, 2])]),
            ("Tier2_Equivocal_Score_3", df_cohort[df_cohort["Exact_Exam_UCLA_Score"] == 3]),
            ("Tier3_High_Score_4_to_5", df_cohort[df_cohort["Exact_Exam_UCLA_Score"].isin([4, 5])]),
            ("Score_Missing_Unassignable", df_cohort[df_cohort["Exact_Exam_UCLA_Score"] == -1]),
            ("All_Examinations_Overall", df_cohort),
        ]

        table_rows = []
        for tier_label, sub in tier_defs:
            n_exams = len(sub)
            cs_all = sub[sub["New_Sess_Is_csPCa_Independent"] == 1]
            n_cs_all = len(cs_all)
            cs_val = cs_all[cs_all["Sess_Has_Valid_Pos_Core"] == 1]
            n_cs_val = len(cs_val)

            val_sids = set(cs_val["Session_ID"])

            # 1. Human Target Mask at CoM 10mm
            h_hits = int(np.sum(cs_val["Human_Sess_CoM_Dist_csPCa_mm"].fillna(999.0) <= 10.0))
            h_sens = round(h_hits / n_cs_val * 100.0, 2) if n_cs_val > 0 else 0.0

            # 2. Recorded Targeted Biopsy (All and Valid)
            b_hits_all = int(np.sum(cs_all["Targeted_csPCa_Hit"] == 1))
            b_sens_all = round(b_hits_all / n_cs_all * 100.0, 2) if n_cs_all > 0 else 0.0
            b_hits_val = int(np.sum(cs_val["Targeted_csPCa_Hit"] == 1))
            b_sens_val = round(b_hits_val / n_cs_val * 100.0, 2) if n_cs_val > 0 else 0.0

            # 3. SegResNet RGA Top-1 and Top-3
            seg_c_sub = cands_paired[(cands_paired["Extractor"] == "Official_RGA_Dynamic") & (cands_paired["Model"] == "3D_SegResNet") & (cands_paired["Session_ID"].isin(val_sids))]
            seg1_hit_sids = set(seg_c_sub[(seg_c_sub["Candidate_Rank"] == 1) & (seg_c_sub["CoM_Dist_csPCa_mm"] <= 10.0)]["Session_ID"])
            seg3_hit_sids = set(seg_c_sub[(seg_c_sub["Candidate_Rank"] <= 3) & (seg_c_sub["CoM_Dist_csPCa_mm"] <= 10.0)]["Session_ID"])

            seg1_hits = len(seg1_hit_sids)
            seg1_sens = round(seg1_hits / n_cs_val * 100.0, 2) if n_cs_val > 0 else 0.0
            seg3_hits = len(seg3_hit_sids)
            seg3_sens = round(seg3_hits / n_cs_val * 100.0, 2) if n_cs_val > 0 else 0.0

            # 4. nnU-Net RGA Top-1 and Top-3
            nnu_c_sub = cands_paired[(cands_paired["Extractor"] == "Official_RGA_Dynamic") & (cands_paired["Model"] == "Pretrained_nnUNet") & (cands_paired["Session_ID"].isin(val_sids))]
            nnu1_hit_sids = set(nnu_c_sub[(nnu_c_sub["Candidate_Rank"] == 1) & (nnu_c_sub["CoM_Dist_csPCa_mm"] <= 10.0)]["Session_ID"])
            nnu3_hit_sids = set(nnu_c_sub[(nnu_c_sub["Candidate_Rank"] <= 3) & (nnu_c_sub["CoM_Dist_csPCa_mm"] <= 10.0)]["Session_ID"])

            nnu1_hits = len(nnu1_hit_sids)
            nnu1_sens = round(nnu1_hits / n_cs_val * 100.0, 2) if n_cs_val > 0 else 0.0
            nnu3_hits = len(nnu3_hit_sids)
            nnu3_sens = round(nnu3_hits / n_cs_val * 100.0, 2) if n_cs_val > 0 else 0.0

            # 5. Combined Human + SegResNet
            h_hit_sids = set(cs_val[cs_val["Human_Sess_CoM_Dist_csPCa_mm"].fillna(999.0) <= 10.0]["Session_ID"])
            comb1_sids = h_hit_sids.union(seg1_hit_sids)
            comb3_sids = h_hit_sids.union(seg3_hit_sids)

            comb1_hits = len(comb1_sids)
            comb1_sens = round(comb1_hits / n_cs_val * 100.0, 2) if n_cs_val > 0 else 0.0
            comb3_hits = len(comb3_sids)
            comb3_sens = round(comb3_hits / n_cs_val * 100.0, 2) if n_cs_val > 0 else 0.0

            extra_geo1 = comb1_hits - h_hits
            extra_geo3 = comb3_hits - h_hits

            # 6. Overlap Breakdown
            overlap_both_1 = len(h_hit_sids.intersection(seg1_hit_sids))
            overlap_h_only_1 = len(h_hit_sids - seg1_hit_sids)
            overlap_ai_only_1 = len(seg1_hit_sids - h_hit_sids)
            overlap_neither_1 = n_cs_val - len(comb1_sids)

            overlap_both_3 = len(h_hit_sids.intersection(seg3_hit_sids))
            overlap_h_only_3 = len(h_hit_sids - seg3_hit_sids)
            overlap_ai_only_3 = len(seg3_hit_sids - h_hit_sids)
            overlap_neither_3 = n_cs_val - len(comb3_sids)

            table_rows.append({
                "Cohort": cohort_name,
                "Suspicion_Stratum": tier_label,
                "N_Total_Examinations": n_exams,
                "N_csPCa_All_Denom": n_cs_all,
                "N_csPCa_ValidTraj_Denom": n_cs_val,
                "Human_Mask_Hits": h_hits,
                "Human_Mask_Sens_Pct": h_sens,
                "Human_Mask_95CI": fmt_ci(h_hits, n_cs_val),
                "Recorded_Targeted_Biopsy_Hits_All": b_hits_all,
                "Recorded_Targeted_Biopsy_Sens_All_Pct": b_sens_all,
                "Recorded_Targeted_Biopsy_Hits_ValidTraj": b_hits_val,
                "Recorded_Targeted_Biopsy_Sens_ValidTraj_Pct": b_sens_val,
                "SegResNet_Top1_Hits": seg1_hits,
                "SegResNet_Top1_Sens_Pct": seg1_sens,
                "SegResNet_Top1_95CI": fmt_ci(seg1_hits, n_cs_val),
                "SegResNet_Top3_Hits": seg3_hits,
                "SegResNet_Top3_Sens_Pct": seg3_sens,
                "SegResNet_Top3_95CI": fmt_ci(seg3_hits, n_cs_val),
                "nnUNet_Top1_Hits": nnu1_hits,
                "nnUNet_Top1_Sens_Pct": nnu1_sens,
                "nnUNet_Top1_95CI": fmt_ci(nnu1_hits, n_cs_val),
                "nnUNet_Top3_Hits": nnu3_hits,
                "nnUNet_Top3_Sens_Pct": nnu3_sens,
                "nnUNet_Top3_95CI": fmt_ci(nnu3_hits, n_cs_val),
                "Combined_Human_SegTop1_Hits": comb1_hits,
                "Combined_Human_SegTop1_Sens_Pct": comb1_sens,
                "Combined_Human_SegTop1_95CI": fmt_ci(comb1_hits, n_cs_val),
                "Combined_Human_SegTop3_Hits": comb3_hits,
                "Combined_Human_SegTop3_Sens_Pct": comb3_sens,
                "Combined_Human_SegTop3_95CI": fmt_ci(comb3_hits, n_cs_val),
                "Incremental_Spatial_Correspondences_SegTop1": extra_geo1,
                "Incremental_Sens_Gain_SegTop1_Pct": round(comb1_sens - h_sens, 2),
                "Incremental_Spatial_Correspondences_SegTop3": extra_geo3,
                "Incremental_Sens_Gain_SegTop3_Pct": round(comb3_sens - h_sens, 2),
                "Overlap_Top1_Both_Hits": overlap_both_1,
                "Overlap_Top1_Human_Only_Hits": overlap_h_only_1,
                "Overlap_Top1_AI_Only_Hits": overlap_ai_only_1,
                "Overlap_Top1_Neither_Hits": overlap_neither_1,
                "Overlap_Top3_Both_Hits": overlap_both_3,
                "Overlap_Top3_Human_Only_Hits": overlap_h_only_3,
                "Overlap_Top3_AI_Only_Hits": overlap_ai_only_3,
                "Overlap_Top3_Neither_Hits": overlap_neither_3,
            })
            print(f"[{cohort_name} - {tier_label}]: Valid csPCa N={n_cs_val} | Human={h_sens}% | SegTop1={seg1_sens}% | SegTop3={seg3_sens}% | CombTop1={comb1_sens}% (+{extra_geo1}) | CombTop3={comb3_sens}% (+{extra_geo3})")
        return pd.DataFrame(table_rows)

    df_t3a_canon = generate_tier_table(df_canon, "Canonical_Cohort_N809")
    df_t3a_paired = generate_tier_table(df_paired, "Paired_Cohort_N800")

    df_t3a_canon.to_csv(os.path.join(AUDIT_DIR, "TABLE3A_PIRADS_TIER_STRATIFIED_EVALUATION.csv"), index=False)
    df_t3a_paired.to_csv(os.path.join(AUDIT_DIR, "TABLE3A_PAIRED_COHORT_STRATIFIED_EVALUATION.csv"), index=False)

    # 5B. Export EXAMINATION_LEVEL_ENDPOINTS.csv & Dual-Panel Table 3 (Panel A: Geometric H vs Panel B: Recorded Yield C)
    rga_cands_ep = cands_paired[cands_paired["Extractor"] == "Official_RGA_Dynamic"].copy()
    hit_maps_ep = {}
    for model_ep, m_prefix in [("3D_SegResNet", "SegResNet"), ("Pretrained_nnUNet", "nnUNet")]:
        m_df_ep = rga_cands_ep[rga_cands_ep["Model"] == model_ep]
        for k_ep in [1, 2, 3]:
            topk_ep = m_df_ep[m_df_ep["Candidate_Rank"] <= k_ep]
            hits_10_ep = topk_ep[topk_ep["CoM_Dist_csPCa_mm"] <= 10.0]
            hit_maps_ep[f"{m_prefix}_A_{k_ep}"] = set(hits_10_ep["Session_ID"])

    ep_records = []
    for _, row_ep in df_paired.iterrows():
        sid_ep = row_ep["Session_ID"]
        pid_ep = row_ep["Patient_ID"]
        score_ep = row_ep["Exact_Exam_UCLA_Score"]
        is_cs_ep = int(row_ep["New_Sess_Is_csPCa_Independent"])
        has_valid_traj_ep = int(row_ep["Sess_Has_Valid_Pos_Core"] == 1) if is_cs_ep else int(row_ep["Total_Cores_Linked"] > 0)
        if score_ep in [1, 2]:
            tier_ep = "Low_Suspicion_Scores_1_to_2"
        elif score_ep == 0:
            tier_ep = "Score_0_Unrated_or_Atypical"
        elif score_ep == 3:
            tier_ep = "Equivocal_Suspicion_Score_3"
        elif score_ep in [4, 5]:
            tier_ep = "High_Suspicion_Scores_4_to_5"
        else:
            tier_ep = "Score_Missing_or_Unassigned"
        targ_cores_ep = int(row_ep["Targeted_Cores_Linked"])
        sys_cores_ep = int(row_ep["Systematic_Cores_Linked"])
        c_flag_ep = int(row_ep["Targeted_csPCa_Hit"] == 1) if is_cs_ep else 0
        sys_cs_hit_ep = int(row_ep["Systematic_csPCa_Hit"] == 1) if is_cs_ep else 0
        h_dist_ep = row_ep["Human_Sess_CoM_Dist_csPCa_mm"]
        has_human_target_ep = int(pd.notna(row_ep["Human_Rad_Mask_Vol_cc"]) and row_ep["Human_Rad_Mask_Vol_cc"] > 0)
        h_flag_ep = int(pd.notna(h_dist_ep) and h_dist_ep <= 10.0) if (is_cs_ep and has_valid_traj_ep) else 0
        seg_a1_ep = int(sid_ep in hit_maps_ep["SegResNet_A_1"]) if (is_cs_ep and has_valid_traj_ep) else 0
        seg_a2_ep = int(sid_ep in hit_maps_ep["SegResNet_A_2"]) if (is_cs_ep and has_valid_traj_ep) else 0
        seg_a3_ep = int(sid_ep in hit_maps_ep["SegResNet_A_3"]) if (is_cs_ep and has_valid_traj_ep) else 0
        nnu_a1_ep = int(sid_ep in hit_maps_ep["nnUNet_A_1"]) if (is_cs_ep and has_valid_traj_ep) else 0
        nnu_a2_ep = int(sid_ep in hit_maps_ep["nnUNet_A_2"]) if (is_cs_ep and has_valid_traj_ep) else 0
        nnu_a3_ep = int(sid_ep in hit_maps_ep["nnUNet_A_3"]) if (is_cs_ep and has_valid_traj_ep) else 0
        ep_records.append({
            "Session_ID": sid_ep, "Patient_ID": pid_ep, "Exact_Exam_UCLA_Score": score_ep,
            "Clinical_Suspicion_Tier": tier_ep, "Is_csPCa": is_cs_ep, "Has_Valid_3D_Trajectory": has_valid_traj_ep,
            "Targeted_Cores_Linked": targ_cores_ep, "Systematic_Cores_Linked": sys_cores_ep,
            "Targeted_Biopsy_csPCa_Yield_C": c_flag_ep, "Systematic_Biopsy_csPCa_Hit": sys_cs_hit_ep,
            "Has_Human_Target_Contour": has_human_target_ep,
            "Human_Target_CoM_Dist_csPCa_mm": round(h_dist_ep, 3) if pd.notna(h_dist_ep) else np.nan,
            "Human_Target_Geometry_H": h_flag_ep,
            "SegResNet_A_1": seg_a1_ep, "SegResNet_A_2": seg_a2_ep, "SegResNet_A_3": seg_a3_ep,
            "H_OR_SegResNet_A_1": int(h_flag_ep or seg_a1_ep), "H_OR_SegResNet_A_2": int(h_flag_ep or seg_a2_ep), "H_OR_SegResNet_A_3": int(h_flag_ep or seg_a3_ep),
            "C_OR_SegResNet_A_1": int(c_flag_ep or seg_a1_ep), "C_OR_SegResNet_A_2": int(c_flag_ep or seg_a2_ep), "C_OR_SegResNet_A_3": int(c_flag_ep or seg_a3_ep),
            "nnUNet_A_1": nnu_a1_ep, "nnUNet_A_2": nnu_a2_ep, "nnUNet_A_3": nnu_a3_ep,
            "H_OR_nnUNet_A_1": int(h_flag_ep or nnu_a1_ep), "H_OR_nnUNet_A_2": int(h_flag_ep or nnu_a2_ep), "H_OR_nnUNet_A_3": int(h_flag_ep or nnu_a3_ep),
            "C_OR_nnUNet_A_1": int(c_flag_ep or nnu_a1_ep), "C_OR_nnUNet_A_2": int(c_flag_ep or nnu_a2_ep), "C_OR_nnUNet_A_3": int(c_flag_ep or nnu_a3_ep),
        })
    df_ep = pd.DataFrame(ep_records)
    endpoints_csv = os.path.join(AUDIT_DIR, "EXAMINATION_LEVEL_ENDPOINTS.csv")
    df_ep.to_csv(endpoints_csv, index=False)
    df_ep.to_csv(os.path.join(SOURCES_DIR, "EXAMINATION_LEVEL_ENDPOINTS.csv"), index=False)

    if os.path.exists(endpoints_csv):
        df_ep = pd.read_csv(endpoints_csv)
        strata_def = [
            ("Scores 1-2 (Low)", df_ep["Exact_Exam_UCLA_Score"].isin([1, 2])),
            ("Score 3 (Equivocal)", df_ep["Exact_Exam_UCLA_Score"] == 3),
            ("Scores 4-5 (High)", df_ep["Exact_Exam_UCLA_Score"].isin([4, 5])),
            ("Overall Evaluable Cohort", [True] * len(df_ep)),
            ("Score 0 (Footnote Cohort)", df_ep["Exact_Exam_UCLA_Score"] == 0),
            ("Scores 0-2 (Pooled Sensitivity Analysis)", df_ep["Exact_Exam_UCLA_Score"].isin([0, 1, 2])),
            ("Missing / Unassigned Score", df_ep["Exact_Exam_UCLA_Score"] == -1),
        ]

        def fmt_cell(hits, total):
            if total == 0:
                return "0 (0.00%) [0.00, 0.00]"
            pct = hits / total * 100.0
            low, high = wilson_score_interval(hits, total)
            return f"{hits} ({pct:.2f}%) [{low:.2f}, {high:.2f}]"

        p_a_rows, p_b_rows = [], []
        for s_name, s_mask in strata_def:
            s_sub = df_ep[s_mask]
            s_tot = len(s_sub)
            s_cs = int(s_sub["Is_csPCa"].sum())
            s_val = s_sub[(s_sub["Is_csPCa"] == 1) & (s_sub["Has_Valid_3D_Trajectory"] == 1)]
            n_v = len(s_val)

            h_v = int(s_val["Human_Target_Geometry_H"].sum())
            c_v = int(s_val["Targeted_Biopsy_csPCa_Yield_C"].sum())

            s1_v = int(s_val["SegResNet_A_1"].sum())
            s3_v = int(s_val["SegResNet_A_3"].sum())
            n1_v = int(s_val["nnUNet_A_1"].sum())
            n3_v = int(s_val["nnUNet_A_3"].sum())

            hs1_v = int(s_val["H_OR_SegResNet_A_1"].sum())
            hs3_v = int(s_val["H_OR_SegResNet_A_3"].sum())
            hn1_v = int(s_val["H_OR_nnUNet_A_1"].sum())
            hn3_v = int(s_val["H_OR_nnUNet_A_3"].sum())

            cs1_v = int(s_val["C_OR_SegResNet_A_1"].sum())
            cs3_v = int(s_val["C_OR_SegResNet_A_3"].sum())
            cn1_v = int(s_val["C_OR_nnUNet_A_1"].sum())
            cn3_v = int(s_val["C_OR_nnUNet_A_3"].sum())

            p_a_rows.append({
                "Stratum": s_name, "N_Total_Exams": s_tot, "N_csPCa_All": s_cs, "N_Evaluable_csPCa": n_v,
                "Human_Target_Geometry_H": fmt_cell(h_v, n_v),
                "SegResNet_A1": fmt_cell(s1_v, n_v),
                "SegResNet_A3": fmt_cell(s3_v, n_v),
                "nnUNet_A1": fmt_cell(n1_v, n_v),
                "nnUNet_A3": fmt_cell(n3_v, n_v),
                "H_OR_SegResNet_A1": f"{fmt_cell(hs1_v, n_v)} (+{hs1_v - h_v})",
                "H_OR_SegResNet_A3": f"{fmt_cell(hs3_v, n_v)} (+{hs3_v - h_v})",
                "H_OR_nnUNet_A1": f"{fmt_cell(hn1_v, n_v)} (+{hn1_v - h_v})",
                "H_OR_nnUNet_A3": f"{fmt_cell(hn3_v, n_v)} (+{hn3_v - h_v})",
            })

            p_b_rows.append({
                "Stratum": s_name, "N_Total_Exams": s_tot, "N_csPCa_All": s_cs, "N_Evaluable_csPCa": n_v,
                "Recorded_Targeted_Yield_C": fmt_cell(c_v, n_v),
                "C_OR_SegResNet_A1": f"{fmt_cell(cs1_v, n_v)} (+{cs1_v - c_v})",
                "C_OR_SegResNet_A3": f"{fmt_cell(cs3_v, n_v)} (+{cs3_v - c_v})",
                "C_OR_nnUNet_A1": f"{fmt_cell(cn1_v, n_v)} (+{cn1_v - c_v})",
                "C_OR_nnUNet_A3": f"{fmt_cell(cn3_v, n_v)} (+{cn3_v - c_v})",
            })

        df_pa_out = pd.DataFrame(p_a_rows)
        df_pb_out = pd.DataFrame(p_b_rows)
        df_pa_out.to_csv(os.path.join(AUDIT_DIR, "TABLE3_PANEL_A_GEOMETRIC.csv"), index=False)
        df_pb_out.to_csv(os.path.join(AUDIT_DIR, "TABLE3_PANEL_B_RECORDED_YIELD.csv"), index=False)
        # Combined dual-panel representation
        df_p_merged = df_pa_out.merge(df_pb_out.drop(columns=["N_Total_Exams", "N_csPCa_All", "N_Evaluable_csPCa"]), on="Stratum", how="left")
        df_p_merged.to_csv(os.path.join(AUDIT_DIR, "TABLE3_DUAL_COMPARATOR_TIER_STRATIFIED.csv"), index=False)
        print(f"Saved TABLE3_DUAL_COMPARATOR_TIER_STRATIFIED.csv ({len(df_p_merged)} rows)")

    # ==============================================================================
    # 6. RECORDED PROCEDURAL TARGETED BIOPSY PATHWAY RESULTS
    # ==============================================================================
    print("\n" + "-" * 75)
    print("ANALYSIS 6: RECORDED PROCEDURAL TARGETED BIOPSY PATHWAY RESULTS")
    print("-" * 75)

    proc_rows = []
    for c_name, c_df in [("Canonical_Cohort_N809", df_canon), ("Paired_Cohort_N800", df_paired)]:
        cs_all = c_df[c_df["New_Sess_Is_csPCa_Independent"] == 1]
        cs_val = cs_all[cs_all["Sess_Has_Valid_Pos_Core"] == 1]

        # 1. All recorded targeted cores unfiltered
        hits_unfilt_all = int(np.sum(cs_all["Targeted_csPCa_Hit"] == 1))
        sens_unfilt_all = round(hits_unfilt_all / len(cs_all) * 100.0, 2)
        sys_only_unfilt_all = len(cs_all) - hits_unfilt_all

        hits_unfilt_val = int(np.sum(cs_val["Targeted_csPCa_Hit"] == 1))
        sens_unfilt_val = round(hits_unfilt_val / len(cs_val) * 100.0, 2)
        sys_only_unfilt_val = len(cs_val) - hits_unfilt_val

        # Sub-breakdown of systematic-only (Targeted-Negative vs Targeted-Absent)
        t_neg_all = int(np.sum((cs_all["Targeted_csPCa_Hit"] == 0) & (cs_all["Targeted_Cores_Linked"] > 0)))
        t_abs_all = int(np.sum((cs_all["Targeted_csPCa_Hit"] == 0) & (cs_all["Targeted_Cores_Linked"] == 0)))

        t_neg_val = int(np.sum((cs_val["Targeted_csPCa_Hit"] == 0) & (cs_val["Targeted_Cores_Linked"] > 0)))
        t_abs_val = int(np.sum((cs_val["Targeted_csPCa_Hit"] == 0) & (cs_val["Targeted_Cores_Linked"] == 0)))

        # 2. Retrospective Gated Score >= 3
        gated_hit_mask_all = (cs_all["Targeted_csPCa_Hit"] == 1) & (cs_all["Exact_Exam_UCLA_Score"] >= 3)
        hits_gated_all = int(np.sum(gated_hit_mask_all))
        sens_gated_all = round(hits_gated_all / len(cs_all) * 100.0, 2)

        gated_hit_mask_val = (cs_val["Targeted_csPCa_Hit"] == 1) & (cs_val["Exact_Exam_UCLA_Score"] >= 3)
        hits_gated_val = int(np.sum(gated_hit_mask_val))
        sens_gated_val = round(hits_gated_val / len(cs_val) * 100.0, 2)

        proc_rows.append({
            "Cohort": c_name,
            "Targeted_Biopsy_Definition": "All_Recorded_Targeted_Cores_Unfiltered",
            "N_csPCa_All": len(cs_all),
            "Targeted_Positive_Hits_All": hits_unfilt_all,
            "Targeted_Positive_Sens_All_Pct": sens_unfilt_all,
            "Targeted_Positive_All_95CI": fmt_ci(hits_unfilt_all, len(cs_all)),
            "Systematic_Only_Total_All": sys_only_unfilt_all,
            "Systematic_Only_Total_All_Pct": round(sys_only_unfilt_all / len(cs_all) * 100.0, 2),
            "Systematic_Only_Targeted_Negative_All": t_neg_all,
            "Systematic_Only_Targeted_Absent_All": t_abs_all,
            "N_csPCa_ValidTraj": len(cs_val),
            "Targeted_Positive_Hits_ValidTraj": hits_unfilt_val,
            "Targeted_Positive_Sens_ValidTraj_Pct": sens_unfilt_val,
            "Targeted_Positive_ValidTraj_95CI": fmt_ci(hits_unfilt_val, len(cs_val)),
            "Systematic_Only_Total_ValidTraj": sys_only_unfilt_val,
            "Systematic_Only_Targeted_Negative_ValidTraj": t_neg_val,
            "Systematic_Only_Targeted_Absent_ValidTraj": t_abs_val,
        })

        proc_rows.append({
            "Cohort": c_name,
            "Targeted_Biopsy_Definition": "Retrospective_Gated_UCLA_Score_ge_3",
            "N_csPCa_All": len(cs_all),
            "Targeted_Positive_Hits_All": hits_gated_all,
            "Targeted_Positive_Sens_All_Pct": sens_gated_all,
            "Targeted_Positive_All_95CI": fmt_ci(hits_gated_all, len(cs_all)),
            "Systematic_Only_Total_All": len(cs_all) - hits_gated_all,
            "Systematic_Only_Total_All_Pct": round((len(cs_all) - hits_gated_all) / len(cs_all) * 100.0, 2),
            "Systematic_Only_Targeted_Negative_All": np.nan,
            "Systematic_Only_Targeted_Absent_All": np.nan,
            "N_csPCa_ValidTraj": len(cs_val),
            "Targeted_Positive_Hits_ValidTraj": hits_gated_val,
            "Targeted_Positive_Sens_ValidTraj_Pct": sens_gated_val,
            "Targeted_Positive_ValidTraj_95CI": fmt_ci(hits_gated_val, len(cs_val)),
            "Systematic_Only_Total_ValidTraj": len(cs_val) - hits_gated_val,
            "Systematic_Only_Targeted_Negative_ValidTraj": np.nan,
            "Systematic_Only_Targeted_Absent_ValidTraj": np.nan,
        })
        print(f"[{c_name}] Unfiltered Targeted Biopsy Sens: All csPCa = {hits_unfilt_all}/{len(cs_all)} ({sens_unfilt_all}%), ValidTraj = {hits_unfilt_val}/{len(cs_val)} ({sens_unfilt_val}%)")
        print(f"[{c_name}] Gated Score>=3 Targeted Biopsy Sens: All csPCa = {hits_gated_all}/{len(cs_all)} ({sens_gated_all}%)")

    df_proc = pd.DataFrame(proc_rows)
    df_proc.to_csv(os.path.join(AUDIT_DIR, "RECORDED_PROCEDURAL_BIOPSY_RESULTS.csv"), index=False)

    # ==============================================================================
    # 7. SYSTEMATIC-ONLY csPCa HEAD-TO-HEAD COMPARISON
    # ==============================================================================
    print("\n" + "-" * 75)
    print("ANALYSIS 7: SYSTEMATIC-ONLY csPCa HEAD-TO-HEAD COMPARISON")
    print("-" * 75)

    sys_only_sessions = df_paired[
        (df_paired["New_Sess_Is_csPCa_Independent"] == 1) &
        (df_paired["Targeted_csPCa_Hit"] == 0) &
        (df_paired["Systematic_csPCa_Hit"] == 1)
    ]
    n_sys_all = len(sys_only_sessions)
    sys_only_val = sys_only_sessions[sys_only_sessions["Sess_Has_Valid_Pos_Core"] == 1]
    n_sys_val = len(sys_only_val)

    sys_targ_neg = sys_only_val[sys_only_val["Targeted_Cores_Linked"] > 0]
    sys_targ_abs = sys_only_val[sys_only_val["Targeted_Cores_Linked"] == 0]
    n_targ_neg = len(sys_targ_neg)
    n_targ_abs = len(sys_targ_abs)

    # Human Target Geometry within this systematic-only subgroup
    h_hit_sids_sys = set(sys_only_val[sys_only_val["Human_Sess_CoM_Dist_csPCa_mm"].fillna(999.0) <= 10.0]["Session_ID"])
    h_hits_sys_count = len(h_hit_sids_sys)
    h_hits_sys_pct = round(h_hits_sys_count / n_sys_val * 100.0, 2)

    sys_rows = []
    ranks = [1, 2, 3]

    for ext in ["Official_RGA_Dynamic", "Shared_Fixed_CC_0.10"]:
        ext_cands = cands_paired[cands_paired["Extractor"] == ext]
        for k in ranks:
            for mod in models:
                mod_cands = ext_cands[(ext_cands["Model"] == mod) & (ext_cands["Candidate_Rank"] <= k)]
                mod_cands_val = mod_cands[mod_cands["Session_ID"].isin(set(sys_only_val["Session_ID"]))]

                # Hits at CoM 10mm
                hit_sids_com = set(mod_cands_val[mod_cands_val["CoM_Dist_csPCa_mm"] <= 10.0]["Session_ID"])
                hits_com = len(hit_sids_com)
                sens_com = round(hits_com / n_sys_val * 100.0, 2)

                # Subgroup hits
                hits_neg = len(hit_sids_com.intersection(set(sys_targ_neg["Session_ID"])))
                sens_neg = round(hits_neg / n_targ_neg * 100.0, 2) if n_targ_neg > 0 else 0.0

                hits_abs = len(hit_sids_com.intersection(set(sys_targ_abs["Session_ID"])))
                sens_abs = round(hits_abs / n_targ_abs * 100.0, 2) if n_targ_abs > 0 else 0.0

                # Actual geometric union with human target geometry (H OR AI)
                union_sids_sys = h_hit_sids_sys.union(hit_sids_com)
                hits_union_sys = len(union_sids_sys)
                sens_union_sys = round(hits_union_sys / n_sys_val * 100.0, 2)

                # Secondary metrics: Peak 10mm, Surf 10mm, Direct 0mm
                hit_sids_peak = set(mod_cands_val[mod_cands_val["Peak_Dist_csPCa_mm"] <= 10.0]["Session_ID"])
                hit_sids_surf = set(mod_cands_val[mod_cands_val["Surf3D_Dist_csPCa_mm"] <= 10.0]["Session_ID"])
                hit_sids_dir0 = set(mod_cands_val[mod_cands_val["Surf3D_Dist_csPCa_mm"] == 0.0]["Session_ID"])

                hits_peak = len(hit_sids_peak)
                hits_surf = len(hit_sids_surf)
                hits_dir0 = len(hit_sids_dir0)

                sys_rows.append({
                    "Extractor": ext,
                    "Operating_Point": f"Top_{k}",
                    "Model": mod,
                    "N_SysOnly_All": n_sys_all,
                    "N_SysOnly_ValidTraj": n_sys_val,
                    "Human_Target_Geometry_Hits": h_hits_sys_count,
                    "Human_Target_Geometry_Sens_Pct": h_hits_sys_pct,
                    "Hits_CoM10": hits_com,
                    "Sens_CoM10_ValidTraj_Pct": sens_com,
                    "Sens_CoM10_ValidTraj_95CI": fmt_ci(hits_com, n_sys_val),
                    "Geometric_Union_Hits": hits_union_sys,
                    "Geometric_Union_Sens_Pct": sens_union_sys,
                    "Geometric_Union_95CI": fmt_ci(hits_union_sys, n_sys_val),
                    "Hits_Targeted_Negative_Subgroup": hits_neg,
                    "N_Targeted_Negative_Subgroup": n_targ_neg,
                    "Sens_Targeted_Negative_Subgroup_Pct": sens_neg,
                    "Hits_Targeted_Absent_Subgroup": hits_abs,
                    "N_Targeted_Absent_Subgroup": n_targ_abs,
                    "Sens_Targeted_Absent_Subgroup_Pct": sens_abs,
                    "Hits_Peak10": hits_peak,
                    "Sens_Peak10_ValidTraj_Pct": round(hits_peak / n_sys_val * 100.0, 2),
                    "Hits_Surf10": hits_surf,
                    "Sens_Surf10_ValidTraj_Pct": round(hits_surf / n_sys_val * 100.0, 2),
                    "Hits_DirectPuncture_0mm": hits_dir0,
                    "Sens_DirectPuncture_0mm_Pct": round(hits_dir0 / n_sys_val * 100.0, 2),
                })
                print(f"[{ext} - {mod} Top-{k}]: SysOnly Hits CoM10 = {hits_com}/{n_sys_val} ({sens_com:.2f}%) | TargNeg = {hits_neg}/{n_targ_neg} | TargAbs = {hits_abs}/{n_targ_abs}")

    df_sys = pd.DataFrame(sys_rows)
    df_sys.to_csv(os.path.join(AUDIT_DIR, "SYSTEMATIC_ONLY_RESULTS.csv"), index=False)

    # ==============================================================================
    # 8. INCREMENTAL SPATIAL CORRESPONDENCE & CANDIDATE WORKLOAD (TABLE 5)
    # ==============================================================================
    print("\n" + "-" * 75)
    print("ANALYSIS 8: INCREMENTAL SPATIAL CORRESPONDENCE & CANDIDATE WORKLOAD (TABLE 5)")
    print("-" * 75)

    inc_rows = []
    tot_human_targets = int(np.sum(df_paired["Human_Rad_Mask_Num_Components"]))
    h_hits_val = int(np.sum(paired_cs_val["Human_Sess_CoM_Dist_csPCa_mm"].fillna(999.0) <= 10.0))
    h_sens_val = round(h_hits_val / n_cs_val_denom * 100.0, 2)

    inc_rows.append({
        "Biopsy_Strategy": "Human_3D_Target_Mask_Alone",
        "Total_Targets_Proposed": tot_human_targets,
        "Targets_Per_100_Examinations": round(tot_human_targets / n_paired * 100.0, 1),
        "Additional_AI_Targets_Total": 0,
        "Additional_AI_Targets_Per_100_Exams": 0.0,
        "Geometric_csPCa_Hits": h_hits_val,
        "Geometric_Sensitivity_Pct": h_sens_val,
        "Additional_Geometric_csPCa_Hits_Over_Human": 0,
        "Workload_Extra_Targets_Per_Extra_csPCa_Hit": np.nan,
    })

    configs = [
        ("Human + SegResNet (Official RGA Top-1)", "Official_RGA_Dynamic", "3D_SegResNet", 1),
        ("Human + SegResNet (Official RGA Top-2)", "Official_RGA_Dynamic", "3D_SegResNet", 2),
        ("Human + SegResNet (Official RGA Top-3)", "Official_RGA_Dynamic", "3D_SegResNet", 3),
        ("Human + SegResNet (Shared Fixed-CC Top-1, p>=0.10)", "Shared_Fixed_CC_0.10", "3D_SegResNet", 1),
        ("Human + SegResNet (Shared Fixed-CC Top-2, p>=0.10)", "Shared_Fixed_CC_0.10", "3D_SegResNet", 2),
        ("Human + SegResNet (Shared Fixed-CC Top-3, p>=0.10)", "Shared_Fixed_CC_0.10", "3D_SegResNet", 3),
        ("Human + Pretrained nnU-Net (Official RGA Top-1)", "Official_RGA_Dynamic", "Pretrained_nnUNet", 1),
        ("Human + Pretrained nnU-Net (Official RGA Top-2)", "Official_RGA_Dynamic", "Pretrained_nnUNet", 2),
        ("Human + Pretrained nnU-Net (Official RGA Top-3)", "Official_RGA_Dynamic", "Pretrained_nnUNet", 3),
        ("Human + Pretrained nnU-Net (Shared Fixed-CC Top-1, p>=0.10)", "Shared_Fixed_CC_0.10", "Pretrained_nnUNet", 1),
        ("Human + Pretrained nnU-Net (Shared Fixed-CC Top-3, p>=0.10)", "Shared_Fixed_CC_0.10", "Pretrained_nnUNet", 3),
    ]

    h_hit_sids = set(paired_cs_val[paired_cs_val["Human_Sess_CoM_Dist_csPCa_mm"].fillna(999.0) <= 10.0]["Session_ID"])

    for strat_label, ext_name, mod_name, k_rank in configs:
        sub_c = cands_paired[(cands_paired["Extractor"] == ext_name) & (cands_paired["Model"] == mod_name) & (cands_paired["Candidate_Rank"] <= k_rank)]
        tot_ai_cands = len(sub_c)
        tot_targets = tot_human_targets + tot_ai_cands

        ai_hit_sids = set(sub_c[(sub_c["Session_ID"].isin(set(paired_cs_val["Session_ID"]))) & (sub_c["CoM_Dist_csPCa_mm"] <= 10.0)]["Session_ID"])
        comb_hit_sids = h_hit_sids.union(ai_hit_sids)

        comb_hits = len(comb_hit_sids)
        comb_sens = round(comb_hits / n_cs_val_denom * 100.0, 2)

        extra_hits = comb_hits - h_hits_val
        workload = round(tot_ai_cands / extra_hits, 1) if extra_hits > 0 else np.nan

        inc_rows.append({
            "Biopsy_Strategy": strat_label,
            "Total_Targets_Proposed": tot_targets,
            "Targets_Per_100_Examinations": round(tot_targets / n_paired * 100.0, 1),
            "Additional_AI_Targets_Total": tot_ai_cands,
            "Additional_AI_Targets_Per_100_Exams": round(tot_ai_cands / n_paired * 100.0, 1),
            "Geometric_csPCa_Hits": comb_hits,
            "Geometric_Sensitivity_Pct": comb_sens,
            "Additional_Geometric_csPCa_Hits_Over_Human": extra_hits,
            "Workload_Extra_Targets_Per_Extra_csPCa_Hit": workload,
        })
        print(f"[{strat_label}]: Total Targets = {tot_targets} | Hits = {comb_hits}/{n_cs_val_denom} ({comb_sens}%) | Extra Hits Over Human = +{extra_hits} | Workload = {workload} targets/extra hit")

    df_inc = pd.DataFrame(inc_rows)
    df_inc.to_csv(os.path.join(AUDIT_DIR, "TABLE5_INCREMENTAL_CORRESPONDENCE_AI_PLUS_HUMAN.csv"), index=False)

    # ==============================================================================
    # 9. CLINICAL SUBGROUP STRATIFICATION (TABLE 7) - WITH EXACT ALLOCATION NULL
    # ==============================================================================
    print("\n" + "-" * 75)
    print("ANALYSIS 9: CLINICAL SUBGROUP STRATIFICATION (TABLE 7)")
    print("-" * 75)

    subgroup_rows = []
    paired_cs_val_copy = paired_cs_val.copy()

    # Aggregate measured human target volume tertiles (ONLY for cases with positive volume > 0)
    has_target_mask = (paired_cs_val_copy["Human_Rad_Mask_Vol_cc"] > 0) & paired_cs_val_copy["Human_Rad_Mask_Vol_cc"].notna()
    vol_pos = paired_cs_val_copy.loc[has_target_mask, "Human_Rad_Mask_Vol_cc"]
    t1_cut = float(vol_pos.quantile(1.0/3.0))
    t2_cut = float(vol_pos.quantile(2.0/3.0))

    def get_lesion_vol_cat(row):
        v = row["Human_Rad_Mask_Vol_cc"]
        if pd.isna(v) or v <= 0:
            return "No_Prospective_Target_Placed_or_Unavailable"
        elif v <= t1_cut:
            return f"T1_Small_le_{t1_cut:.2f}cc"
        elif v <= t2_cut:
            return f"T2_Medium_{t1_cut:.2f}_to_{t2_cut:.2f}cc"
        else:
            return f"T3_Large_gt_{t2_cut:.2f}cc"

    paired_cs_val_copy["Lesion_Vol_Tertile"] = paired_cs_val_copy.apply(get_lesion_vol_cat, axis=1)

    g_pos = paired_cs_val_copy["Prostate_Mask_Vol_cc"]
    gt1_cut = float(g_pos.quantile(1.0/3.0))
    gt2_cut = float(g_pos.quantile(2.0/3.0))

    def get_gland_vol_cat(v):
        if v <= gt1_cut:
            return f"T1_Small_le_{gt1_cut:.1f}cc"
        elif v <= gt2_cut:
            return f"T2_Medium_{gt1_cut:.1f}_to_{gt2_cut:.1f}cc"
        else:
            return f"T3_Large_gt_{gt2_cut:.1f}cc"

    paired_cs_val_copy["Gland_Vol_Tertile"] = paired_cs_val_copy["Prostate_Mask_Vol_cc"].apply(get_gland_vol_cat)

    subgroup_definitions = [
        ("ISUP_Grade", "ISUP_2_GG2_(3+4)", paired_cs_val_copy[paired_cs_val_copy["Sess_Max_ISUP"] == 2]),
        ("ISUP_Grade", "ISUP_3_GG3_(4+3)", paired_cs_val_copy[paired_cs_val_copy["Sess_Max_ISUP"] == 3]),
        ("ISUP_Grade", "ISUP_4_to_5_GG4_5", paired_cs_val_copy[paired_cs_val_copy["Sess_Max_ISUP"].isin([4, 5])]),
        ("ISUP_Grade", "ISUP_ge_3_HighGrade_Combined", paired_cs_val_copy[paired_cs_val_copy["Sess_Max_ISUP"] >= 3]),
        ("Human_Target_Aggregate_Volume", f"T1_Small_le_{t1_cut:.2f}cc", paired_cs_val_copy[paired_cs_val_copy["Lesion_Vol_Tertile"].str.startswith("T1")]),
        ("Human_Target_Aggregate_Volume", f"T2_Medium_{t1_cut:.2f}_to_{t2_cut:.2f}cc", paired_cs_val_copy[paired_cs_val_copy["Lesion_Vol_Tertile"].str.startswith("T2")]),
        ("Human_Target_Aggregate_Volume", f"T3_Large_gt_{t2_cut:.2f}cc", paired_cs_val_copy[paired_cs_val_copy["Lesion_Vol_Tertile"].str.startswith("T3")]),
        ("Human_Target_Aggregate_Volume", "No_Prospective_Target_Placed_or_Unavailable", paired_cs_val_copy[paired_cs_val_copy["Lesion_Vol_Tertile"] == "No_Prospective_Target_Placed_or_Unavailable"]),
        ("Prostate_Gland_Volume", f"T1_Small_le_{gt1_cut:.1f}cc", paired_cs_val_copy[paired_cs_val_copy["Gland_Vol_Tertile"].str.startswith("T1")]),
        ("Prostate_Gland_Volume", f"T2_Medium_{gt1_cut:.1f}_to_{gt2_cut:.1f}cc", paired_cs_val_copy[paired_cs_val_copy["Gland_Vol_Tertile"].str.startswith("T2")]),
        ("Prostate_Gland_Volume", f"T3_Large_gt_{gt2_cut:.1f}cc", paired_cs_val_copy[paired_cs_val_copy["Gland_Vol_Tertile"].str.startswith("T3")]),
    ]

    for dim_name, sg_name, sg_sub in subgroup_definitions:
        n_sg = len(sg_sub)
        if n_sg == 0:
            continue
        sg_sids = set(sg_sub["Session_ID"])

        h_hits = int(np.sum(sg_sub["Human_Sess_CoM_Dist_csPCa_mm"].fillna(999.0) <= 10.0))
        h_sens = round(h_hits / n_sg * 100.0, 2)

        sg_cands = cands_paired[(cands_paired["Extractor"] == "Official_RGA_Dynamic") & (cands_paired["Session_ID"].isin(sg_sids))]

        seg1_hits = sg_cands[(sg_cands["Model"] == "3D_SegResNet") & (sg_cands["Candidate_Rank"] == 1) & (sg_cands["CoM_Dist_csPCa_mm"] <= 10.0)]["Session_ID"].nunique()
        seg1_sens = round(seg1_hits / n_sg * 100.0, 2)

        seg3_hits = sg_cands[(sg_cands["Model"] == "3D_SegResNet") & (sg_cands["Candidate_Rank"] <= 3) & (sg_cands["CoM_Dist_csPCa_mm"] <= 10.0)]["Session_ID"].nunique()
        seg3_sens = round(seg3_hits / n_sg * 100.0, 2)

        nnu1_hits = sg_cands[(sg_cands["Model"] == "Pretrained_nnUNet") & (sg_cands["Candidate_Rank"] == 1) & (sg_cands["CoM_Dist_csPCa_mm"] <= 10.0)]["Session_ID"].nunique()
        nnu1_sens = round(nnu1_hits / n_sg * 100.0, 2)

        nnu3_hits = sg_cands[(sg_cands["Model"] == "Pretrained_nnUNet") & (sg_cands["Candidate_Rank"] <= 3) & (sg_cands["CoM_Dist_csPCa_mm"] <= 10.0)]["Session_ID"].nunique()
        nnu3_sens = round(nnu3_hits / n_sg * 100.0, 2)

        # Exact allocation null for subgroup
        m_top3_sg = np.array([m_top3_map.get(sid, 0) for sid in sg_sub["Session_ID"]])
        q10_vals = sg_sub["q_undil_sess_10.0"].values
        null_top3_sens = round(float(np.mean(1.0 - np.power(1.0 - q10_vals, m_top3_sg))) * 100.0, 2)

        subgroup_rows.append({
            "Stratification_Dimension": dim_name,
            "Subgroup": sg_name,
            "N_ValidTraj_csPCa": n_sg,
            "Human_Mask_CoM10_Hits": h_hits,
            "Human_Mask_CoM10_Sens_Pct": h_sens,
            "Human_Mask_CoM10_Wilson95CI": fmt_ci(h_hits, n_sg),
            "SegResNet_Top1_CoM10_Hits": seg1_hits,
            "SegResNet_Top1_CoM10_Sens_Pct": seg1_sens,
            "SegResNet_Top1_CoM10_Wilson95CI": fmt_ci(seg1_hits, n_sg),
            "SegResNet_Top3_CoM10_Hits": seg3_hits,
            "SegResNet_Top3_CoM10_Sens_Pct": seg3_sens,
            "SegResNet_Top3_CoM10_Wilson95CI": fmt_ci(seg3_hits, n_sg),
            "nnUNet_Top1_CoM10_Hits": nnu1_hits,
            "nnUNet_Top1_CoM10_Sens_Pct": nnu1_sens,
            "nnUNet_Top1_CoM10_Wilson95CI": fmt_ci(nnu1_hits, n_sg),
            "nnUNet_Top3_CoM10_Hits": nnu3_hits,
            "nnUNet_Top3_CoM10_Sens_Pct": nnu3_sens,
            "nnUNet_Top3_CoM10_Wilson95CI": fmt_ci(nnu3_hits, n_sg),
            "Analytic_Null_Top3_CoM10_Pct": null_top3_sens,
            "SegResNet_Top3_Excess_Over_Null_Pct": round(seg3_sens - null_top3_sens, 2),
        })
        print(f"[{dim_name} - {sg_name}]: N={n_sg} | Human={h_sens}% | SegTop1={seg1_sens}% | SegTop3={seg3_sens}% | nnUTop3={nnu3_sens}% | NullTop3={null_top3_sens}%")

    df_subgroup = pd.DataFrame(subgroup_rows)
    df_subgroup.to_csv(os.path.join(AUDIT_DIR, "TABLE7_CLINICAL_SUBGROUP_ANALYSIS.csv"), index=False)

    # ==============================================================================
    # 10. HUMAN-AI CONCORDANCE RECONCILIATION (TABLE 3B DYNAMIC CALCULATION)
    # ==============================================================================
    print("\n" + "-" * 75)
    print("ANALYSIS 10: HUMAN-AI SPATIAL CONCORDANCE RECONCILIATION (TABLE 3B)")
    print("-" * 75)

    concordance_rows = []

    # Dynamic calculation across Paired Cohort (N=800)
    g1 = df_paired[(df_paired["Human_Rad_Mask_Vol_cc"] > 0) & (df_paired["Targeted_csPCa_Hit"] == 1)]
    g2 = df_paired[(df_paired["Human_Rad_Mask_Vol_cc"] > 0) & (df_paired["Targeted_Cores_Linked"] > 0) & (df_paired["Targeted_csPCa_Hit"] == 0)]
    g3 = df_paired[(df_paired["Human_Rad_Mask_Vol_cc"] > 0) & (df_paired["Targeted_Cores_Linked"] == 0)]
    g4 = df_paired[(df_paired["Human_Rad_Mask_Vol_cc"] == 0) | df_paired["Human_Rad_Mask_Vol_cc"].isna()]

    groups = [
        ("1_Has_STL_Mask_And_Positive_Targeted_Core", g1),
        ("2_Has_STL_Mask_And_Negative_Targeted_Core", g2),
        ("3_Has_STL_Mask_But_No_TargMRI_BiopsyCore", g3),
        ("4_No_STL_Target_Mask_MissingTargetData", g4),
    ]

    for g_name, g_df in groups:
        n_sess = len(g_df)
        n_cs = int(g_df["New_Sess_Is_csPCa_Independent"].sum())

        g_sids = set(g_df["Session_ID"])
        g_cands = cands_paired[(cands_paired["Extractor"] == "Official_RGA_Dynamic") & (cands_paired["Model"] == "3D_SegResNet") & (cands_paired["Session_ID"].isin(g_sids))]

        # Concordant: has candidate with spatial overlap or near human target (or dice > 0.10)
        # Using master_df SegResNet_Max_Dice_With_Rad if available, or candidate proximity to human CoM
        g_master = master_df[master_df["Session_ID"].isin(g_sids)]
        if "SegResNet_Max_Dice_With_Rad" in g_master.columns:
            n_conc = int(np.sum(g_master["SegResNet_Max_Dice_With_Rad"] >= 0.10))
            mean_dice = float(g_master[g_master["SegResNet_Max_Dice_With_Rad"] >= 0.10]["SegResNet_Max_Dice_With_Rad"].mean()) if n_conc > 0 else np.nan
        else:
            n_conc = 0
            mean_dice = np.nan

        # Sessions with zero AI candidates
        sess_with_cands = set(g_cands["Session_ID"])
        n_zero_cands = n_sess - len(sess_with_cands)

        pct_conc = round(n_conc / n_sess * 100.0, 2) if n_sess > 0 else 0.0
        pct_zero = round(n_zero_cands / n_sess * 100.0, 2) if n_sess > 0 else 0.0

        concordance_rows.append({
            "Patient_Subgroup": g_name,
            "N_Sessions": n_sess,
            "N_Session_Level_csPCa": n_cs,
            "N_AI_Concordant_DiceGt010": n_conc,
            "AI_Concordant_Pct": pct_conc,
            "AI_Concordant_Wilson95CI": fmt_ci(n_conc, n_sess),
            "Mean_Dice_When_Concordant": round(mean_dice, 4) if pd.notna(mean_dice) else np.nan,
            "N_AI_Zero_Candidates_Suppressed": n_zero_cands,
            "AI_Zero_Candidates_Pct": pct_zero,
        })
        print(f"[{g_name}]: N={n_sess} | csPCa={n_cs} | Concordant={n_conc} ({pct_conc}%) | ZeroCands={n_zero_cands} ({pct_zero}%)")

    df_3b = pd.DataFrame(concordance_rows)
    df_3b.to_csv(os.path.join(AUDIT_DIR, "TABLE3B_HUMAN_AI_CONCORDANCE.csv"), index=False)
    print(f"Saved TABLE3B_HUMAN_AI_CONCORDANCE.csv ({len(df_3b)} rows)")

    print("\n" + "=" * 90)
    print(f"STATISTICAL ANALYSIS & CLINICAL AUDIT COMPLETED IN {(time.time()-t0):.2f} SECONDS")
    print("=" * 90)

if __name__ == "__main__":
    run_burden_matched_and_statistical_analysis()

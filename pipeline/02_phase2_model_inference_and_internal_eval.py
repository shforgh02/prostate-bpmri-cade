"""
================================================================================
PHASE 2: MODAL CLOUD 3D SEGRESNET TRAINING, INFERENCE & INTERNAL FOLD-4 AUDIT
================================================================================
"""

import os
import sys
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

# ==============================================================================
# GLOBAL PATHS
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.environ.get(
    "PROSTATE_DATA_ROOT",
    os.path.abspath(os.path.join(PROJECT_DIR, "..", ".."))
)
PICAI_BASE_DIR = os.path.join(DATA_ROOT, "PI-CAI_Validation")
PICAI_LABELS_DIR = os.path.join(PICAI_BASE_DIR, "picai_labels-main")
PICAI_MASTER_CSV = os.path.join(PICAI_BASE_DIR, "Script", "output", "picai_master_clinical_cohort_new.csv")
PICAI_CANDIDATES_CSV = os.path.join(PICAI_BASE_DIR, "Script", "output", "Phase2_Blob_Candidates_Isotropic_New.csv")
PICAI_MODEL_WEIGHTS = os.path.join(PICAI_BASE_DIR, "Script", "output", "picai_segresnet_best_loss_new.pth")

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================================
# PART 1: MODAL CLOUD UPLOAD & TRAINING ARCHITECTURE
# ==============================================================================
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

if MODAL_AVAILABLE:
    app = modal.App("picai-phase2-unified-pipeline")
    volume = modal.Volume.from_name("picai-data-new", create_if_missing=True)
    ucla_volume = modal.Volume.from_name("ucla-New", create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch", "torchvision", "monai", "SimpleITK", "nibabel",
            "pandas", "numpy", "scikit-learn", "scipy", "tqdm"
        )
    )


def get_combo_focal_dice_loss():
    """
    Returns the fixed ComboLoss (50% FocalLoss(gamma=2.0) + 50% DiceLoss)
    used to train the 3D SegResNet architecture on PI-CAI Folds 0-2 (validated on Fold 3).
    """
    import torch.nn as nn
    from monai.losses import DiceLoss, FocalLoss

    class ComboLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.dice = DiceLoss(
                sigmoid=True, squared_pred=True,
                 include_background=True, reduction="mean"
            )
            self.focal = FocalLoss(
                include_background=True, to_onehot_y=False,
                gamma=2.0, reduction="mean"
            )

        def forward(self, pred, target):
            return 0.5 * self.dice(pred, target) + 0.5 * self.focal(pred, target)

    return ComboLoss()


def build_locked_segresnet_model(device=None):
    """
    Instantiates the exact 3D SegResNet architecture used across Phase 2 & 3:
      - spatial_dims=3, init_filters=32, in_channels=3, out_channels=1
      - dropout_prob=0.2, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
    """
    import torch
    from monai.networks.nets import SegResNet

    model = SegResNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=3,
        out_channels=1,
        dropout_prob=0.2,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        norm="instance",
    )
    if device is not None:
        model = model.to(device)
    return model


# ==============================================================================
# PART 2: STATISTICAL HELPERS (ZERO-EXTERNAL-DEPENDENCY NUMPY/SCIPY IMPLEMENTATIONS)
# ==============================================================================
def compute_auroc(y_true, y_score):
    """Standard scikit-learn AUROC calculation."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def compute_pr_auc(y_true, y_score):
    """Standard scikit-learn Average Precision (AP / PR-AUC) calculation."""
    from sklearn.metrics import average_precision_score
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if np.sum(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE) across uniform probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b_low, b_high in zip(bins[:-1], bins[1:]):
        if b_high == 1.0:
            mask = (y_prob >= b_low) & (y_prob <= b_high)
        else:
            mask = (y_prob >= b_low) & (y_prob < b_high)
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def bootstrap_patient_metrics(y_true, y_prob, n_boot=1000, seed=42):
    """Computes 95% percentile bootstrap CIs for AUROC, PR-AUC, and Brier Score."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    aurocs, praucs, briers = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(compute_auroc(yt, yp))
        praucs.append(compute_pr_auc(yt, yp))
        briers.append(float(np.mean((yp - yt) ** 2)))
    return {
        "AUROC_CI_Low": float(np.percentile(aurocs, 2.5)),
        "AUROC_CI_High": float(np.percentile(aurocs, 97.5)),
        "PRAUC_CI_Low": float(np.percentile(praucs, 2.5)),
        "PRAUC_CI_High": float(np.percentile(praucs, 97.5)),
        "Brier_CI_Low": float(np.percentile(briers, 2.5)),
        "Brier_CI_High": float(np.percentile(briers, 97.5)),
    }


# ==============================================================================
# PART 3: INTERNAL PI-CAI FOLD 4 COMPREHENSIVE EVALUATION (CRITIQUES #6 & #9)
# ==============================================================================
def run_local_internal_fold4_evaluation():
    t0 = time.time()
    print("=" * 80)
    print("PHASE 2: INTERNAL PI-CAI FOLD 4 EVALUATION & PROVENANCE AUDIT")
    print("=" * 80)

    if not os.path.exists(PICAI_MASTER_CSV):
        raise FileNotFoundError(f"Missing PI-CAI master CSV: {PICAI_MASTER_CSV}")
    if not os.path.exists(PICAI_CANDIDATES_CSV):
        raise FileNotFoundError(f"Missing PI-CAI candidates CSV: {PICAI_CANDIDATES_CSV}")

    df_master = pd.read_csv(PICAI_MASTER_CSV)
    df_cand_all = pd.read_csv(PICAI_CANDIDATES_CSV)

    # Filter both master and candidate tables strictly to Internal Test Fold 4
    df_f4 = df_master[df_master["Fold"] == "picai_public_images_fold4"].copy()
    df_cand = df_cand_all[df_cand_all["Fold"] == "picai_public_images_fold4"].copy()
    print(f"Loaded PI-CAI Fold 4: {len(df_f4)} sessions across {df_f4['Patient_ID'].nunique()} unique patients ({len(df_cand)} extracted 3D candidates).")

    # Map exact annotation provenance (Human Expert vs AI_Bosma22a vs None)
    human_res_dir = os.path.join(PICAI_LABELS_DIR, "csPCa_lesion_delineations", "human_expert", "resampled")
    human_orig_dir = os.path.join(PICAI_LABELS_DIR, "csPCa_lesion_delineations", "human_expert", "original")
    ai_dir = os.path.join(PICAI_LABELS_DIR, "csPCa_lesion_delineations", "AI", "Bosma22a")

    provenance_list = []
    for _, row in df_f4.iterrows():
        sid = str(row["Session_ID"])
        has_ann = int(row["Has_Annotation"])
        is_cspca = int(row["Clinical_csPCa"])
        if has_ann == 0:
            prov = "Unannotated_csPCa" if is_cspca == 1 else "Benign_ZeroMask"
        else:
            if os.path.exists(os.path.join(human_res_dir, f"{sid}.nii.gz")):
                prov = "Human_Expert_Resampled"
            elif os.path.exists(os.path.join(human_orig_dir, f"{sid}.nii.gz")):
                prov = "Human_Expert_Original"
            elif os.path.exists(os.path.join(ai_dir, f"{sid}.nii.gz")):
                prov = "AI_Bosma22a"
            else:
                prov = "Other_Annotation"
        provenance_list.append(prov)
    df_f4["Annotation_Provenance"] = provenance_list

    # Sort candidates by Session_ID and Max_Prob descending and assign Candidate_Rank
    df_cand = df_cand.sort_values(["Session_ID", "Max_Prob"], ascending=[True, False]).copy()
    df_cand["Candidate_Rank"] = df_cand.groupby("Session_ID").cumcount() + 1

    # Merge Session-level provenance into Candidate table
    df_cand_master = df_cand.merge(
        df_f4[["Session_ID", "Clinical_csPCa", "Annotation_Provenance", "Has_Annotation", "PSA", "MRI_Prostate_Vol_cc", "Target_Vol_cc"]],
        on="Session_ID",
        how="left",
    )

    # Save Candidate-Level Master CSV for Internal Fold 4
    cand_out_path = os.path.join(TABLES_DIR, "Phase2_Internal_Fold4_Candidate_Master.csv")
    df_cand_master.to_csv(cand_out_path, index=False)
    print(f"Saved Internal Fold 4 Candidate Master CSV ({len(df_cand_master)} blobs) -> {cand_out_path}")

    # Build Patient-Level Master Table for Internal Fold 4
    patient_rows = []
    for _, row in df_f4.iterrows():
        sid = row["Session_ID"]
        sub = df_cand[df_cand["Session_ID"] == sid]
        n_blobs = len(sub)
        max_prob = float(sub["Max_Prob"].max()) if n_blobs > 0 else 0.0

        # Top-1, Top-2, Top-3 hit flags and Dice scores
        top1_sub = sub.head(1)
        top2_sub = sub.head(2)
        top3_sub = sub.head(3)

        top1_hit = int((top1_sub["Is_True_Hit"] == 1).any()) if n_blobs > 0 else 0
        top2_hit = int((top2_sub["Is_True_Hit"] == 1).any()) if n_blobs > 0 else 0
        top3_hit = int((top3_sub["Is_True_Hit"] == 1).any()) if n_blobs > 0 else 0
        any_hit = int((sub["Is_True_Hit"] == 1).any()) if n_blobs > 0 else 0

        top1_dice = float(top1_sub["Dice_Score"].max()) if n_blobs > 0 else 0.0
        best_tp_dice = float(sub[sub["Is_True_Hit"] == 1]["Dice_Score"].max()) if any_hit else 0.0
        top1_vol_cc = float(top1_sub["Blob_Vol_cc"].iloc[0]) if n_blobs > 0 else 0.0

        patient_rows.append({
            "Session_ID": sid,
            "Patient_ID": row["Patient_ID"],
            "Is_csPCa": int(row["Clinical_csPCa"]),
            "Has_Annotation": int(row["Has_Annotation"]),
            "Annotation_Provenance": row["Annotation_Provenance"],
            "PSA": row["PSA"],
            "MRI_Prostate_Vol_cc": row["MRI_Prostate_Vol_cc"],
            "Target_Vol_cc": row["Target_Vol_cc"],
            "Num_Extracted_Blobs": n_blobs,
            "Patient_Max_Prob": max_prob,
            "Top1_Blob_Vol_cc": top1_vol_cc,
            "Top1_Hit": top1_hit,
            "Top2_Hit": top2_hit,
            "Top3_Hit": top3_hit,
            "Any_Candidate_Hit": any_hit,
            "Top1_Dice": top1_dice,
            "Best_TP_Candidate_Dice": best_tp_dice,
            "Unconditional_Lesion_Dice": best_tp_dice if int(row["Clinical_csPCa"]) == 1 else np.nan,
        })

    df_pat_f4 = pd.DataFrame(patient_rows)
    pat_out_path = os.path.join(TABLES_DIR, "Phase2_Internal_Fold4_Patient_Master.csv")
    df_pat_f4.to_csv(pat_out_path, index=False)
    print(f"Saved Internal Fold 4 Patient Master CSV ({len(df_pat_f4)} patients) -> {pat_out_path}")

    # ==========================================================================
    # COMPUTE SUMMARY METRICS ACROSS SUBCOHORTS
    # ==========================================================================
    subcohorts = [
        ("Full_Fold4_All_Patients", df_pat_f4),
        ("Human_Expert_csPCa_Plus_Benign", df_pat_f4[df_pat_f4["Annotation_Provenance"].isin(["Human_Expert_Resampled", "Human_Expert_Original", "Benign_ZeroMask"])]),
        ("Human_Plus_Unannotated_csPCa_Excl_AI", df_pat_f4[df_pat_f4["Annotation_Provenance"] != "AI_Bosma22a"]),
        ("AI_Bosma22a_csPCa_Plus_Benign", df_pat_f4[df_pat_f4["Annotation_Provenance"].isin(["AI_Bosma22a", "Benign_ZeroMask"])]),
    ]

    summary_rows = []
    for cohort_name, df_sub in subcohorts:
        y_true = df_sub["Is_csPCa"].values
        y_prob = df_sub["Patient_Max_Prob"].values
        n_total = len(df_sub)
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        prev = n_pos / n_total

        auroc = compute_auroc(y_true, y_prob)
        prauc = compute_pr_auc(y_true, y_prob)
        brier = float(np.mean((y_prob - y_true) ** 2))
        brier_null = float(prev * (1.0 - prev))  # Constant-prevalence predictor baseline
        brier_skill_score = 1.0 - (brier / brier_null)
        ece = compute_ece(y_true, y_prob, n_bins=10)
        mean_pred_prob = float(np.mean(y_prob))
        calib_in_large = mean_pred_prob - prev

        boot = bootstrap_patient_metrics(y_true, y_prob, n_boot=1000, seed=42)

        # Lesion-level & Conditional vs Unconditional Dice
        pos_sub = df_sub[df_sub["Is_csPCa"] == 1]
        ann_pos_sub = pos_sub[pos_sub["Has_Annotation"] == 1]
        n_ann_pos = len(ann_pos_sub)

        top1_sens_all_cspca = float(pos_sub["Top1_Hit"].mean()) if n_pos > 0 else np.nan
        top2_sens_all_cspca = float(pos_sub["Top2_Hit"].mean()) if n_pos > 0 else np.nan
        top3_sens_all_cspca = float(pos_sub["Top3_Hit"].mean()) if n_pos > 0 else np.nan

        top1_sens_ann_cspca = float(ann_pos_sub["Top1_Hit"].mean()) if n_ann_pos > 0 else np.nan
        top2_sens_ann_cspca = float(ann_pos_sub["Top2_Hit"].mean()) if n_ann_pos > 0 else np.nan
        top3_sens_ann_cspca = float(ann_pos_sub["Top3_Hit"].mean()) if n_ann_pos > 0 else np.nan

        # Conditional Dice on Top-1 TP hits
        top1_tp_dices = ann_pos_sub[ann_pos_sub["Top1_Hit"] == 1]["Top1_Dice"].values
        cond_top1_dice_mean = float(np.mean(top1_tp_dices)) if len(top1_tp_dices) > 0 else np.nan
        cond_top1_dice_sd = float(np.std(top1_tp_dices, ddof=1)) if len(top1_tp_dices) > 1 else np.nan

        # Conditional Dice across all TP blobs in this subcohort
        sub_sids = set(df_sub["Session_ID"])
        cand_sub_tp = df_cand[(df_cand["Session_ID"].isin(sub_sids)) & (df_cand["Is_True_Hit"] == 1)]
        cond_all_blob_dice_mean = float(cand_sub_tp["Dice_Score"].mean()) if len(cand_sub_tp) > 0 else np.nan
        cond_all_blob_dice_sd = float(cand_sub_tp["Dice_Score"].std(ddof=1)) if len(cand_sub_tp) > 1 else np.nan

        # Unconditional Dice across ALL annotated csPCa patients (assigning 0 to missed lesions)
        uncond_ann_dices = ann_pos_sub["Best_TP_Candidate_Dice"].values
        uncond_ann_dice_mean = float(np.mean(uncond_ann_dices)) if len(uncond_ann_dices) > 0 else np.nan
        uncond_ann_dice_sd = float(np.std(uncond_ann_dices, ddof=1)) if len(uncond_ann_dices) > 1 else np.nan

        # Unconditional Dice across ALL csPCa patients (including 3 unannotated csPCa)
        uncond_all_dices = pos_sub["Best_TP_Candidate_Dice"].values
        uncond_all_dice_mean = float(np.mean(uncond_all_dices)) if len(uncond_all_dices) > 0 else np.nan
        uncond_all_dice_sd = float(np.std(uncond_all_dices, ddof=1)) if len(uncond_all_dices) > 1 else np.nan

        # Patient-level specificity at threshold 0.15, 0.25, 0.35, 0.50
        neg_sub = df_sub[df_sub["Is_csPCa"] == 0]
        spec_015 = float(np.mean(neg_sub["Patient_Max_Prob"] < 0.15))
        spec_025 = float(np.mean(neg_sub["Patient_Max_Prob"] < 0.25))
        spec_035 = float(np.mean(neg_sub["Patient_Max_Prob"] < 0.35))
        spec_050 = float(np.mean(neg_sub["Patient_Max_Prob"] < 0.50))

        # Mean false positive blobs per patient at threshold 0.15
        cand_sub = df_cand[df_cand["Session_ID"].isin(sub_sids)]
        fp_per_pat_015 = float(np.sum(cand_sub["Is_True_Hit"] == 0) / n_total)

        summary_rows.append({
            "Subcohort": cohort_name,
            "N_Patients": n_total,
            "N_csPCa_Total": n_pos,
            "N_csPCa_Annotated": n_ann_pos,
            "N_Benign": n_neg,
            "Prevalence_NoSkill_PRAUC": round(prev, 4),
            "AUROC": round(auroc, 4),
            "AUROC_95CI": f"{boot['AUROC_CI_Low']:.4f}-{boot['AUROC_CI_High']:.4f}",
            "PR_AUC": round(prauc, 4),
            "PR_AUC_95CI": f"{boot['PRAUC_CI_Low']:.4f}-{boot['PRAUC_CI_High']:.4f}",
            "Brier_Score": round(brier, 4),
            "Brier_95CI": f"{boot['Brier_CI_Low']:.4f}-{boot['Brier_CI_High']:.4f}",
            "Null_Prevalence_Brier": round(brier_null, 4),
            "Brier_Skill_Score": round(brier_skill_score, 4),
            "ECE_10Bin": round(ece, 4),
            "Mean_Pred_Prob": round(mean_pred_prob, 4),
            "Calibration_In_The_Large": round(calib_in_large, 4),
            "Top1_Sens_All_csPCa": round(top1_sens_all_cspca, 4),
            "Top2_Sens_All_csPCa": round(top2_sens_all_cspca, 4),
            "Top3_Sens_All_csPCa": round(top3_sens_all_cspca, 4),
            "Top1_Sens_Annotated_csPCa": round(top1_sens_ann_cspca, 4),
            "Top2_Sens_Annotated_csPCa": round(top2_sens_ann_cspca, 4),
            "Top3_Sens_Annotated_csPCa": round(top3_sens_ann_cspca, 4),
            "Conditional_Top1_Dice_Mean": round(cond_top1_dice_mean, 4),
            "Conditional_Top1_Dice_SD": round(cond_top1_dice_sd, 4),
            "Conditional_AllTPBlobs_Dice_Mean": round(cond_all_blob_dice_mean, 4),
            "Conditional_AllTPBlobs_Dice_SD": round(cond_all_blob_dice_sd, 4),
            "Unconditional_Annotated_csPCa_Dice_Mean": round(uncond_ann_dice_mean, 4),
            "Unconditional_Annotated_csPCa_Dice_SD": round(uncond_ann_dice_sd, 4),
            "Unconditional_All_csPCa_Dice_Mean": round(uncond_all_dice_mean, 4),
            "Unconditional_All_csPCa_Dice_SD": round(uncond_all_dice_sd, 4),
            "Specificity_at_0.15": round(spec_015, 4),
            "Specificity_at_0.25": round(spec_025, 4),
            "Specificity_at_0.35": round(spec_035, 4),
            "Specificity_at_0.50": round(spec_050, 4),
            "Mean_FP_Blobs_Per_Patient_at_0.15": round(fp_per_pat_015, 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    sum_out_path = os.path.join(TABLES_DIR, "Phase2_Internal_Fold4_Metrics_Summary.csv")
    df_summary.to_csv(sum_out_path, index=False)
    print("\n--- INTERNAL PI-CAI FOLD 4 SUMMARY (CRITIQUES #6 & #9) ---")
    print(df_summary[[
        "Subcohort", "N_Patients", "N_csPCa_Total", "AUROC", "AUROC_95CI",
        "PR_AUC", "Prevalence_NoSkill_PRAUC", "Brier_Score", "Null_Prevalence_Brier",
        "Conditional_Top1_Dice_Mean", "Unconditional_Annotated_csPCa_Dice_Mean"
    ]].to_string(index=False))

    # Generate FROC & Threshold Operating Table on Internal Fold 4
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    thresh_rows = []
    for th in thresholds:
        sub_c = df_cand[df_cand["Max_Prob"] >= th]
        tp_sids = set(sub_c[sub_c["Is_True_Hit"] == 1]["Session_ID"])
        fp_count = int(np.sum(sub_c["Is_True_Hit"] == 0))
        tp_count = int(np.sum(sub_c["Is_True_Hit"] == 1))

        sens_all_90 = len(tp_sids) / 90.0
        sens_ann_87 = len(tp_sids) / 87.0
        fp_per_pat = fp_count / 296.0
        ppv_blob = tp_count / len(sub_c) if len(sub_c) > 0 else np.nan

        pat_pos_pred = df_pat_f4["Patient_Max_Prob"] >= th
        pat_tp = int(np.sum((pat_pos_pred) & (df_pat_f4["Is_csPCa"] == 1)))
        pat_fp = int(np.sum((pat_pos_pred) & (df_pat_f4["Is_csPCa"] == 0)))
        pat_tn = int(np.sum((~pat_pos_pred) & (df_pat_f4["Is_csPCa"] == 0)))
        pat_fn = int(np.sum((~pat_pos_pred) & (df_pat_f4["Is_csPCa"] == 1)))

        thresh_rows.append({
            "Threshold": th,
            "Total_Blobs": len(sub_c),
            "TP_Blobs": tp_count,
            "FP_Blobs": fp_count,
            "Blob_PPV": round(ppv_blob, 4) if not np.isnan(ppv_blob) else np.nan,
            "FP_Blobs_Per_Patient": round(fp_per_pat, 4),
            "Lesion_Hit_Sens_All90": round(sens_all_90, 4),
            "Lesion_Hit_Sens_Ann87": round(sens_ann_87, 4),
            "Patient_Class_Sens_All90": round(pat_tp / 90.0, 4),
            "Patient_Class_Spec_206Neg": round(pat_tn / 206.0, 4),
            "Patient_Class_PPV": round(pat_tp / (pat_tp + pat_fp), 4) if (pat_tp + pat_fp) > 0 else np.nan,
            "Patient_Class_NPV": round(pat_tn / (pat_tn + pat_fn), 4) if (pat_tn + pat_fn) > 0 else np.nan,
        })

    df_thresh = pd.DataFrame(thresh_rows)
    thresh_out_path = os.path.join(TABLES_DIR, "Phase2_Internal_Fold4_Threshold_Sweep.csv")
    df_thresh.to_csv(thresh_out_path, index=False)
    print(f"\nSaved Internal Fold 4 Threshold Sweep -> {thresh_out_path}")
    print(df_thresh.to_string(index=False))

    # Also compute Candidate/Blob-Level Discrimination on Internal Fold 4 (N=815 blobs)
    yb_true = df_cand["Is_True_Hit"].values
    yb_prob = df_cand["Max_Prob"].values
    b_auroc = compute_auroc(yb_true, yb_prob)
    b_prauc = compute_pr_auc(yb_true, yb_prob)
    b_brier = float(np.mean((yb_prob - yb_true) ** 2))
    b_prev = float(np.mean(yb_true))
    b_null = float(b_prev * (1.0 - b_prev))
    b_boot = bootstrap_patient_metrics(yb_true, yb_prob, n_boot=1000, seed=42)
    print(
        f"\n[Candidate-Level Discrimination (N={len(df_cand)} blobs, TP={int(np.sum(yb_true))})]: "
        f"AUROC={b_auroc:.4f} ({b_boot['AUROC_CI_Low']:.4f}-{b_boot['AUROC_CI_High']:.4f}), "
        f"PR-AUC={b_prauc:.4f} ({b_boot['PRAUC_CI_Low']:.4f}-{b_boot['PRAUC_CI_High']:.4f}) [No-skill={b_prev:.4f}], "
        f"Brier={b_brier:.4f} [Null={b_null:.4f}]"
    )

    # Parse and export 101-epoch learning curve from `phase 1 logs.txt` (Train Folds 0-2, Val Fold 3)
    parse_and_plot_learning_curves()

    # Plot Internal Fold 4 ROC, PR, Calibration & Human-vs-AI Provenance Comparison
    plot_internal_fold4_diagnostics(df_pat_f4, df_summary, df_thresh)
    print(f"\nPhase 2 Internal Fold 4 Evaluation completed in {time.time() - t0:.2f}s.")
    return df_summary, df_thresh


def parse_and_plot_learning_curves():
    """Parses `phase 1 logs.txt` from Final Script and exports CSV + Learning Curves plot."""
    import re
    candidates = [
        os.path.join(os.path.dirname(PROJECT_DIR), "phase 1 logs.txt"),
        os.path.join(PROJECT_DIR, "phase 1 logs.txt"),
        os.path.join(DATA_ROOT, "Final Script", "phase 1 logs.txt"),
    ]
    log_file = None
    for c in candidates:
        if os.path.exists(c):
            log_file = c
            break
    if not log_file or not os.path.exists(log_file):
        return
    regex = r"Epoch (\d+)/\d+ \| LR: ([0-9.e+-]+) \| Train Loss: ([0-9.]+)(?: \| Val Loss: ([0-9.]+) \| Val Dice: ([0-9.]+))?"
    rows = []
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(regex, line)
            if m:
                rows.append({
                    "Epoch": int(m.group(1)),
                    "LR": float(m.group(2)),
                    "Train_Loss": float(m.group(3)),
                    "Val_Loss": float(m.group(4)) if m.group(4) is not None else np.nan,
                    "Val_Dice": float(m.group(5)) if m.group(5) is not None else np.nan,
                })
    if not rows:
        return
    df_lc = pd.DataFrame(rows)
    lc_csv = os.path.join(TABLES_DIR, "Phase2_Training_Learning_Curves.csv")
    df_lc.to_csv(lc_csv, index=False)

    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax1.plot(df_lc["Epoch"], df_lc["Train_Loss"], "b-", lw=2, label="Train Loss (Folds 0-2, Combo Dice+Focal)")
    val_sub = df_lc.dropna(subset=["Val_Loss"])
    if not val_sub.empty:
        ax1.plot(val_sub["Epoch"], val_sub["Val_Loss"], "r--", lw=2, label="Validation Loss (Fold 3)")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Combo Loss", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)

    if not val_sub.empty:
        ax2 = ax1.twinx()
        ax2.plot(val_sub["Epoch"], val_sub["Val_Dice"], "g-", lw=2.2, label="Validation Lesion Dice (Fold 3)")
        ax2.set_ylabel("Validation Dice Score (Fold 3)", color="green", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.set_ylim(0, 0.60)
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="center right")
    else:
        ax1.legend(loc="upper right")

    plt.title(
        f"3D SegResNet Training (Folds 0-2) & Validation (Fold 3) Trajectory ({len(df_lc)}/200 Epochs, Early Stopped)",
        fontweight="bold",
    )
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "Phase2_Learning_Curves.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved Training Trajectory CSV ({len(df_lc)} epochs) -> {lc_csv} and figure -> {fig_path}")


def plot_internal_fold4_diagnostics(df_pat_f4, df_summary, df_thresh):
    """Generates publication-grade diagnostic plots for Internal PI-CAI Fold 4."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Panel A: Provenance-stratified AUROC & PR-AUC
    ax = axes[0]
    labels = ["All Fold 4\n(N=296)", "Human Expert\nOnly (N=252)", "Excl. AI Masks\n(N=255)", "AI Bosma22a\nOnly (N=247)"]
    aurocs = df_summary["AUROC"].values
    praucs = df_summary["PR_AUC"].values
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, aurocs, width=w, label="AUROC", color="#1f77b4", edgecolor="black")
    ax.bar(x + w/2, praucs, width=w, label="PR-AUC", color="#ff7f0e", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric Score", fontweight="bold")
    ax.set_title("A. Internal Fold 4: Label Provenance Audit", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: Reliability / Calibration Plot on Full Fold 4
    ax = axes[1]
    y_true = df_pat_f4["Is_csPCa"].values
    y_prob = df_pat_f4["Patient_Max_Prob"].values
    bins = np.linspace(0.0, 1.0, 8)
    bin_means, bin_trues, bin_counts = [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (y_prob >= b0) & (y_prob <= b1 if b1 == 1.0 else y_prob < b1)
        if np.sum(m) > 0:
            bin_means.append(np.mean(y_prob[m]))
            bin_trues.append(np.mean(y_true[m]))
            bin_counts.append(np.sum(m))
    ax.plot([0, 1], [0, 1], "k--", label="Ideal Calibration")
    ax.plot(bin_means, bin_trues, "o-", color="#d62728", lw=2, markersize=7, label="3D SegResNet (Raw Prob)")
    ax.axhline(np.mean(y_true), color="gray", linestyle=":", label=f"Prevalence ({np.mean(y_true):.2f})")
    ax.set_xlabel("Predicted Patient Max Probability", fontweight="bold")
    ax.set_ylabel("Observed csPCa Frequency", fontweight="bold")
    ax.set_title(
        f"B. Internal Calibration (Brier={df_summary.iloc[0]['Brier_Score']:.3f} vs Null={df_summary.iloc[0]['Null_Prevalence_Brier']:.3f})",
        fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel C: FROC Curve (Lesion Sensitivity vs FP/patient)
    ax = axes[2]
    ax.plot(
        df_thresh["FP_Blobs_Per_Patient"], df_thresh["Lesion_Hit_Sens_All90"] * 100,
        "s-", color="#2ca02c", lw=2, label="All csPCa (N=90)"
    )
    ax.plot(
        df_thresh["FP_Blobs_Per_Patient"], df_thresh["Lesion_Hit_Sens_Ann87"] * 100,
        "o--", color="#9467bd", lw=2, label="Annotated csPCa (N=87)"
    )
    ax.set_xlabel("Mean False Positive Blobs per Patient", fontweight="bold")
    ax.set_ylabel("Lesion Detection Sensitivity (%)", fontweight="bold")
    ax.set_title("C. Internal Fold 4 FROC Curve", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "Phase2_Internal_Fold4_Audit_Panel.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved Internal Fold 4 Diagnostic Figure -> {fig_path}")


if __name__ == "__main__":
    run_local_internal_fold4_evaluation()

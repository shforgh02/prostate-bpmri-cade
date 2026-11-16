# Frozen ProViCNet features sampled along every tracked biopsy needle (no training on Modal).
# For each exam: patch tokens of the T2, ADC and high-b models (32 x 32 patches per slice, 384 numbers each) and the fusion
# model's probabilities; for each core: mean and max of the tokens over 21 points along the needle (2304 numbers) and the
# model's own csPCa / any-cancer probability along the needle.
# Workspace <your Modal profile>, volume ucla-provicnet-vol:  inputs/ (already there) + cores/cores.csv  ->  needle_feats/<case>.npz
#
#   (once)  python analysis/C2_needle_probe.py --make-cores          then
#           MODAL_PROFILE=<profile> python analysis/_modal_sync.py put ucla-provicnet-vol outputs/cores_for_modal.csv cores/cores.csv
#   MODAL_PROFILE=<profile> PYTHONIOENCODING=utf-8 modal run analysis/B8_modal_provicnet_needle_features.py --mode pilot
#   MODAL_PROFILE=<profile> PYTHONIOENCODING=utf-8 modal run --detach analysis/B8_modal_provicnet_needle_features.py --mode run
#
# "pilot" processes 5 exams and prints the orientation check: how well tokens predict the mean T2 intensity of each patch (R2) when the
# 32 x 32 token grid is read as (row = y, column = x) versus transposed. The first must be clearly higher (for example 0.9 against 0.5 or less
# against clearly less). If not, STOP: the mapping from needle position to token is wrong.
import modal

COMMIT = "1b016ac90e4eb57f34114b33914f21ff079894f9"
app = modal.App("ucla-provicnet-needle")
vol = modal.Volume.from_name("ucla-provicnet-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.0.1", "torchvision==0.15.2", "timm==1.0.3", "einops==0.8.0", "safetensors==0.4.3", "SimpleITK==2.3.1", "nibabel==5.1.0",
                 "monai==1.3.1", "opencv-python-headless==4.9.0.80", "numpy==1.24.4", "pandas==2.0.3", "scikit-learn==1.3.2", "umap-learn==0.5.6",
                 "matplotlib==3.7.5", "tqdm", "huggingface-hub==0.26.2", "albumentations==1.4.7", "PyYAML", "icecream", "scipy")
    .run_commands(f"git clone https://github.com/pimed/ProViCNet /opt/ProViCNet && cd /opt/ProViCNet && git checkout {COMMIT}")
)


@app.function(image=image, gpu="T4", cpu=4, memory=16384, timeout=8 * 3600, volumes={"/data": vol})
def run(mode: str = "pilot", limit: int = 0):
    import os, sys, glob, time, types
    import numpy as np, pandas as pd, torch, SimpleITK as sitk
    from huggingface_hub import hf_hub_download
    os.chdir("/opt/ProViCNet")
    sys.path.insert(0, "/opt/ProViCNet")
    from ProViCNet.ModelArchitectures.Models import GetModel
    from ProViCNet.ModelArchitectures.ProViCNet.ProViCNet import FusionModalities, load_partial_weights
    from ProViCNet.util_functions.Prostate_DataGenerator import US_MRI_Generator
    from ProViCNet.util_functions.utils_weighted import set_seed
    from ProViCNet.util_functions.inference import ProViCNet_data_preparation
    from ProViCNet.util_functions.train_functions import getPatchTokens

    args = types.SimpleNamespace(ModelName="ProViCNet", vit_backbone="dinov2_s_reg", img_size=448, nClass=4, nChannel=3, contrastive=True, cuda_device=0,
                                 only_csPCa=True, threshold=0.4, small_batchsize=16, device=torch.device("cuda:0"))
    set_seed(42)

    def weights(name):
        return torch.load(hf_hub_download(repo_id="pimed/ProViCNet", filename=name), map_location=args.device)

    MODELs = {}
    for seq in ("T2", "ADC", "DWI"):
        m = GetModel(args.ModelName, args.nClass, args.nChannel, args.img_size, vit_backbone=args.vit_backbone, contrastive=args.contrastive).to(args.device)
        m.load_state_dict(weights(f"{seq}_best.pth"), strict=True)
        m.eval()
        MODELs[seq] = m
    fusion = FusionModalities(embedding_size=MODELs["T2"].embedding_size, target_channels=128, num_classes=args.nClass)
    fusion.load_state_dict(weights("mpMRI_Fusion_best.pth"), strict=True)
    fusion = fusion.to(args.device)
    load_partial_weights(MODELs["T2"].head.segmentation_conv, fusion.Stream_T2W.UpsampleModule)
    load_partial_weights(MODELs["ADC"].head.segmentation_conv, fusion.Stream_ADC.UpsampleModule)
    load_partial_weights(MODELs["DWI"].head.segmentation_conv, fusion.Stream_DWI.UpsampleModule)
    fusion.eval()
    cores = pd.read_csv("/data/cores/cores.csv")
    CG = {s: g for s, g in cores.groupby("Session_ID")}
    print("models loaded | cores", len(cores), "| sessions with cores", len(CG), flush=True)
    os.makedirs("/data/needle_feats", exist_ok=True)
    cases = sorted(os.path.basename(f)[:-12] for f in glob.glob("/data/inputs/*_0000.nii.gz"))
    todo = [c for c in cases if c[5:] in CG and not os.path.exists(f"/data/needle_feats/{c}.npz")]
    if mode == "pilot":
        todo = [c for c in cases if c[5:] in CG][:5]
    elif limit:
        todo = todo[:limit]
    print("cases", len(cases), "| to run", len(todo), flush=True)
    fr = np.linspace(0, 1, 21)
    t0 = time.time()
    fails = 0
    for i, c in enumerate(todo):
        try:
            b = f"/data/inputs/{c}"
            ref = sitk.ReadImage(b + "_0000.nii.gz")
            W, H, N = ref.GetSize()
            z = sitk.Image(ref.GetSize(), sitk.sitkUInt8)
            z.CopyInformation(ref)
            sitk.WriteImage(z, "/tmp/zero_cancer.nii.gz")
            gens = {s: US_MRI_Generator(imageFileName=[p], glandFileName=[b + "_mask.nii.gz"], cancerFileName=["/tmp/zero_cancer.nii.gz"], modality="MRI",
                                        cancerTo2=False, Augmentation=False, img_size=args.img_size)
                    for s, p in (("T2", b + "_0000.nii.gz"), ("ADC", b + "_0001.nii.gz"), ("DWI", b + "_0002.nii.gz"))}
            prep = ProViCNet_data_preparation(0, args, gens, modality="MRI")
            if prep is False:
                raise RuntimeError("data preparation failed")
            I2, IA, ID, posit, label = prep
            with torch.no_grad():
                T = {s: getPatchTokens(MODELs[s], im, posit, args).to(args.device) for s, im in (("T2", I2), ("ADC", IA), ("DWI", ID))}
                sm = torch.softmax(fusion(T["T2"], T["ADC"], T["DWI"]).cpu(), dim=1).numpy()        # (slices, 4, h, w): background, gland, indolent, csPCa
            tok = {s: v.float().cpu().numpy() for s, v in T.items()}                                  # (slices, patches, 384)
            n_s, n_p, emb = tok["T2"].shape
            g = int(round(n_p ** 0.5))
            if n_s != N or g * g != n_p:
                raise RuntimeError(f"unexpected token shape {tok['T2'].shape} for image {(W, H, N)}")
            grid = {s: v.reshape(n_s, g, g, emb) for s, v in tok.items()}                             # [slice, row, col, emb]
            p_cs, p_any = sm[:, 3], sm[:, 2] + sm[:, 3]
            h, w = p_cs.shape[1:]
            if mode == "pilot":                                                                       # orientation check
                # Target: mean T2 intensity of each 14 x 14 input patch (the pelvis is not symmetric under transposition,
                # unlike the prostate mask). Tokens must predict it much better in the correct layout than transposed.
                from sklearn.linear_model import Ridge
                from sklearn.metrics import r2_score
                img = I2[:, 1].float().cpu().numpy()                                                  # (slices, 448, 448), rows = y, columns = x
                ps = img.shape[1] // g
                tgt = img.reshape(N, g, ps, g, ps).mean(axis=(2, 4))                                  # [slice, row, col]
                ev, od = np.arange(N) % 2 == 0, np.arange(N) % 2 == 1
                res = []
                for name, G in (("row=y,col=x", grid["T2"]), ("transposed", grid["T2"].transpose(0, 2, 1, 3))):
                    X0, X1 = G[ev].reshape(-1, emb), G[od].reshape(-1, emb)
                    res.append(f"{name} R2 {r2_score(tgt[od].ravel(), Ridge(1.0).fit(X0, tgt[ev].ravel()).predict(X1)):.3f}")
                print("orientation check", c[-10:], "|", " | ".join(res), "| token grid", (n_s, g, g, emb), "| prob map", sm.shape, flush=True)
            cg = CG[c[5:]]
            A = cg[["bx", "by", "bz"]].to_numpy(float)
            B = cg[["tx", "ty", "tz"]].to_numpy(float)
            feats = np.zeros((len(cg), 6 * emb), np.float16)
            probs = np.full((len(cg), 4), np.nan, np.float32)
            nval = np.zeros(len(cg), np.int16)
            for k in range(len(cg)):
                pts = A[k][None] + (B[k] - A[k])[None] * fr[:, None]
                idx = np.array([ref.TransformPhysicalPointToContinuousIndex(tuple(map(float, p))) for p in pts])   # x, y, z
                ok = (idx[:, 0] >= 0) & (idx[:, 0] <= W - 1) & (idx[:, 1] >= 0) & (idx[:, 1] <= H - 1) & (idx[:, 2] >= -0.5) & (idx[:, 2] <= N - 0.5)
                nval[k] = int(ok.sum())
                if not ok.any():
                    continue
                x, y, zz = idx[ok, 0], idx[ok, 1], np.clip(np.round(idx[ok, 2]).astype(int), 0, N - 1)
                col, row = np.clip((x * g / W).astype(int), 0, g - 1), np.clip((y * g / H).astype(int), 0, g - 1)
                parts = []
                for s in ("T2", "ADC", "DWI"):
                    v = grid[s][zz, row, col]                                                         # (points, emb)
                    parts += [v.mean(0), v.max(0)]
                feats[k] = np.concatenate(parts).astype(np.float16)
                yy, xx = np.clip(np.round(y * h / H).astype(int), 0, h - 1), np.clip(np.round(x * w / W).astype(int), 0, w - 1)
                probs[k] = [p_cs[zz, yy, xx].max(), p_cs[zz, yy, xx].mean(), p_any[zz, yy, xx].max(), p_any[zz, yy, xx].mean()]
            if mode != "pilot":
                np.savez_compressed(f"/data/needle_feats/{c}.npz", core_id=cg["Original_Row_ID"].to_numpy(), feats=feats, probs=probs, nvalid=nval)
            else:
                print("   cores", len(cg), "| points inside image per core (median)", int(np.median(nval)), "| own csPCa prob along needle: max over cores %.2f" % np.nanmax(probs[:, 0]), flush=True)
        except Exception as e:
            fails += 1
            print("FAILED", c, str(e)[:200], flush=True)
        if mode != "pilot" and ((i + 1) % 25 == 0 or i + 1 == len(todo)):
            vol.commit()
            print(f"{i + 1}/{len(todo)} done | {(time.time() - t0) / (i + 1):.1f} s per case | failures {fails}", flush=True)
    print("ELAPSED_MINUTES", round((time.time() - t0) / 60, 1), "| feature files on volume", len(glob.glob("/data/needle_feats/*.npz")))


@app.local_entrypoint()
def main(mode: str = "pilot", limit: int = 0):
    run.remote(mode, limit)

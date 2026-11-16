# PROSTATEx replication, step 3: ProViCNet on the matched PROSTATEx examinations. Identical to B4_modal_provicnet.py (same commit, weights,
# settings and code path); only the app and volume names differ.
# Volume prostatex-provicnet-vol:  /inputs/PX_<ProxID>_0000|_0001|_0002.nii.gz + /inputs/PX_<ProxID>_mask.nii.gz  ->  /pred/PX_<ProxID>.nii.gz
#   PYTHONIOENCODING=utf-8 modal run analysis/R3_modal_provicnet_prostatex.py --mode sample
#   PYTHONIOENCODING=utf-8 modal run --detach analysis/R3_modal_provicnet_prostatex.py --mode run
import modal

COMMIT = "1b016ac90e4eb57f34114b33914f21ff079894f9"
app = modal.App("prostatex-provicnet")
vol = modal.Volume.from_name("prostatex-provicnet-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.0.1", "torchvision==0.15.2", "timm==1.0.3", "einops==0.8.0", "safetensors==0.4.3", "SimpleITK==2.3.1", "nibabel==5.1.0",
                 "monai==1.3.1", "opencv-python-headless==4.9.0.80", "numpy==1.24.4", "pandas==2.0.3", "scikit-learn==1.3.2", "umap-learn==0.5.6",
                 "matplotlib==3.7.5", "tqdm", "huggingface-hub==0.26.2", "albumentations==1.4.7", "PyYAML", "icecream", "scipy")
    .run_commands(f"git clone https://github.com/pimed/ProViCNet /opt/ProViCNet && cd /opt/ProViCNet && git checkout {COMMIT}")
)


@app.function(image=image, gpu="T4", cpu=4, memory=16384, timeout=8 * 3600, volumes={"/data": vol})
def run(mode: str = "sample", limit: int = 0):
    import os, sys, glob, time, types
    import numpy as np, torch, SimpleITK as sitk
    from huggingface_hub import hf_hub_download
    os.chdir("/opt/ProViCNet")
    sys.path.insert(0, "/opt/ProViCNet")
    from ProViCNet.ModelArchitectures.Models import GetModel
    from ProViCNet.ModelArchitectures.ProViCNet.ProViCNet import FusionModalities, load_partial_weights
    from ProViCNet.util_functions.Prostate_DataGenerator import US_MRI_Generator
    from ProViCNet.util_functions.utils_weighted import set_seed
    from ProViCNet.util_functions.inference import ProViCNet_Inference, ProViCNet_data_preparation

    # same settings as inference_mpMRI.py defaults, except only_csPCa=True
    args = types.SimpleNamespace(ModelName="ProViCNet", vit_backbone="dinov2_s_reg", img_size=448, nClass=4, nChannel=3, contrastive=True, cuda_device=0,
                                 only_csPCa=True, threshold=0.4, small_batchsize=16, save_folder="/tmp/out/", visualization_folder="/tmp/vis/",
                                 device=torch.device("cuda:0"))
    set_seed(42)
    print("GPU", torch.cuda.get_device_name(0), flush=True)

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
    print("models loaded", flush=True)

    def predict(t2, adc, dwi, gland, cancer):
        gens = {s: US_MRI_Generator(imageFileName=[p], glandFileName=[gland], cancerFileName=[cancer], modality="MRI", cancerTo2=False, Augmentation=False,
                                    img_size=args.img_size) for s, p in (("T2", t2), ("ADC", adc), ("DWI", dwi))}
        prep = ProViCNet_data_preparation(0, args, gens, modality="MRI")
        if prep is False:
            raise RuntimeError("data preparation failed")
        I2, IA, ID, posit, label = prep
        _, _, _, mp = ProViCNet_Inference(I2, IA, ID, posit, args, MODELs, fusion, True)
        p = mp[:, 2].float()                                         # csPCa channel, (slices, h, w)
        ref = sitk.ReadImage(t2)
        W, H, N = ref.GetSize()
        if p.shape[0] != N:
            raise RuntimeError(f"slice count {p.shape[0]} != {N}")
        if tuple(p.shape[1:]) != (H, W):                             # network works at img_size; bring back to the input grid
            p = torch.nn.functional.interpolate(p[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        out = sitk.GetImageFromArray(p.numpy().astype(np.float32))
        out.CopyInformation(ref)
        return out, tuple(mp.shape)

    if mode == "sample":
        d = "/opt/ProViCNet/dataset_MRI/"
        out, shp = predict(d + "sample_t2.nii.gz", d + "sample_adc.nii.gz", d + "sample_dwi.nii.gz", d + "sample_mask.nii.gz", d + "sample_lesions.nii.gz")
        a = sitk.GetArrayFromImage(out)
        les = sitk.GetArrayFromImage(sitk.ReadImage(d + "sample_lesions.nii.gz")) > 0
        gl = sitk.GetArrayFromImage(sitk.ReadImage(d + "sample_mask.nii.gz")) > 0
        pk = np.unravel_index(a.argmax(), a.shape)
        print("raw output shape", shp, "| saved shape", a.shape, "| max prob %.3f at (z,y,x) %s | peak inside annotated lesion: %s" % (a.max(), pk, bool(les[pk])))
        print("mean prob inside lesion %.3f | in gland outside lesion %.3f | outside gland %.4f" % (a[les].mean(), a[gl & ~les].mean(), a[~gl].mean()))
        return
    os.makedirs("/data/pred", exist_ok=True)
    cases = sorted(os.path.basename(f)[:-12] for f in glob.glob("/data/inputs/*_0000.nii.gz"))
    todo = [c for c in cases if not os.path.exists(f"/data/pred/{c}.nii.gz")]
    if limit:
        todo = todo[:limit]
    print("cases on volume", len(cases), "| to run", len(todo), flush=True)
    t0 = time.time()
    fails = 0
    for i, c in enumerate(todo):
        try:
            b = f"/data/inputs/{c}"
            ref = sitk.ReadImage(b + "_0000.nii.gz")
            z = sitk.Image(ref.GetSize(), sitk.sitkUInt8)
            z.CopyInformation(ref)
            sitk.WriteImage(z, "/tmp/zero_cancer.nii.gz")
            out, _ = predict(b + "_0000.nii.gz", b + "_0001.nii.gz", b + "_0002.nii.gz", b + "_mask.nii.gz", "/tmp/zero_cancer.nii.gz")
            sitk.WriteImage(out, f"/data/pred/{c}.nii.gz", True)
        except Exception as e:
            fails += 1
            print("FAILED", c, str(e)[:200], flush=True)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            vol.commit()
            print(f"{i + 1}/{len(todo)} done | {(time.time() - t0) / (i + 1):.1f} s per case | failures {fails}", flush=True)
    print("ELAPSED_MINUTES", round((time.time() - t0) / 60, 1), "| predictions on volume", len(glob.glob("/data/pred/*.nii.gz")))


@app.local_entrypoint()
def main(mode: str = "sample", limit: int = 0):
    run.remote(mode, limit)

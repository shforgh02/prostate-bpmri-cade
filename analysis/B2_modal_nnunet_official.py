# PI-CAI nnU-Net baseline on the PI-CAI-preprocessed UCLA inputs, test-time augmentation ON, T4 GPU.
# Workspace: <your Modal profile> (volume ucla-nnunet-vol holds /imagesTs_official, /results weights, /predictions_official).
#
#   set PYTHONIOENCODING=utf-8
#   python analysis/B2_modal_nnunet_official.py timing          # 20 cases, streams the log, prints minutes per case
#   python analysis/B2_modal_nnunet_official.py full 4          # all remaining cases in 4 parallel sandboxes, returns at once
#   python analysis/B2_modal_nnunet_official.py status          # sandbox states and number of finished cases
#   python analysis/B2_modal_nnunet_official.py stop            # terminate all sandboxes of this job
#
# Pitfalls: MODAL_PROFILE must be the workspace that holds the volume. Never add --disable_augmentation. Cases with an existing .npz are skipped,
# so "full" can be relaunched safely after a failure.
import sys, os, json, time, subprocess
import modal

APP = "ucla-nnunet-official"
GPU = os.environ.get("NNUNET_GPU", "T4")
IDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "B2_sandbox_ids.json")
vol = modal.Volume.from_name("ucla-nnunet-vol")
image = modal.Image.from_registry("joeranbosma/picai_nnunet:latest")
IN, OUT = "/data/imagesTs_official", "/data/predictions_official"


def command(select):
    """select: shell filter choosing case names from /tmp/all.txt"""
    return (
        f"set -e; mkdir -p {OUT} /tmp/in; cd {IN}; "
        "ls | grep '_0000.nii.gz$' | sed 's/_0000.nii.gz$//' | sort > /tmp/all.txt; "
        f"{select} | while read c; do [ -f {OUT}/$c.npz ] || cp ${{c}}_000*.nii.gz /tmp/in/; done; "
        "N=$(ls /tmp/in | grep -c '_0000.nii.gz$' || true); echo CASES_TO_RUN $N; "
        "if [ \"$N\" = \"0\" ]; then echo NOTHING_TO_DO; exit 0; fi; "
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; T0=$(date +%s); "
        "nnunet predict Task2201_picai_baseline --trainer nnUNetTrainerV2_Loss_FL_and_CE_checkpoints "
        "--fold 0,1,2,3,4 --checkpoint model_best --results /data/results "
        f"--input /tmp/in --output {OUT} --store_probability_maps; "
        "T1=$(date +%s); echo ELAPSED_SECONDS $((T1-T0)) FOR_CASES $N; "
        f"echo DONE_TOTAL $(ls {OUT} | grep -c '.npz$')"
    )


def launch(select, timeout):
    app = modal.App.lookup(APP, create_if_missing=True)
    return modal.Sandbox.create("bash", "-c", command(select), image=image, app=app, gpu=GPU, cpu=4, memory=16384,
                                timeout=timeout, volumes={"/data": vol})


mode = sys.argv[1] if len(sys.argv) > 1 else "status"
if mode == "timing":
    sb = launch("head -20 /tmp/all.txt", 3 * 3600)
    print("sandbox", sb.object_id, "GPU", GPU, flush=True)
    t0 = time.time()
    for line in sb.stdout:
        if any(k in line for k in ("CASES_TO_RUN", "ELAPSED", "DONE_TOTAL", "NOTHING", "Tesla", "NVIDIA", "Error", "error", "Traceback")):
            print(line, end="", flush=True)
    sb.wait()
    print("return code", sb.returncode, "| wall minutes", round((time.time() - t0) / 60, 1))
    if sb.returncode != 0:
        print(sb.stderr.read()[-3000:])
elif mode == "full":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ids = [launch(f"awk 'NR % {n} == {k}' /tmp/all.txt", 24 * 3600).object_id for k in range(n)]
    json.dump(ids, open(IDS, "w"))
    print("launched", n, "sandboxes on", GPU, ids)
elif mode in ("status", "stop"):
    ids = json.load(open(IDS)) if os.path.exists(IDS) else []
    for i in ids:
        sb = modal.Sandbox.from_id(i)
        if mode == "stop":
            sb.terminate()
            print(i, "terminated")
        else:
            print(i, "running" if sb.poll() is None else f"finished rc={sb.poll()}")
    r = subprocess.run("modal volume ls ucla-nnunet-vol predictions_official", shell=True, capture_output=True, text=True)
    print("finished cases (.npz on volume):", r.stdout.count(".npz"))

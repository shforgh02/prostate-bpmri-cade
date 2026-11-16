import modal

# SOTA Fix: Add tqdm to the environment so you can monitor progress
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "monai", "pandas", "numpy", "SimpleITK", "scikit-learn", "nibabel", "tqdm")
)

app = modal.App("picai-segresnet-training-new")
volume = modal.Volume.from_name("picai-dataset-vol")

@app.function(
    image=image, 
    gpu="A10G",           
    timeout=28800, # 4 hours
    memory=32768,
    volumes={"/data": volume} 
)
def train_bulletproof_isotropic():
    import os
    import torch
    import numpy as np
    import random
    import pandas as pd
    import warnings
    
    warnings.filterwarnings("ignore", category=UserWarning, module="monai.losses")
    warnings.filterwarnings("ignore", message=".*single channel prediction.*")

    import monai
    # SOTA Fix: Import standard Dataset to prevent RAM exhaustion and keepalive timeouts
    from monai.data import Dataset, DataLoader
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd, 
        CropForegroundd, SpatialPadd, RandCropByPosNegLabeld, 
        RandFlipd, RandAffined, RandAdjustContrastd, 
        RandScaleIntensityd, RandGaussianNoised, EnsureTyped, SelectItemsd
    )
    from monai.networks.nets import SegResNet
    from monai.losses import DiceFocalLoss
    from monai.metrics import DiceMetric
    from monai.utils import set_determinism
    from monai.inferers import sliding_window_inference

    # ---------------------------------------------------------
    # CLOUD PATHS
    # ---------------------------------------------------------
    DATA_DIR = "/data/standardized_picai_sessions_new"
    MASTER_CSV = "/data/picai_master_clinical_cohort_new.csv"
    MODEL_DIR = "/data/models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    MODEL_DICE_PATH = os.path.join(MODEL_DIR, "picai_segresnet_best_dice_new.pth")
    MODEL_LOSS_PATH = os.path.join(MODEL_DIR, "picai_segresnet_best_loss_new.pth")
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "picai_training_checkpoint_new.pth")

    PATCH_SIZE = (32, 64, 64)  
    BATCH_SIZE = 2 
    EPOCHS = 200            
    LEARNING_RATE = 2e-4    

    set_determinism(seed=42)

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    print("🚀 Initializing SOTA 3-Channel SegResNet Cloud Training Loop...")
    df = pd.read_csv(MASTER_CSV)
    
    train_df = df[df['Fold'].isin(['picai_public_images_fold0', 'picai_public_images_fold1', 'picai_public_images_fold2'])]
    val_df = df[df['Fold'] == 'picai_public_images_fold3']

    def create_data_dicts(dataframe):
        data = []
        for _, row in dataframe.iterrows():
            sess_id = row['Session_ID']
            sdir = os.path.join(DATA_DIR, sess_id)
            data.append({
                "t2": os.path.join(sdir, "t2_norm.nii.gz"),
                "adc": os.path.join(sdir, "adc_norm.nii.gz"),
                "dwi": os.path.join(sdir, "dwi_norm.nii.gz"), 
                "pros": os.path.join(sdir, "prostate_mask.nii.gz"),
                "label": os.path.join(sdir, "target_1_mask.nii.gz")
            })
        return data

    train_files = create_data_dicts(train_df)
    val_files = create_data_dicts(val_df)

    # =====================================================================
    # TRANSFORMS
    # =====================================================================
    train_transforms = Compose([
        LoadImaged(keys=["t2", "adc", "dwi", "pros", "label"]),
        EnsureChannelFirstd(keys=["t2", "adc", "dwi", "pros", "label"]),
        RandAdjustContrastd(keys=["t2", "dwi"], prob=0.3, gamma=(0.75, 1.25)),
        RandScaleIntensityd(keys=["t2", "dwi"], factors=0.1, prob=0.2),
        ConcatItemsd(keys=["t2", "adc", "dwi"], name="image"),
        CropForegroundd(keys=["image", "label"], source_key="pros", margin=(15, 15, 15)),
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE, mode="constant"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], 
            label_key="label", 
            spatial_size=PATCH_SIZE, 
            pos=1, 
            neg=1, 
            num_samples=2
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandAffined(
            keys=["image", "label"], 
            prob=0.3, 
            rotate_range=(0.15, 0.15, 0.15), 
            scale_range=(0.1, 0.1, 0.1), 
            mode=("bilinear", "nearest")
        ),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.02),
        EnsureTyped(keys=["image", "label"], data_type="tensor"),
        SelectItemsd(keys=["image", "label"])
    ])

    val_transforms = Compose([
        LoadImaged(keys=["t2", "adc", "dwi", "pros", "label"]),
        EnsureChannelFirstd(keys=["t2", "adc", "dwi", "pros", "label"]),
        ConcatItemsd(keys=["t2", "adc", "dwi"], name="image"),
        CropForegroundd(keys=["image", "label"], source_key="pros", margin=(15, 15, 15)),
        EnsureTyped(keys=["image", "label"], data_type="tensor"),
        SelectItemsd(keys=["image", "label"])
    ])

    # SOTA Fix: Swapping CacheDataset with Standard Dataset to drop RAM usage
    print("⏳ Initializing Standard Datasets (RAM utilization optimized <4GB)...")
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # SOTA Fix: Set num_workers=2 to prevent CPU thread thrashing on GPU nodes
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2) 

    device = torch.device("cuda")
    print(f"✅ Active GPU: {torch.cuda.get_device_name(0)}")

    # =====================================================================
    # MODEL ARCHITECTURE
    # =====================================================================
    model = SegResNet(
        spatial_dims=3, 
        in_channels=3, 
        out_channels=1, 
        init_filters=32, 
        dropout_prob=0.2,
        norm="instance"
    ).to(device)

    loss_function = DiceFocalLoss(include_background=False, sigmoid=True, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    scaler = torch.cuda.amp.GradScaler()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # =====================================================================
    # RESUME & EARLY STOPPING STATE PRESERVATION
    # =====================================================================
    start_epoch = 0
    best_dice = -1.0
    best_loss = float('inf')
    patience = 30
    patience_counter = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n🔄 Checkpoint found at {CHECKPOINT_PATH}! Resuming training...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        best_loss = checkpoint['best_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"   -> Resumed from Epoch {start_epoch+1:03d}/{EPOCHS}")
        print(f"   -> State: Best Dice: {best_dice:.4f} | Best Loss: {best_loss:.4f} | Patience: {patience_counter}/{patience}\n")

    print(f"\n{'='*80}")
    print(f" 🧠 STARTING SEGRESNET TRAINING (3-CHANNEL ISOTROPIC SOTA)")
    print(f"{'='*80}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
        epoch_loss /= step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save state every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_dice': best_dice,
            'best_loss': best_loss,
            'patience_counter': patience_counter
        }, CHECKPOINT_PATH)
        
        # Evaluation every 2 epochs
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_loss_total = 0
            val_steps = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_steps += 1
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    with torch.cuda.amp.autocast():
                        val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 4, model, overlap=0.5, mode="gaussian")
                        v_loss = loss_function(val_outputs, val_labels).item()
                        val_loss_total += v_loss
                        
                    val_outputs_bin = (torch.sigmoid(val_outputs) > 0.5).float()
                    if val_labels.sum() > 0:
                        dice_metric(y_pred=val_outputs_bin, y=val_labels)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                avg_val_loss = val_loss_total / val_steps
                
                scheduler.step(avg_val_loss)
                
                print(f"Epoch {epoch+1:03d}/{EPOCHS} | LR: {current_lr:.6e} | Train Loss: {epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {metric:.4f}")
                
                if metric > best_dice:
                    best_dice = metric
                    torch.save(model.state_dict(), MODEL_DICE_PATH)
                    print(f"  ⭐ New Best Validation Dice: {best_dice:.4f} (Model Saved)")
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0  
                    torch.save(model.state_dict(), MODEL_LOSS_PATH)
                else:
                    patience_counter += 2 
                    
                if patience_counter >= patience:
                    print(f"\n🛑 EARLY STOPPING TRIGGERED: Validation loss has not improved for {patience} epochs.")
                    break
        else:
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | LR: {current_lr:.6e} | Train Loss: {epoch_loss:.4f}")

    print(f"\n✅ TRAINING COMPLETE. Best Validation Dice: {best_dice:.4f} | Best Val Loss: {best_loss:.4f}")

@app.local_entrypoint()
def run():
    train_bulletproof_isotropic.remote()
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp
import torch.nn.functional as F

from post_prop_utils import PostProcessingNet, focal_dice_loss, boundary_loss
from post_prop_utils import compute_multiclass_iou

from torchvision import transforms as v2

from transformers import SamProcessor, SamModel
from tqdm import tqdm

from datasett import CatheterDataset, SAMDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_memory_efficient(
    model,
    model_post_pro,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    num_epochs=50,
    checkpoint_path="best_post_pro_model.pth",
    device="cuda",
    batch_size=None,
    val_frequency=5,
):
    best_iou = 0.0
    best_loss = 2

    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    if batch_size is not None and batch_size < train_loader.batch_size:
        print(
            f"Warning: Batch reduction {train_loader.batch_size} à {batch_size}"
        )

    for epoch in range(num_epochs):
        model_post_pro.train()
        total_loss = 0.0
        batch_count = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for batch in loop:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            images, masks = batch["pixel_values"], batch["ground_truth_mask"]
            images, masks = images.to(device), masks.to(device)

            model.to(device)

            with torch.no_grad():
                initial_outputs = model(images)
                initial_outputs = initial_outputs.pred_masks.squeeze(1)
                probabilities = torch.softmax(
                    initial_outputs, dim=1
                )
                del initial_outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            optimizer.zero_grad()
            post_pro_outputs = model_post_pro(probabilities)

            del probabilities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            loss = focal_dice_loss(post_pro_outputs, masks) + 0.5 * boundary_loss(
                post_pro_outputs, masks
            )
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            del post_pro_outputs, loss, masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_loss += batch_loss
            batch_count += 1
            loop.set_postfix(loss=batch_loss)

        avg_train_loss = total_loss / batch_count
        history["train_loss"].append(avg_train_loss)

        if (
            epoch % val_frequency == 0
            or epoch == num_epochs - 1
            or avg_train_loss < best_loss
        ):
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
            val_loss, val_iou, _ = validate_memory_efficient(
                model, model_post_pro, val_loader, criterion, device
            )
            history["val_loss"].append(val_loss)
            history["val_iou"].append(val_iou)

            if scheduler:
                scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}"
            )
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model_post_pro.state_dict(), checkpoint_path)
                print(
                    f"Bets model weights saved: {best_iou:.4f}"
                )
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} (no validation)"
            )
            history["val_loss"].append(None)
            history["val_iou"].append(None)

            if scheduler and isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                pass
            elif scheduler:
                scheduler.step()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return history


def validate_memory_efficient(
    model, model_post_pro, val_loader, criterion, device="cuda"
):
    model.eval()
    model_post_pro.eval()
    total_loss = 0.0

    all_post_pro_outputs = []
    all_initial_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            images, masks, _ = batch["image"], batch["label"], batch["resized"]
            images, masks = images.to(device), masks.to(device)

            initial_outputs = model(images)

            all_initial_outputs.append(initial_outputs.cpu())

            probabilities = torch.softmax(
                initial_outputs, dim=1
            )

            del initial_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            post_pro_outputs = model_post_pro(probabilities)
            all_post_pro_outputs.append(post_pro_outputs.cpu())

            del probabilities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if masks.shape[1] == 1:
                target_masks = masks.squeeze(1)
            else:
                target_masks = masks

            loss = focal_dice_loss(post_pro_outputs, masks) + 0.5 * boundary_loss(
                post_pro_outputs, masks
            )
            total_loss += loss.item()

            all_targets.append(masks.cpu())

            del post_pro_outputs, loss, target_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_post_pro_outputs = torch.cat(all_post_pro_outputs, dim=0)
    all_initial_outputs = torch.cat(all_initial_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    initial_iou, initial_class_ious = compute_multiclass_iou(
        all_initial_outputs, all_targets, num_classes=3
    )
    post_pro_iou, post_pro_class_ious = compute_multiclass_iou(
        all_post_pro_outputs, all_targets, num_classes=3
    )

    print(f"IoU of initial model: {initial_iou:.4f}")
    for cls, iou in enumerate(initial_class_ious):
        print(f"  - Class {cls}: {iou:.4f}")

    print(f"IoU after post-processing: {post_pro_iou:.4f}")
    for cls, iou in enumerate(post_pro_class_ious):
        print(f"  - Classe {cls}: {iou:.4f}")

    print(
        f"Différence between global IoUs: {post_pro_iou - initial_iou:.4f} ({(post_pro_iou - initial_iou) / initial_iou * 100:.2f}%)"
    )

    for cls in range(len(initial_class_ious)):
        diff = post_pro_class_ious[cls] - initial_class_ious[cls]
        percent = (
            (diff / initial_class_ious[cls] * 100)
            if initial_class_ious[cls] > 0
            else float("inf")
        )
        print(f"  - Class {cls}: {diff:.4f} ({percent:.2f}%)")

    del all_post_pro_outputs, all_initial_outputs, all_targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    val_loss = total_loss / len(val_loader)
    return val_loss, post_pro_iou, initial_iou 


def visualize_memory_efficient(
    model, model_post_pro, val_loader, num_samples=2, device="cuda"
):
    model.eval()
    model_post_pro.eval()

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    samples_seen = 0

    with torch.no_grad():
        for batch in val_loader:
            if samples_seen >= num_samples:
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            images, masks, _ = batch["image"], batch["label"], batch["resized"]
            images, masks = images.to(device), masks.to(device)

            initial_outputs = model(images)
            probabilities = torch.softmax(initial_outputs, dim=1)
            initial_masks = torch.argmax(initial_outputs, dim=1).cpu().numpy()

            del initial_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            post_pro_outputs = model_post_pro(probabilities)
            post_pro_masks = torch.argmax(post_pro_outputs, dim=1).cpu().numpy()

            del probabilities, post_pro_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if masks.shape[1] == 1:
                target_masks = masks.squeeze(1).cpu().numpy()
            else:
                target_masks = torch.argmax(masks, dim=1).cpu().numpy()

            i = 7  # Premier échantillon du batch
            if samples_seen < num_samples:
                axes[samples_seen, 0].imshow(images[i, 0].cpu().numpy(), cmap="gray")
                axes[samples_seen, 0].set_title("Image originale")
                axes[samples_seen, 0].axis("off")

                axes[samples_seen, 1].imshow(initial_masks[i], cmap="jet")
                axes[samples_seen, 1].set_title("Prédiction initiale")
                axes[samples_seen, 1].axis("off")

                axes[samples_seen, 2].imshow(post_pro_masks[i], cmap="jet")
                axes[samples_seen, 2].set_title("Post-traité")
                axes[samples_seen, 2].axis("off")

                axes[samples_seen, 3].imshow(target_masks[i], cmap="jet")
                axes[samples_seen, 3].set_title("Ground Truth")
                axes[samples_seen, 3].axis("off")

                samples_seen += 1

            del images, masks, initial_masks, post_pro_masks, target_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    plt.tight_layout()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return fig


# -------------- SET SEED --------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# -------------- DATASET --------------
TRANSFORM = v2.Compose(
    [
        v2.Resize((256, 256)),
        v2.ToTensor(),
    ]
)
DATA_PATH = "./dataset/"
DATASET = CatheterDataset(DATA_PATH, transform=TRANSFORM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
processor.image_processor.do_rescale = False

train_size = int(0.8 * len(DATASET))
val_size = len(DATASET) - train_size
train_dataset, val_dataset = random_split(DATASET, [train_size, val_size])

train_dataset = SAMDataset(dataset=train_dataset, processor=processor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = SAMDataset(dataset=val_dataset, processor=processor)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

checkpoint_path = "best_SAM_state_CELoss_ALL.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("✅ Modèle chargé avec succès !")

model_post_pro = PostProcessingNet(in_channels=3, out_channels=1, base_filters=32).to(
    device
)

optimizer = torch.optim.Adam(model_post_pro.parameters(), lr=0.0005)  # LR réduit
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=5, factor=0.5
)
criterion = torch.nn.CrossEntropyLoss()

history = train_memory_efficient(
    model,
    model_post_pro,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    num_epochs=30,
    val_frequency=5,
    checkpoint_path="best_post_pro_model_SAM.pth",
)

model_post_pro.load_state_dict(torch.load("best_post_pro_model_SAM.pth"))

fig = visualize_memory_efficient(model, model_post_pro, val_loader, num_samples=2)


# ########################################################
model_post_pro = PostProcessingNet(in_channels=3, out_channels=1, base_filters=32).to(
    device
)
model_post_pro.load_state_dict(torch.load("best_post_pro_model_SAM.pth"))
model = SamModel.from_pretrained()

checkpoint_path = "best_SAM_state_CELoss_ALL.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
fig = visualize_memory_efficient(model, model_post_pro, val_loader, num_samples=5)
plt.show()

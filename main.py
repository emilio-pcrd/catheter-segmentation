import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torchvision import transforms as v2

from transformers import SamProcessor, SamModel
import monai
from tqdm import tqdm

from datasett import CatheterDataset, SAMDataset
from utils import ACLoss


"""
Parses command line arguments for configuring the model and the mode we want to use. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, etc.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description="Catheter Segmentation")
parser.add_argument(
    "--train",
    action="store_false",
    default=False,
    help="finetune the model",
)
parser.add_argument(
    "--eval",
    action="store_false",
    default=False,
    help="evaluate the model",
)
parser.add_argument(
    "--train_all",
    action="store_false",
    default=True,
    help="train the entire model",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="batch size for training",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=20,
    help="number of epochs for training",
)
parser.add_argument(
    "--weights",
    type=str,
    default="flaviagiammarino/medsam-vit-base",
    help="pretrained weights for the model (can choose between sam-vit-base and medsam-vit-base)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="learning rate for training",
)
parser.add_argument(
    "--loss",
    type=str,
    default="DiceCELoss",
    help="loss function for training, choice between the 'Dice', 'DiceCELoss', 'ACLoss', and 'clDice'",
)

args = parser.parse_args()

# -------------- SET SEED --------------
seed = 42
# seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)


# -------------- SETUP --------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr

# -------------- DATASET --------------
TRANSFORM = v2.Compose(
    [
        v2.Resize((256, 256)),
        v2.ToTensor(),
    ]
)
DATA_PATH = "./dataset/"
DATASET = CatheterDataset(DATA_PATH, transform=TRANSFORM)


# -------------- TRAINING CONFIG --------------
model = SamModel.from_pretrained(args.weights)
processor = SamProcessor.from_pretrained(args.weights)
processor.image_processor.do_rescale = False

train_size = int(0.8 * len(DATASET))
val_size = len(DATASET) - train_size
train_dataset, val_dataset = random_split(DATASET, [train_size, val_size])

train_dataset = SAMDataset(dataset=train_dataset, processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = SAMDataset(dataset=val_dataset, processor=processor)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(model.mask_decoder.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Loss functions
if args.loss == "Dice":
    seg_loss = monai.losses.DiceLoss(
        include_background=False,
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
    )
elif args.loss == "DiceCELoss":
    seg_loss = monai.losses.DiceCELoss(
        sigmoid=True, squared_pred=True, reduction="mean"
    )
elif args.loss == "clDice":
    seg_loss = monai.losses.SoftDiceclDiceLoss()
elif args.loss == "ACLoss":
    seg_loss1 = ACLoss(classes=1)
    seg_loss2 = monai.losses.DiceCELoss(
        sigmoid=True, squared_pred=True, reduction="sum"
    )
    seg_loss3 = monai.losses.SoftDiceclDiceLoss()
elif args.loss == "CombinedLoss":
    from utils import CombinedLoss

    seg_loss = CombinedLoss()

from utils import dice_metric

print(args.weights)
# -------------- TRAINING --------------
if args.train:

    from monai.losses.dice import one_hot  # NOQA

    iou_metric = monai.metrics.MeanIoU(include_background=False, reduction="Mean")

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    NUM_EPOCHS = args.num_epochs
    print(f"----------- Number of Epochs: {NUM_EPOCHS} ------------")
    model.to(DEVICE)

    print(f"----------- number of batches: {BATCH_SIZE} ------------")
    print(f"----------- device {DEVICE} ---------------------")

    model.train()

    train_losses = []
    val_losses = []
    best_iou = 0
    for epoch in range(NUM_EPOCHS):
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_iou = []
        epoch_dice_scores = []
        for batch in tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"
        ):
            # forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(DEVICE),
                input_boxes=batch["input_boxes"].to(DEVICE),
                multimask_output=False,
            )
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = (
                batch["ground_truth_mask"].float().to(DEVICE)
            )  # (B, H, W)
            threshold = 0.5
            ground_truth_masks = (ground_truth_masks > threshold).float()

            # compute loss
            foreground_logits = predicted_masks  # (B, 1, H, W)
            background_logits = 1 - predicted_masks  # (B, 1, H, W)
            predicted_masks_2 = torch.cat(
                [background_logits, foreground_logits], dim=1
            )  # (B, 2, H, W)

            gt_onehot = one_hot(
                batch["ground_truth_mask"].long()[:, None, ...],
                num_classes=2,
            )
            gt_onehot = gt_onehot.to(DEVICE)  # (B, 2, H, W)

            if args.loss == "Dice":
                loss = seg_loss(input=predicted_masks_2, target=gt_onehot)
            elif args.loss == "DiceCELoss":
                loss1 = seg_loss(
                    input=predicted_masks, target=ground_truth_masks.unsqueeze(1)
                )
                loss2 = dice_metric(predicted_masks, ground_truth_masks.unsqueeze(1))
                loss = 0.5 * loss1 + 0.5 * (1 - loss2)
            else:
                loss = seg_loss(input=two_channel_logits, target=ground_truth_masks)

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for batch in tqdm(
                val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"
            ):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(DEVICE),
                    input_boxes=batch["input_boxes"].to(DEVICE),
                    multimask_output=False,
                )
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(DEVICE)
                gt_onehot = one_hot(
                    ground_truth_masks[:, None, ...],
                    num_classes=2,
                )
                gt_onehot = gt_onehot.to(DEVICE)

                foreground_logits = predicted_masks  # (B, 1, H, W)
                background_logits = 1 - predicted_masks  # (B, 1, H, W)
                predicted_masks_2 = torch.cat(
                    [background_logits, foreground_logits], dim=1
                )  # (B, 2, H, W)

                pred_probs = torch.sigmoid(
                    outputs.pred_masks.squeeze(1)
                )  # Convert logits to probabilities
                predicted_masks_iou = (pred_probs > 0.5).float()

                foreground_logits_iou = predicted_masks_iou  # (B, 1, H, W)
                background_logits_iou = 1 - predicted_masks_iou  # (B, 1, H, W)
                two_channel_logits_iou = torch.cat(
                    [background_logits_iou, foreground_logits_iou], dim=1
                )  # (B, 2, H, W)
                threshold = 0.5
                ground_truth_masks = (ground_truth_masks > threshold).float()

                if args.loss == "ACLoss":
                    loss1 = seg_loss1(predicted_masks, ground_truth_masks.unsqueeze(1))
                    loss2 = seg_loss2(predicted_masks, ground_truth_masks.unsqueeze(1))
                    loss3 = seg_loss3(predicted_masks, ground_truth_masks.unsqueeze(1))
                    loss = 0.05 * loss1 + 0.45 * loss2 + 0.5 * loss3
                elif args.loss == "Dice":
                    loss = seg_loss(
                        input=predicted_masks_2,
                        target=gt_onehot,
                    )
                elif args.loss == "DiceCELoss":
                    loss1 = seg_loss(
                        input=predicted_masks,
                        target=ground_truth_masks.unsqueeze(1),
                    )
                    loss2 = dice_metric(
                        predicted_masks, ground_truth_masks.unsqueeze(1)
                    )
                    loss = 0.5 * loss1 + 0.5 * (1 - loss2)
                else:
                    loss = seg_loss(
                        input=predicted_masks.squeeze(1),
                        target=batch["ground_truth_mask"].long().to(DEVICE),
                    )

                epoch_val_losses.append(loss.item())
                iou_score = iou_metric(
                    y_pred=two_channel_logits_iou,
                    y=ground_truth_masks.unsqueeze(1),
                )
                epoch_iou.append(iou_score.item())

                epoch_dice_scores.append(
                    dice_metric(predicted_masks, ground_truth_masks.unsqueeze(1)).item()
                )

        train_loss_mean = np.mean(epoch_train_losses)
        val_loss_mean = np.mean(epoch_val_losses)

        train_losses.append(train_loss_mean)
        val_losses.append(val_loss_mean)
        epoch_iou_mean = np.mean(epoch_iou)
        mean_dice_score = sum(epoch_dice_scores) / len(epoch_dice_scores)

        print(
            f"EPOCH {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss_mean:.4f} | Val Loss: {val_loss_mean:.4f} | mIOU: {epoch_iou_mean:.4f} | Dice: {mean_dice_score:.4f}"
        )
        model.train()
        scheduler.step()

        # Save best model state_dict
        if epoch_iou_mean > best_iou:
            best_iou = epoch_iou_mean
            best_model_state = model.state_dict()
            torch.save(best_model_state, "best_SAM_state_CELossDICE_med.pt")

    torch.save(model, "finetuned_sam_DICE+CEorigindataset.pt")
    # -------------- PLOTTING --------------
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("loss_plot.pdf")


if args.train_all:
    iou_metric = monai.metrics.MeanIoU(include_background=True, reduction="mean")

    model.load_state_dict(
        torch.load("best_SAM_state_CELossDICE_med.pt", weights_only=False)
    )
    optimizer = Adam(model.parameters(), lr=0.00003, betas=(0.9, 0.999))
    # make sure we compute gradients for all parameters
    for param in model.parameters():
        param.requires_grad_(True)

    NUM_EPOCHS = 10
    print(f"----------- Training all params of finetuned model... ------------")
    print(f"----------- Number of Epochs: {NUM_EPOCHS} ------------")
    model.to(DEVICE)

    print(f"----------- number of batches: {BATCH_SIZE} ------------")
    print(f"----------- device {DEVICE} ---------------------")

    model.train()

    train_losses = []
    val_losses = []
    best_iou = 0
    for epoch in range(NUM_EPOCHS):
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_dice_scores = []
        epoch_iou = []

        for batch in tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"
        ):
            # forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(DEVICE),
                input_boxes=batch["input_boxes"].to(DEVICE),
                multimask_output=False,
            )

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(DEVICE)
            threshold = 0.5
            ground_truth_masks = (ground_truth_masks > threshold).float()

            loss1 = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            loss2 = dice_metric(predicted_masks, ground_truth_masks.unsqueeze(1))
            loss = 0.5 * loss1 + 0.5 * (1 - loss2)

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for batch in tqdm(
                val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"
            ):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(DEVICE),
                    input_boxes=batch["input_boxes"].to(DEVICE),
                    multimask_output=False,
                )

                # Apply sigmoid and threshold
                pred_probs = torch.sigmoid(
                    outputs.pred_masks
                )  # Convert logits to probabilities

                predicted_masks_iou = (pred_probs > 0.5).float()
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(DEVICE)
                threshold = 0.5
                ground_truth_masks = (ground_truth_masks > threshold).float()
                if args.loss == "ACLoss":
                    loss1 = seg_loss1(predicted_masks, ground_truth_masks.unsqueeze(1))
                    loss2 = seg_loss2(predicted_masks, ground_truth_masks.unsqueeze(1))
                    loss3 = seg_loss3(predicted_masks, ground_truth_masks.unsqueeze(1))
                    loss = 0.05 * loss1 + 0.45 * loss2 + 0.5 * loss3
                else:
                    loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

                epoch_val_losses.append(loss.item())
                iou_score = iou_metric(
                    y_pred=predicted_masks_iou.squeeze(1),
                    y=ground_truth_masks.unsqueeze(1),
                )
                epoch_iou.append(iou_score.item())

                epoch_dice_scores.append(
                    dice_metric(predicted_masks, ground_truth_masks.unsqueeze(1)).item()
                )

        train_loss_mean = np.mean(epoch_train_losses)
        val_loss_mean = np.mean(epoch_val_losses)

        train_losses.append(train_loss_mean)
        val_losses.append(val_loss_mean)
        epoch_iou_mean = np.mean(epoch_iou)
        mean_dice_score = sum(epoch_dice_scores) / len(epoch_dice_scores)

        print(
            f"EPOCH {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss_mean:.4f} | Val Loss: {val_loss_mean:.4f} | mIOU: {epoch_iou_mean:.4f} | Dice: {mean_dice_score:.4f}"
        )
        # Save best model state_dict
        if epoch_iou_mean > best_iou:
            best_iou = epoch_iou_mean
            best_model_state = model.state_dict()
            torch.save(best_model_state, "best_SAM_stateALL_GOOD.pt")

        model.train()

    torch.save(model, "finetuned_sam2_ALL_PARAMS_medsam.pt")

    # -------------- PLOTTING --------------
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("loss_plot_ALL.pdf")

# ------------ EVALUATION --------------
if args.eval:
    from torchvision.utils import save_image
    from monai.losses.dice import one_hot  # NOQA
    from post_prop_utils import postprocess_mask
    os.makedirs("./predictions", exist_ok=True)

    # Define MONAI metrics
    iou_metric = monai.metrics.MeanIoU(include_background=True, reduction="mean")

    model.load_state_dict(
        torch.load("./models_hist/best_SAM_state_CELoss_ALL_medsam.pt", weights_only=False)
    )
    eval_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    model.to(DEVICE)
    model.eval()

    dice_scores = []
    iou_scores = []

    print("Evaluating model...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader)):
            # Get input images and ground truth masks
            images = batch["pixel_values"].to(DEVICE)
            ground_truth_masks = batch["ground_truth_mask"].to(DEVICE)
            input_boxes = batch["input_boxes"].to(DEVICE)

            # Forward pass
            outputs = model(
                pixel_values=images,
                input_boxes=input_boxes,
                multimask_output=False,
            )

            predicted_masks = outputs.pred_masks.squeeze(1)

            # compute for dice:
            foreground_logits = predicted_masks  # (B, 1, H, W)
            background_logits = 1 - predicted_masks  # (B, 1, H, W)
            two_channel_logits = torch.cat(
                [background_logits, foreground_logits], dim=1
            )  # (B, 2, H, W)
            ground_truth_masks_both = one_hot(
                batch["ground_truth_mask"].long()[:, None, ...],
                num_classes=2,
            )
            ground_truth_masks_both = ground_truth_masks_both.to(DEVICE)

            # Apply sigmoid and threshold
            pred_probs = torch.sigmoid(
                outputs.pred_masks
            )  # Convert logits to probabilities
            predicted_masks = (pred_probs > 0.5).float()  # Convert to binary mask
            predicted_masks_post = postprocess_mask(predicted_masks.squeeze())  # Apply post-processing

            # Check if ground_truth_masks are binary
            threshold = 0.5
            ground_truth_masks = (ground_truth_masks > threshold).float()
            ground_truth_masks_binary = ground_truth_masks
            post_prop = False
            if post_prop:
                # Compute Dice and IoU using MONAI
                dice_scores.append(
                    dice_metric(
                        predicted_masks_post.unsqueeze(0).to(DEVICE), ground_truth_masks_binary.unsqueeze(1)
                    ).item()
                )

                iou_score = iou_metric(
                    y_pred=predicted_masks_post.unsqueeze(0).unsqueeze(0).to(DEVICE), y=ground_truth_masks_binary.unsqueeze(1)
                )
            else:
                # Compute Dice and IoU using MONAI
                dice_scores.append(
                    dice_metric(
                        predicted_masks.unsqueeze(1), ground_truth_masks_binary.unsqueeze(1)
                    ).item()
                )

                iou_score = iou_metric(
                    y_pred=predicted_masks.squeeze(1), y=ground_truth_masks_binary.unsqueeze(1)
                )
            iou_scores.append(iou_score.item())

            # Save predicted masks
            for j in range(predicted_masks.shape[0]):  # Iterate over batch
                print(f"Saving prediction {i * eval_dataloader.batch_size + j}")
                save_path = (
                    f"./predictions/pred_mask_{i * eval_dataloader.batch_size + j}.png"
                )
                save_image(predicted_masks[j].cpu(), save_path)

    # Final Scores
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)

    print(f"\nEvaluation Results:")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice Score: {mean_dice:.4f}")

    # Save results
    with open("./predictions/eval_results.txt", "w") as f:
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Mean Dice Score: {mean_dice:.4f}\n")

    print("Predictions and evaluation results saved in ./predictions/")

import os
import cv2
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import exposure, io
from PIL import Image
import transforms as T
from matplotlib.patches import Rectangle
import torch.utils.data
import torchvision.models.segmentation
import torchvision.transforms.functional as F
from engine import train_one_epoch, evaluate
from plot_curve import plot_loss_and_lr, plot_map
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.Custom_datas import platelet_pt, MitoDataset, UroMitoDataset
from natsort import natsorted
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_transform(train):
    transforms = [T.PILToTensor(), T.ConvertImageDtype(torch.float)]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def compute_iou(gt_mask, pred_mask):
    # Convert masks to binary format
    gt_mask = np.where(gt_mask >= 0.5, 1, 0)
    pred_mask = np.where(pred_mask >= 0.5, 1, 0)

    # Compute intersection and union
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)

    # Calculate areas
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def compute_dice_coefficient(gt_annotations, pred_results):
    # Flatten the annotations and predictions
    gt_annotations_flat = gt_annotations.flatten()
    pred_results_flat = pred_results.flatten()

    # Calculate true positives, false positives, and false negatives
    true_positives = np.sum(np.logical_and(pred_results_flat == 255, gt_annotations_flat == 255))
    false_positives = np.sum(np.logical_and(pred_results_flat == 255, gt_annotations_flat == 0))
    false_negatives = np.sum(np.logical_and(pred_results_flat == 0, gt_annotations_flat == 255))
    true_negatives = np.sum(np.logical_and(pred_results_flat == 0, gt_annotations_flat == 0))

    # Calculate the Dice coefficient
    dice_coefficient = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)

    jaccard_foreground = true_positives / (true_positives + false_positives + false_negatives)
    jaccard_background = true_negatives / (true_negatives + false_positives + false_negatives)
    Voc_score = (jaccard_foreground + jaccard_background) / 2.
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    mAP = average_precision_score(gt_annotations_flat, pred_results_flat, pos_label=255)
    f1 = f1_score(gt_annotations_flat, pred_results_flat, pos_label=255)

    return dice_coefficient, jaccard_foreground, jaccard_background, Voc_score, accuracy, precision, recall, mAP, f1, true_positives, false_positives, false_negatives, true_negatives


# UROSCELL MitoDataset
dataset_UROmito = UroMitoDataset("...", get_transform(train=True))
dataset_UROmito_test = UroMitoDataset("...",get_transform(train=False))


# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(dataset_UROmito, batch_size=2, shuffle=True, num_workers=0,
                                                collate_fn=collate_fn)

data_loader_valid = torch.utils.data.DataLoader(dataset_UROmito_test, batch_size=1, shuffle=False,
                                                num_workers=0,
                                                collate_fn=collate_fn)


# Inteference
dataset_mito_test = dataset_UROmito_test
model = get_model_instance_segmentation(num_classes=2)

model.to(device)  # move model to the right device

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# training model parameters
train_loss = []
learning_rate = []
val_map = []

num_epochs = 100
'''
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    metric_logger = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)

    train_loss.append(metric_logger.meters["loss"].global_avg)

    learning_rate.append(metric_logger.meters["lr"].value)
    # print(optimizer.param_groups[0]['lr'])

    # torch.cuda.empty_cache()
    # update the learning rate
    lr_scheduler.step()    # evaluate on the test dataset
    coco_evaluator = evaluate(model, data_loader_valid, device=device)

    val_map.append(coco_evaluator.coco_eval['bbox'].stats[1])
    torch.cuda.empty_cache()

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        plot_map(val_map)

    # if epoch % 1 == 0:
    #     torch.save(model.state_dict(), str(epoch) + ".pt")

torch.save(model.state_dict(), f'_{num_epochs}.pt')
'''


###################################################################
#'''
############################
# Load the trained model
model.load_state_dict(torch.load(
    f'_{num_epochs}.pt'))
model.eval()
#'''

#'''
# # Inference
# pick one image from the test set
img, gtt = dataset_mito_test[89]

# Convert tensor to NumPy array
img_np = img.numpy()

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img_np, clip_limit=0.009)

# Convert the NumPy array to a PyTorch tensor
img_tensor = torch.from_numpy(img_adapteq)

with torch.no_grad():
    pred = model([img.to(device)])

im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
im = np.array(im)

im2 = im.copy()

boxe = pred[0]['boxes'].detach().cpu().numpy()
mas = pred[0]['masks'].detach().cpu().numpy()
sco = pred[0]['scores'].detach().cpu().numpy()
labe = pred[0]['labels'].detach().cpu().numpy()

print("boxes:", boxe)
print("Masks:", mas)
print("Scores:", sco)
print("Labels:", len(labe))

# Create a figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(im)
thresh = 0.7
# Iterate over the predicted boxes and scores
for box, score in zip(pred[0]['boxes'], pred[0]['scores']):
    if score > thresh:
        xmin, ymin, xmax, ymax = box.detach().cpu().numpy()

        # Create a rectangle patch for the bounding box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle patch to the axes
        ax.add_patch(rect)

for i in range(len(pred[0]['masks'])):
    msk = pred[0]['masks'][i, 0].detach().cpu().numpy()
    scr = pred[0]['scores'][i].detach().cpu().numpy()
    if scr > thresh:

        im2[:, :, 0][msk > 0.5] = random.randint(0, 255)
        im2[:, :, 1][msk > 0.5] = random.randint(0, 255)
        im2[:, :, 2][msk > 0.5] = random.randint(0, 255)

# Display the combined image
plt.imshow(np.hstack([im, im2]), cmap="gray")
plt.show()

binary_masks = []
# Iterate over the predicted masks
for i in range(len(pred[0]['masks'])):
    msk = pred[0]['masks'][i, 0].detach().cpu().numpy()
    scr = pred[0]['scores'][i].detach().cpu().numpy()
    if scr > thresh:
        # Threshold the mask to obtain binary values
        binary_mask = (msk > 0.5).astype(np.uint8)
        binary_masks.append(binary_mask)

# Combine all the binary masks into a single image
combined_mask = np.sum(binary_masks, axis=0)

# Display the combined mask
plt.imshow(combined_mask, cmap='gray')
plt.show()

binary_mask = np.zeros_like(im, dtype=np.uint8)[:, :, 0]
for i in range(len(pred[0]['masks'])):
    msk = pred[0]['masks'][i, 0].detach().cpu().numpy()
    scr = pred[0]['scores'][i].detach().cpu().numpy()
    if scr > thresh:
        # Threshold the mask to obtain binary values
        binary_mask_i = (msk > 0.5).astype(np.uint8)

        # Add the binary mask to the combined binary mask
        binary_mask = np.logical_or(binary_mask, binary_mask_i).astype(np.uint8)
plt.imshow(binary_mask, cmap='gray')
plt.show()

# To compute IoU, Dice, mAP, precision, recall, f1, voc, accuracy
# Convert ground truth masks to binary format
binary_gt = np.array(gtt['mask'])

ms_predict = np.array(binary_mask) * 255


dice_coefficient, jaccard_foreground, jaccard_background, Voc_score, accuracy, precision, recall, mAP, f1, true_positives, false_positives, false_negatives, true_negatives = compute_dice_coefficient(
        binary_gt, ms_predict)
print("dice_coefficient:", dice_coefficient, "jaccard_foreground:", jaccard_foreground, "jaccard_background:", jaccard_background, "Voc_score:", Voc_score, "accuracy:", accuracy, "precision:", precision, "recall:", recall)

print("mAP:", mAP, "precision:", "f1:", f1)

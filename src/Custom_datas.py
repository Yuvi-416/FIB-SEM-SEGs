import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from matplotlib.patches import Rectangle
from scipy.ndimage import label
import tifffile as tiff
import cv2

class UroMitoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        # print("idx:", idx)
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "labels", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert the binary mask to non-binary and assign labels
        img_labels, num_objs = label(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(img_labels)
        
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = img_labels == obj_ids[:, None, None]

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmin == xmax or ymin == ymax:
                continue
            else:
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["mask"] = mask
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # print("target:", target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Visualize the image with boxes
        # self.visualize_boxes(img, boxes)

        return img, target

    # def visualize_boxes(self, img, boxes):
    #     # Convert image to numpy array
    #     img = np.array(img)
    #     img = img.transpose((1, 2, 0))
    #
    #     # Create a figure and axes
    #     fig, ax = plt.subplots(1)
    #
    #     # Plot the image
    #     ax.imshow(img)
    #
    #     # Plot the bounding boxes
    #     for box in boxes:
    #         xmin, ymin, xmax, ymax = box
    #         width = xmax - xmin
    #         height = ymax - ymin
    #         rect = Rectangle((xmin, ymin), width, height, fill=False, edgecolor='r')
    #         ax.add_patch(rect)
    #
    #     # Set axis labels and title
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_title('Object Detection')
    #
    #     # Show the plot
    #     plt.show()

    def __len__(self):
        return len(self.imgs)

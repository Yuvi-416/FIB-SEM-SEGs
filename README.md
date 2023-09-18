# Robust Ensemble Approach to Automatic Segmentation of Mitochondria from FIB-SEM Images

In this repository, we have employed the Mask R-CNN model to detect and segment the mitochondrial region of interest from the UroCell dataset. We utilized the `maskrcnn_resnet50_fpn` backbone for building the Mask R-CNN network, but users can easily switch to another pre-trained network if desired.

## Training and Testing

To train your own model with custom data, please use `train_file.py`. This file also provides a convenient way to test your data. You can download the UroCell dataset from (https://github.com/MancaZerovnikMekuc/UroCell).

For our 3D segmentation work, we employed the PyTorch 3D U-Net model available at (https://github.com/wolny/pytorch-3dunet). We've provided a config YAML file for both training and testing. Before using it, ensure that you've installed the `pytorch-3dunet` package on your workstation.

## Results with MASK R-CNN:
![test_fib1-2-3-2_obj_0](https://github.com/Yuvi-416/FIB-SEM-SEGs/assets/65744819/6e064ce0-7ef8-4fac-b9f9-9d2aa54fcb1c)
![valid_pred_image](https://github.com/Yuvi-416/FIB-SEM-SEGs/assets/65744819/8a85daac-f902-40a9-9abf-888490f9286c)

## Results with 3D U-Net:
![3D_unet](https://github.com/Yuvi-416/FIB-SEM-SEGs/assets/65744819/436beacf-1625-405c-a792-374b1e20db66)

## Results with ensemble fusion method:
- Mask R-CNN output:
![mask_rcnn](https://github.com/Yuvi-416/FIB-SEM-SEGs/assets/65744819/4782ebba-082b-4df3-88c9-3be602060be8)

- 3D U-Net output:
![3D_unet](https://github.com/Yuvi-416/FIB-SEM-SEGs/assets/65744819/4f65ca69-750a-497e-a1cf-f06e9298a6f9)

- Proposed ensemble fusion method:
![ensemble_slice_001](https://github.com/Yuvi-416/FIB-SEM-SEGs/assets/65744819/8000bb14-a3b3-450c-85b8-17cecf567cb5)

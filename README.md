# FIB-SEM-SEGs
In this repository, we have used the Mask R-CNN model to detect and segment the mitochondrial region of interest from the UroCell dataset. Here, we used the maskrcnn_resnet50_fpn backbone while building the Mask R-CNN network. The user can change to another pre-trained network if they want.

Please use train_file.py to train the model with your own data, and within the same file, we also provide a way to test the testing data.
The data which I have used can be downloaded from: https://github.com/MancaZerovnikMekuc/UroCell

# Obtained result:

![test_fib1-2-3-2_obj_0](https://github.com/Yuvi-416/FIB-SEM-SEGs/assets/65744819/6e064ce0-7ef8-4fac-b9f9-9d2aa54fcb1c)
![valid_pred_image](https://github.com/Yuvi-416/FIB-SEM-SEGs/assets/65744819/8a85daac-f902-40a9-9abf-888490f9286c)

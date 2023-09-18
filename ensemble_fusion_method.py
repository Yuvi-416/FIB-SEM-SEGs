import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

# Load the predicted images from both models
Model_3D_unet_1 = "..."
Model_Mask_RCNN_2 = "........."

# Create empty lists to store predicted masks
predicted_masks_model1 = []
predicted_masks_model2 = []


# Load predicted masks for each model (replace with actual loading code)
for filename in natsorted(os.listdir(Model_3D_unet_1)):
    if filename.endswith(".tif"):
        mask = cv2.imread(os.path.join(Model_3D_unet_1, filename), cv2.IMREAD_GRAYSCALE)
        predicted_masks_model1.append(mask)

for filename in natsorted(os.listdir(Model_Mask_RCNN_2)):
    if filename.endswith(".tif"):
        mask = cv2.imread(os.path.join(Model_Mask_RCNN_2, filename), cv2.IMREAD_GRAYSCALE)
        predicted_masks_model2.append(mask)

# Convert lists to numpy arrays
predicted_masks_model1 = np.array(predicted_masks_model1)
predicted_masks_model2 = np.array(predicted_masks_model2)


# Evaluation metrics for each model
model_metrics = {
    'model1': {'JIF': 0.9329, 'ACC': 0.9952, 'PRE': 0.9344, 'Recall': 0.9622},
    'model2': {'JIF': 0.9231,  'ACC': 0.9954, 'PRE': 0.9579, 'Recall': 0.9579},
}


# Calculate entropy for each model
def calculate_entropy(metrics):
    probs = [metrics['JIF'], metrics['ACC'], metrics['PRE'], metrics['Recall']]
    probs = np.array(probs)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


# Calculate weights based on entropy
def calculate_weights(model_metrics):
    entropies = []
    for model_name, metrics in model_metrics.items():
        entropy = calculate_entropy(metrics)
        entropies.append(entropy)

    entropies = np.array(entropies)
    weights = (1.0 - entropies) / np.sum(1.0 - entropies)
    weights /= np.sum(weights)  # Normalize weights to sum up to 1

    return weights


weights = calculate_weights(model_metrics)

print("Weights:", weights)

# Define the weights for each model
maskrcnn_weight = weights[0]
unet_weight = weights[1]
threshold = 0.2

# Apply weighted averaging fusion
ensemble_prediction = (maskrcnn_weight * predicted_masks_model1) + (unet_weight * predicted_masks_model2)

print(ensemble_prediction)

# Normalize the ensemble prediction to obtain a binary output
ensemble_prediction = np.where(ensemble_prediction >= threshold, 255, 0)

# Create a directory to save the .png files if it doesn't exist
output_directory = "........."
os.makedirs(output_directory, exist_ok=True)

# Loop through all slices along the z-axis
for z_slice in range(256):
    # Display a single 2D slice from the 3D volume
    plt.imshow(ensemble_prediction[z_slice, :, :], cmap='gray')
    plt.title(f'Ensemble Prediction - Slice {z_slice}')
    plt.colorbar()

    # Save the image as a .png file
    image_filename = os.path.join(output_directory, f'ensemble_slice_{z_slice:03d}.png')
    plt.savefig(image_filename, format='png')

    plt.pause(0.01)  # Add a short pause to allow time for display
    plt.clf()  # Clear the previous plot

# Ensure the last plot is displayed
plt.show()
